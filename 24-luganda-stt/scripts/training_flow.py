"""
Metaflow training pipeline for Luganda STT.

Stages:
  prepare   → download + format Common Voice Luganda
  lm        → build SymSpell frequency dictionary + n-gram LM
  finetune  → fine-tune Canary-1B-Flash on Luganda with LoRA
  evaluate  → compute WER on held-out test set
  register  → save best checkpoint + push metadata card

Run locally:
  python scripts/training_flow.py run

Run on AWS Batch / Kubernetes (add @batch / @kubernetes decorators):
  python scripts/training_flow.py --with batch run

Resume after a failed step:
  python scripts/training_flow.py resume
"""

from __future__ import annotations

from metaflow import (
    FlowSpec,
    Parameter,
    Resources,
    current,
    step,
)


class LugandaSTTFlow(FlowSpec):

    # ── Parameters ──────────────────────────────────────────────────────────────

    canary_model = Parameter(
        "canary-model",
        help="Pre-trained Canary checkpoint to fine-tune from",
        default="nvidia/canary-1b-flash",
    )
    max_samples = Parameter(
        "max-samples",
        help="Max training examples to use (use smaller numbers for quick experiments)",
        type=int,
        default=5000,
    )
    max_epochs = Parameter(
        "max-epochs",
        type=int,
        default=20,
    )
    freeze_encoder = Parameter(
        "freeze-encoder",
        help="Freeze the acoustic encoder during fine-tuning (recommended for small data)",
        type=bool,
        default=True,
    )
    boost_factor = Parameter(
        "boost-factor",
        help="GPU-PB word boost factor at inference time",
        type=float,
        default=15.0,
    )
    beam_size = Parameter(
        "beam-size",
        type=int,
        default=8,
    )
    ngram_order = Parameter(
        "ngram-order",
        type=int,
        default=5,
    )

    # ── Steps ───────────────────────────────────────────────────────────────────

    @step
    def start(self):
        """Validate environment and resolve paths."""
        import os
        from pathlib import Path

        print(f"Run ID     : {current.run_id}")
        print(f"Model      : {self.canary_model}")
        print(f"Max samples: {self.max_samples}")
        print(f"Max epochs : {self.max_epochs}")
        print(f"Freeze enc : {self.freeze_encoder}")

        # Soft-check for NeMo — warn early rather than fail deep in the flow
        try:
            import nemo
            print(f"NeMo version: {nemo.__version__}")
        except ImportError:
            print("WARNING: NeMo not installed. Install with: pip install nemo_toolkit[asr]")

        self.output_dir = "outputs/canary_luganda"
        self.next(self.prepare)

    # ── Step 1: Data preparation ─────────────────────────────────────────────────

    @step
    def prepare(self):
        """Download and format Common Voice Luganda into NeMo manifests."""
        import json
        from pathlib import Path

        import soundfile as sf

        manifest_dir = Path("data/manifests")

        try:
            from datasets import load_dataset
            print("Downloading Common Voice Luganda (lg)...")
            ds = load_dataset(
                "mozilla-foundation/common_voice_17_0",
                "lg",
                split="train",
                trust_remote_code=True,
            )
            print(f"  Loaded {len(ds)} raw examples.")
        except Exception as e:
            print(f"  Could not load from HuggingFace: {e}")
            print("  Using stub manifests for pipeline testing.")
            ds = None

        manifest_dir.mkdir(parents=True, exist_ok=True)
        audio_dir = manifest_dir / "audio"
        audio_dir.mkdir(exist_ok=True)

        if ds is not None:
            splits = {"train": 0.8, "dev": 0.1, "test": 0.1}
            n = min(self.max_samples, len(ds))
            rows = list(ds.select(range(n)))

            train_end = int(n * 0.8)
            dev_end   = int(n * 0.9)
            split_rows = {
                "train": rows[:train_end],
                "dev":   rows[train_end:dev_end],
                "test":  rows[dev_end:],
            }

            self.manifest_paths = {}
            for split, items in split_rows.items():
                out_path = manifest_dir / f"{split}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    for i, row in enumerate(items):
                        audio = row["audio"]
                        wav_path = audio_dir / f"{split}_{i:06d}.wav"
                        sf.write(str(wav_path), audio["array"], audio["sampling_rate"])
                        duration = len(audio["array"]) / audio["sampling_rate"]
                        record = {
                            "audio_filepath": str(wav_path.resolve()),
                            "duration": round(duration, 2),
                            "text": row["sentence"].strip(),
                            # Canary has no Luganda token: map onto the English
                            # prompt slot (its BPE already covers Latin-script text).
                            "taskname": "asr",
                            "pnc": "no",
                            "source_lang": "en",
                            "target_lang": "en",
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"  {split}: {len(items)} examples → {out_path}")
                self.manifest_paths[split] = str(out_path)
        else:
            # Stub manifests so the flow can continue end-to-end
            self.manifest_paths = {s: str(manifest_dir / f"{s}.json") for s in ("train", "dev", "test")}
            for split, path in self.manifest_paths.items():
                Path(path).write_text("", encoding="utf-8")

        self.next(self.build_lm)

    # ── Step 2: Language model ────────────────────────────────────────────────────

    @step
    def build_lm(self):
        """
        Build SymSpell spell-checker dictionary and n-gram LM for NGPU-LM fusion.
        These run independently of fine-tuning and can be used at inference time
        even without a fine-tuned checkpoint.
        """
        import json
        import re
        import subprocess
        from collections import Counter
        from pathlib import Path

        # --- Spell dictionary ---
        counter: Counter = Counter()

        def tokenise(text: str):
            return re.findall(r"[a-zA-ZÀ-ÿ']+", text.lower())

        # Collect from manifests
        for split_path in self.manifest_paths.values():
            p = Path(split_path)
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    try:
                        counter.update(tokenise(json.loads(line).get("text", "")))
                    except json.JSONDecodeError:
                        pass

        # Boost word-list vocabulary (domain words should never be flagged as errors)
        wl_dir = Path("data/word_lists")
        if wl_dir.exists():
            for txt in wl_dir.glob("*.txt"):
                for line in txt.read_text(encoding="utf-8").splitlines():
                    word = line.strip().lower()
                    if word and not word.startswith("#"):
                        counter[word] += 1000

        spell_dir = Path("data/spell")
        spell_dir.mkdir(parents=True, exist_ok=True)
        spell_dict_path = spell_dir / "luganda_frequency.txt"
        filtered = {w: c for w, c in counter.items() if len(w) >= 2 and c >= 2}
        with open(spell_dict_path, "w", encoding="utf-8") as f:
            for word, count in sorted(filtered.items(), key=lambda x: -x[1]):
                f.write(f"{word} {count}\n")
        print(f"Spell dictionary: {len(filtered):,} words → {spell_dict_path}")
        self.spell_dict_path = str(spell_dict_path)

        # --- N-gram LM (KenLM) ---
        ngram_dir = Path("outputs/ngram_lm")
        ngram_dir.mkdir(parents=True, exist_ok=True)

        # Merge available text corpora
        corpus_parts = []
        for split_path in self.manifest_paths.values():
            p = Path(split_path)
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    try:
                        text = json.loads(line).get("text", "").strip()
                        if text:
                            corpus_parts.append(text)
                    except json.JSONDecodeError:
                        pass
        for txt in Path("data/ngram").glob("*.txt") if Path("data/ngram").exists() else []:
            corpus_parts.extend(txt.read_text(encoding="utf-8").splitlines())

        corpus_path = ngram_dir / "corpus.txt"
        corpus_path.write_text("\n".join(corpus_parts), encoding="utf-8")
        print(f"LM corpus: {len(corpus_parts):,} lines")

        arpa_path = ngram_dir / f"luganda_{self.ngram_order}gram.arpa"
        if corpus_parts and len(corpus_parts) > 10:
            try:
                subprocess.run(
                    ["lmplz", "-o", str(self.ngram_order),
                     "--discount_fallback",
                     "--prune", "0", "0", "1", "1", "1",
                     "--text", str(corpus_path),
                     "--arpa", str(arpa_path)],
                    check=True, timeout=600,
                )
                print(f"N-gram LM → {arpa_path}")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"KenLM not available or failed ({e}). Skipping LM training.")
        else:
            print("Not enough corpus text to train LM. Skipping.")

        self.ngram_lm_path = str(arpa_path) if arpa_path.exists() else ""
        self.next(self.finetune)

    # ── Step 3: Fine-tuning ───────────────────────────────────────────────────────

    @Resources(gpu=1, memory=24000)
    @step
    def finetune(self):
        """
        Fine-tune Canary-1B-Flash on Luganda (full or encoder-frozen).

        Canary/EncDecMultiTaskModel has no LoRA/PEFT support, so we fine-tune the
        pretrained weights directly. exp_manager provides checkpointing and the
        final model is exported explicitly with save_to().

        @Resources requests 1 GPU + 24 GB RAM on remote schedulers (Batch/K8s).
        When running locally, decorators are ignored.
        """
        from pathlib import Path

        try:
            try:
                import lightning.pytorch as pl
            except ImportError:
                import pytorch_lightning as pl
            from nemo.collections.asr.models import EncDecMultiTaskModel
            from nemo.utils.exp_manager import exp_manager
            from omegaconf import OmegaConf, open_dict
        except ImportError as e:
            print(f"NeMo not installed ({e}). Skipping fine-tuning.")
            self.checkpoint_path = ""
            self.next(self.evaluate)
            return

        cfg = OmegaConf.load("configs/finetune_canary.yaml")

        # Override config from flow parameters
        cfg.model.train_ds.manifest_filepath = self.manifest_paths["train"]
        cfg.model.validation_ds.manifest_filepath = self.manifest_paths["dev"]
        cfg.model.freeze_encoder = self.freeze_encoder
        cfg.trainer.max_epochs = self.max_epochs
        cfg.exp_manager.exp_dir = self.output_dir
        cfg.init_from_pretrained_model = self.canary_model

        trainer = pl.Trainer(**cfg.trainer)
        model = EncDecMultiTaskModel.from_pretrained(cfg.init_from_pretrained_model, trainer=trainer)

        with open_dict(model.cfg):
            model.cfg.train_ds = cfg.model.train_ds
            model.cfg.validation_ds = cfg.model.validation_ds
            model.cfg.optim = cfg.model.optim
        model.setup_training_data(model.cfg.train_ds)
        model.setup_validation_data(model.cfg.validation_ds)
        model.setup_optimization(model.cfg.optim)

        if cfg.model.get("freeze_encoder"):
            model.encoder.freeze()
            print("Encoder frozen for fine-tuning.")

        exp_manager(trainer, cfg.get("exp_manager"))
        trainer.fit(model)

        # Export the fine-tuned model to a single .nemo we can restore later.
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        final_path = out_dir / "canary_luganda.nemo"
        model.save_to(str(final_path))
        self.checkpoint_path = str(final_path)
        print(f"Saved fine-tuned model -> {final_path}")
        self.next(self.evaluate)

    # ── Step 4: Evaluation ────────────────────────────────────────────────────────

    @step
    def evaluate(self):
        """Compute WER on the test set using the fine-tuned (or base) checkpoint."""
        from pathlib import Path

        try:
            import nemo.collections.asr as nemo_asr
        except ImportError:
            print("NeMo not available. Skipping evaluation.")
            self.wer = self.cer = None
            self.next(self.validate_components)
            return

        test_manifest = self.manifest_paths.get("test", "")
        if not test_manifest or not Path(test_manifest).exists():
            print("No test manifest. Skipping WER/CER evaluation.")
            self.wer = self.cer = None
            self.next(self.validate_components)
            return

        ckpt = self.checkpoint_path
        if ckpt and Path(ckpt).exists():
            model = nemo_asr.models.EncDecMultiTaskModel.restore_from(ckpt)
        else:
            print("No fine-tuned checkpoint. Evaluating base model.")
            model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained(self.canary_model)

        model.eval()

        import json
        from src.metrics import corpus_cer, corpus_wer

        records = [
            json.loads(l)
            for l in Path(test_manifest).read_text(encoding="utf-8").splitlines()
            if l.strip()
        ][:200]  # cap at 200 for speed

        if not records:
            self.wer = self.cer = None
            self.next(self.validate_components)
            return

        audio_paths = [r["audio_filepath"] for r in records]
        refs = [r["text"] for r in records]
        hyps = model.transcribe(
            audio=audio_paths, source_lang="en", target_lang="en", pnc=False,
        )
        hyp_texts = [h.text if hasattr(h, "text") else str(h) for h in hyps]

        self.wer = round(corpus_wer(refs, hyp_texts) * 100, 2)
        self.cer = round(corpus_cer(refs, hyp_texts) * 100, 2)
        print(f"WER on {len(records)} test examples: {self.wer}%  CER: {self.cer}%")
        self.next(self.validate_components)

    # ── Step 5: Register ──────────────────────────────────────────────────────────

    @step
    def validate_components(self):
        """
        Smoke-test each pipeline component independently after training.
        Reports which components loaded successfully.
        """
        import struct
        import wave
        import io

        # Generate 2 s of silence for testing
        sr = 16000
        n = sr * 2
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(struct.pack(f"<{n}h", *([0] * n)))
        wav_bytes = buf.getvalue()

        from src.config import Settings
        from src.pipeline import STTPipeline

        s = Settings(
            finetuned_checkpoint=self.checkpoint_path or None,
            spell_dict_path=self.spell_dict_path,
        )
        pipeline = STTPipeline(s, mode="fast")

        status = {}
        try:
            pipeline.load()
            status["pipeline"] = "ok"
        except Exception as e:
            status["pipeline"] = str(e)

        for name, obj in [
            ("vad", pipeline._vad),
            ("diarizer", pipeline._diarizer),
            ("asr", pipeline._asr),
            ("spell_checker", pipeline._spell),
            ("punct", pipeline._punct),
            ("itn", pipeline._itn),
        ]:
            status[name] = "loaded" if getattr(obj, "_model", True) is not None else "fallback"

        print("Component status:")
        for k, v in status.items():
            print(f"  {k}: {v}")

        self.component_status = status
        self.next(self.register)

    @step
    def register(self):
        """Save a metadata card for this training run."""
        import json
        from pathlib import Path

        card = {
            "run_id": current.run_id,
            "canary_model": self.canary_model,
            "max_samples": self.max_samples,
            "max_epochs": self.max_epochs,
            "freeze_encoder": self.freeze_encoder,
            "boost_factor": self.boost_factor,
            "beam_size": self.beam_size,
            "ngram_order": self.ngram_order,
            "checkpoint_path": self.checkpoint_path,
            "spell_dict_path": self.spell_dict_path,
            "ngram_lm_path": self.ngram_lm_path,
            "wer_pct": self.wer,
            "cer_pct": getattr(self, "cer", None),
            "component_status": getattr(self, "component_status", {}),
        }

        card_path = Path("outputs") / f"run_{current.run_id}_card.json"
        card_path.parent.mkdir(parents=True, exist_ok=True)
        card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
        print(f"Model card → {card_path}")
        print(json.dumps(card, indent=2))
        self.next(self.end)

    @step
    def end(self):
        print("Training pipeline complete.")
        if self.wer is not None:
            print(f"Final WER: {self.wer}%")
        if self.checkpoint_path:
            print(f"Checkpoint: {self.checkpoint_path}")


if __name__ == "__main__":
    LugandaSTTFlow()
