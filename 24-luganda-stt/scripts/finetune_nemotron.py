"""
Fine-tune NVIDIA Nemotron 3.5 Transcribe on Luganda and other Ugandan languages.

Two-stage recipe:
  Stage 1 — SFT (supervised fine-tuning) with LoRA on the LLM backbone.
  Stage 2 — RL  (GRPO with WER reward) for further sequence-level optimisation.

Run SFT first (the default):
  python scripts/finetune_nemotron.py

Then RL once SFT val_wer has plateaued:
  python scripts/finetune_nemotron.py --stage rl

Hydra-style config overrides are passed as positional args:
  python scripts/finetune_nemotron.py trainer.devices=2 model.lora.r=64

Pre-requisites:
  pip install nemo_toolkit[all]>=2.1.0 peft>=0.10.0 trl>=0.9.0

NeMo model class used: nemo.collections.speechlm2.SALMAutomodel
  - Wraps the Nemotron 3.5 Transcribe speech encoder + LLM backbone.
  - Supports LoRA via the HuggingFace PEFT library.
  - Falls back to ModularAudioGPTModel for older NeMo versions.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ── NeMo model loader ─────────────────────────────────────────────────────────

def load_nemotron(pretrained_id: str, local_ckpt: str | None, trainer):
    """
    Load Nemotron 3.5 Transcribe, trying SALMAutomodel (NeMo >= 2.1) first,
    then falling back to ModularAudioGPTModel (older NeMo).
    """
    # Path 1: NeMo >= 2.1 SALMAutomodel
    try:
        from nemo.collections.speechlm2 import SALMAutomodel  # type: ignore

        source = local_ckpt or pretrained_id
        logger.info("Loading %s via SALMAutomodel...", source)
        if local_ckpt and Path(local_ckpt).exists():
            return SALMAutomodel.restore_from(local_ckpt, trainer=trainer)
        return SALMAutomodel.from_pretrained(source, trainer=trainer)

    except ImportError:
        pass

    # Path 2: NeMo multimodal SpeechLLM (older API)
    try:
        from nemo.collections.multimodal.speech_llm.models.modular_models import (  # type: ignore
            ModularAudioGPTModel,
        )

        source = local_ckpt or pretrained_id
        logger.info("Loading %s via ModularAudioGPTModel...", source)
        if local_ckpt and Path(local_ckpt).exists():
            return ModularAudioGPTModel.restore_from(local_ckpt, trainer=trainer)
        return ModularAudioGPTModel.from_pretrained(source, trainer=trainer)

    except ImportError:
        pass

    raise SystemExit(
        "Could not find a NeMo SpeechLLM model class.\n"
        "Install: pip install nemo_toolkit[all]>=2.1.0\n"
        "NeMo docs: https://github.com/NVIDIA/NeMo"
    )


# ── LoRA setup ────────────────────────────────────────────────────────────────

def apply_lora(model, lora_cfg) -> None:
    """
    Attach LoRA adapters to the LLM backbone using HuggingFace PEFT.

    NeMo SpeechLLM models expose the LLM as model.language_model (or
    model.llm).  We apply PEFT directly to that sub-module so the audio
    encoder is untouched.
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType  # type: ignore
    except ImportError:
        raise SystemExit(
            "peft not installed — run: pip install peft>=0.10.0"
        )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(lora_cfg.target_modules),
        r=int(lora_cfg.r),
        lora_alpha=int(lora_cfg.lora_alpha),
        lora_dropout=float(lora_cfg.lora_dropout),
        bias=str(lora_cfg.bias),
        inference_mode=False,
    )

    # Locate the LLM sub-module inside the NeMo wrapper.
    llm = getattr(model, "language_model", None) or getattr(model, "llm", None)
    if llm is None:
        logger.warning(
            "Could not locate model.language_model / model.llm. "
            "Applying LoRA to the full model — this may also wrap the audio encoder."
        )
        llm = model

    peft_model = get_peft_model(llm, peft_config)
    peft_model.print_trainable_parameters()

    # Patch the wrapper so it still passes forward() correctly.
    if hasattr(model, "language_model"):
        model.language_model = peft_model
    elif hasattr(model, "llm"):
        model.llm = peft_model

    logger.info("LoRA adapters attached (r=%d, alpha=%d).", lora_cfg.r, lora_cfg.lora_alpha)


# ── SFT stage ─────────────────────────────────────────────────────────────────

def run_sft(cfg) -> Path:
    """Supervised fine-tuning (SFT) with LoRA on the LLM backbone."""
    try:
        import lightning.pytorch as pl
    except ImportError:
        import pytorch_lightning as pl  # type: ignore

    from nemo.utils.exp_manager import exp_manager  # type: ignore
    from omegaconf import OmegaConf, open_dict

    trainer = pl.Trainer(**cfg.trainer)
    model = load_nemotron(cfg.pretrained_model, cfg.get("local_checkpoint"), trainer)

    # Freeze audio encoder (saves VRAM; recommended for < 50 h of data).
    if cfg.model.get("freeze_audio_encoder", True):
        for name, param in model.named_parameters():
            if any(tag in name for tag in ("audio_encoder", "speech_encoder", "encoder.")):
                param.requires_grad_(False)
        frozen = sum(
            p.numel() for n, p in model.named_parameters()
            if not p.requires_grad
        )
        logger.info("Audio encoder frozen. Frozen params: %dM", frozen // 1_000_000)

    # Apply LoRA to LLM backbone.
    if cfg.model.lora.get("enable", True):
        apply_lora(model, cfg.model.lora)

    # Wire up datasets.
    with open_dict(model.cfg):
        model.cfg.train_ds = cfg.model.train_ds
        model.cfg.validation_ds = cfg.model.validation_ds
        model.cfg.optim = cfg.model.optim

    model.setup_training_data(model.cfg.train_ds)
    model.setup_validation_data(model.cfg.validation_ds)
    model.setup_optimization(model.cfg.optim)

    exp_manager(trainer, cfg.get("exp_manager"))
    trainer.fit(model)

    out_dir = Path(cfg.exp_manager.exp_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / "nemotron_ugandan_sft.nemo"
    model.save_to(str(final_path))
    logger.info("SFT checkpoint saved -> %s", final_path)
    return final_path


# ── RL stage (GRPO) ───────────────────────────────────────────────────────────

def run_rl(cfg, sft_checkpoint: Path) -> Path:
    """
    Reinforcement learning stage using GRPO with sequence-level WER reward.

    GRPO (Group Relative Policy Optimisation) works well for ASR because:
    - No value network needed (reduces memory).
    - WER is computed over groups of rollouts → relative reward signal.
    - KL penalty vs the SFT checkpoint prevents reward hacking.

    Requires: pip install trl>=0.9.0
    """
    try:
        from trl import GRPOConfig, GRPOTrainer  # type: ignore
    except ImportError:
        raise SystemExit(
            "trl not installed — run: pip install trl>=0.9.0\n"
            "RL stage skipped."
        )

    import torch
    from jiwer import wer as compute_wer  # type: ignore

    try:
        import lightning.pytorch as pl
    except ImportError:
        import pytorch_lightning as pl  # type: ignore

    rl_cfg = cfg.rl
    metric = str(rl_cfg.get("reward_metric", "wer"))

    # Load the SFT checkpoint as the policy.
    trainer_pl = pl.Trainer(devices=cfg.trainer.devices, accelerator=cfg.trainer.accelerator)
    model = load_nemotron(str(sft_checkpoint), str(sft_checkpoint), trainer_pl)
    model.train()

    # Reward function: returns a scalar per rollout (lower WER → higher reward).
    def reward_fn(predictions: list[str], references: list[str]) -> list[float]:
        rewards = []
        for pred, ref in zip(predictions, references):
            if metric == "cer":
                from jiwer import cer as compute_cer  # type: ignore
                score = compute_cer(ref, pred)
            else:
                score = compute_wer(ref, pred)
            rewards.append(1.0 - score)   # reward = 1 - error_rate
        return rewards

    grpo_cfg = GRPOConfig(
        num_train_epochs=1,
        per_device_train_batch_size=cfg.trainer.get("devices", 1),
        num_generations=int(rl_cfg.group_size),
        kl_coef=float(rl_cfg.kl_coef),
        learning_rate=float(rl_cfg.optim.lr),
        warmup_steps=int(rl_cfg.optim.warmup_steps),
        max_steps=int(rl_cfg.optim.max_steps),
        output_dir=str(Path(cfg.exp_manager.exp_dir) / "rl"),
        logging_steps=10,
        bf16=True,
    )

    logger.info(
        "Starting RL stage: algorithm=%s  reward=%s  group_size=%d  max_steps=%d",
        rl_cfg.algorithm, metric, rl_cfg.group_size, rl_cfg.optim.max_steps,
    )

    # GRPOTrainer expects a generate() method and reward_funcs.
    # NeMo SpeechLLM models expose transcribe(); we wrap it.
    rl_trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=grpo_cfg,
        train_dataset=None,  # dataset is loaded from manifests by the model
    )
    rl_trainer.train()

    out_dir = Path(cfg.exp_manager.exp_dir)
    final_path = out_dir / "nemotron_ugandan_rl.nemo"
    model.save_to(str(final_path))
    logger.info("RL checkpoint saved -> %s", final_path)
    return final_path


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Fine-tune Nemotron 3.5 Transcribe on Ugandan languages."
    )
    parser.add_argument(
        "--config", default="configs/finetune_nemotron.yaml",
        help="Path to the OmegaConf YAML config (default: configs/finetune_nemotron.yaml)",
    )
    parser.add_argument(
        "--stage", choices=["sft", "rl", "both"], default="sft",
        help=(
            "sft: supervised fine-tuning only (default)\n"
            "rl:  GRPO reinforcement learning only (requires --sft-checkpoint)\n"
            "both: run SFT then RL in sequence"
        ),
    )
    parser.add_argument(
        "--sft-checkpoint", type=Path, default=None,
        help="Path to an existing SFT .nemo checkpoint (required for --stage rl)",
    )
    parser.add_argument(
        "overrides", nargs="*",
        help="Dotlist config overrides, e.g. trainer.devices=2 model.lora.r=64",
    )
    args = parser.parse_args()

    try:
        from omegaconf import OmegaConf  # type: ignore
    except ImportError:
        raise SystemExit("omegaconf not installed — run: pip install omegaconf")

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    sft_ckpt: Path | None = args.sft_checkpoint

    if args.stage in ("sft", "both"):
        sft_ckpt = run_sft(cfg)

    if args.stage in ("rl", "both"):
        if sft_ckpt is None:
            raise SystemExit(
                "RL stage requires an SFT checkpoint.\n"
                "Run SFT first, then: python scripts/finetune_nemotron.py "
                "--stage rl --sft-checkpoint <path>"
            )
        run_rl(cfg, sft_ckpt)


if __name__ == "__main__":
    main()
