"""
Train / fine-tune a cache-aware streaming FastConformer-Transducer on Luganda.

Prerequisites:
  1. python scripts/prepare_data.py                 # manifests
  2. python scripts/build_tokenizer.py              # Luganda SPE tokenizer
  3. python scripts/train_streaming_asr.py model.tokenizer.dir=data/tokenizers/luganda_spe

The encoder is initialised from a pretrained cache-aware streaming model (see
init_from_pretrained_model in the config) and the Luganda decoder/joint are
trained from scratch. Mirrors NeMo's examples/asr/asr_transducer training
script but with Hydra overrides and a streaming config.

Multi-GPU: trainer.devices=-1 (default) uses all GPUs.
"""

import argparse
from pathlib import Path


def train(config_path: str, overrides: list[str]) -> str:
    try:
        import lightning.pytorch as pl
    except ImportError:
        import pytorch_lightning as pl

    try:
        from nemo.collections.asr.models import EncDecRNNTBPEModel
        from nemo.utils.exp_manager import exp_manager
    except ImportError as e:
        raise SystemExit(f"NeMo not installed: {e}")

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    if OmegaConf.is_missing(cfg.model.tokenizer, "dir") or cfg.model.tokenizer.dir in (None, "???"):
        raise SystemExit(
            "Set model.tokenizer.dir to the tokenizer built by build_tokenizer.py, e.g.\n"
            "  python scripts/train_streaming_asr.py model.tokenizer.dir=data/tokenizers/luganda_spe"
        )

    print(OmegaConf.to_yaml(cfg))

    trainer = pl.Trainer(**_resolve_trainer(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager"))

    model = EncDecRNNTBPEModel(cfg=cfg.model, trainer=trainer)
    # Transfers encoder weights from the pretrained streaming model (per config).
    model.maybe_init_from_pretrained_checkpoint(cfg)

    trainer.fit(model)

    if cfg.model.test_ds.manifest_filepath and model.prepare_test(trainer):
        trainer.test(model)

    out_dir = Path(cfg.exp_manager.exp_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "streaming_luganda.nemo"
    model.save_to(str(final))
    print(f"Saved streaming model -> {final}")
    return str(final)


def _resolve_trainer(trainer_cfg):
    """Instantiate the DDP strategy _target_ if present; pass through otherwise."""
    try:
        from nemo.utils.trainer_utils import resolve_trainer_cfg
        return resolve_trainer_cfg(trainer_cfg)
    except Exception:
        from omegaconf import OmegaConf
        return OmegaConf.to_container(trainer_cfg, resolve=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/finetune_streaming_conformer.yaml")
    parser.add_argument("overrides", nargs="*",
                        help="Dotlist overrides, e.g. model.tokenizer.dir=... trainer.devices=2")
    args = parser.parse_args()
    train(args.config, args.overrides)
