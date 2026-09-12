"""
Fine-tune Canary-1B-Flash on Luganda speech using NeMo's AED model.

Steps:
  1. python scripts/prepare_data.py           # download + format speech data
  2. python scripts/finetune_canary.py        # run this script

The script reads configs/finetune_canary.yaml. Canary/EncDecMultiTaskModel does
NOT support LoRA adapters, so this performs a normal fine-tune of the pretrained
weights (optionally with a frozen encoder). Checkpoints and the final .nemo are
written under exp_manager.exp_dir (outputs/canary_luganda/ by default).

Multi-GPU: pass Hydra-style overrides, e.g. `trainer.devices=2`.
"""

import argparse
from pathlib import Path


def train(config_path: str, overrides: list[str]) -> str:
    try:
        import lightning.pytorch as pl
    except ImportError:  # older NeMo bundled pytorch_lightning directly
        import pytorch_lightning as pl

    try:
        from nemo.collections.asr.models import EncDecMultiTaskModel
        from nemo.utils.exp_manager import exp_manager
    except ImportError as e:
        raise SystemExit(f"NeMo not installed: {e}")

    from omegaconf import OmegaConf, open_dict

    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    print(OmegaConf.to_yaml(cfg))

    trainer = pl.Trainer(**cfg.trainer)

    # Load the pretrained Canary. Fine-tuning re-uses the full architecture +
    # tokenizer from the checkpoint; we only swap in our data and optimiser.
    if cfg.get("init_from_nemo_model"):
        print(f"Restoring from local .nemo: {cfg.init_from_nemo_model}")
        model = EncDecMultiTaskModel.restore_from(cfg.init_from_nemo_model, trainer=trainer)
    else:
        print(f"Loading pretrained model: {cfg.init_from_pretrained_model}")
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

    out_dir = Path(cfg.exp_manager.exp_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / "canary_luganda.nemo"
    model.save_to(str(final_path))
    print(f"Saved fine-tuned model -> {final_path}")
    return str(final_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/finetune_canary.yaml")
    parser.add_argument("overrides", nargs="*",
                        help="Dotlist overrides, e.g. trainer.devices=2 model.freeze_encoder=false")
    args = parser.parse_args()
    train(args.config, args.overrides)
