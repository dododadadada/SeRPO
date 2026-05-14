"""Entry point for GRPO training. Reads a YAML config and runs offline_trainer."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from grpo.trainer.offline_trainer import GRPOConfig, run_training


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    cfg_dict = yaml.safe_load(args.config.read_text())
    # Convert lora_target_modules from list to tuple if present
    if "lora_target_modules" in cfg_dict and isinstance(cfg_dict["lora_target_modules"], list):
        cfg_dict["lora_target_modules"] = tuple(cfg_dict["lora_target_modules"])
    cfg = GRPOConfig(**cfg_dict)
    run_training(cfg)


if __name__ == "__main__":
    main()
