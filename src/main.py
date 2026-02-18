"""
MAC ASR Training - Entry Point

This is the main entry point for training the MAC ASR model.
Run with: python main.py --config ../config.toml
"""
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import argparse
import subprocess
import logging
import torch.multiprocessing as mp

from config import load_config
from model_init import setup_authentication
from trainer import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("training.log")],
)
logger = logging.getLogger(__name__)


def get_gpu_count():
    """Get GPU count"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        return len(result.stdout.strip().split("\n"))
    except Exception:
        return 0


def main(config_path: str = "config.toml") -> None:
    """Main entry point."""
    try:
        config = load_config(config_path)
        setup_authentication(config)
        world_size = get_gpu_count()
        logger.info(f"World size: {world_size}")
        if world_size == 0:
            raise RuntimeError("No GPUs found. Check nvidia-smi output.")
        logger.info(f"Starting training on {world_size} GPUs")
        mp.spawn(train, args=(config, world_size), nprocs=world_size, join=True)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train MAC ASR model")
    parser.add_argument(
        "--config",
        type=str,
        default="../config.toml",
        help="Path to configuration TOML file",
    )
    args = parser.parse_args()

    main(args.config)
