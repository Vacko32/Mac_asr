"""
Checkpoint save/load utilities.
Author: Martin Vaculik (xvaculm00@stud.fit.vutbr.cz)
"""

import torch
import logging
from pathlib import Path
from typing import Dict, Any

from macasr import MacAsr

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: MacAsr,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    val_loss: float,
    config: Dict[str, Any],
    checkpoint_name: str,
) -> str:
    """
    Save model checkpoint.

    Args:
        model: The model to save
        optimizer: The optimizer to save
        step: Current training step
        loss: Training loss
        val_loss: Validation loss
        config: Configuration dictionary
        checkpoint_name: Name for the checkpoint file

    Returns:
        Path to saved checkpoint
    """
    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / checkpoint_name

    # Handle both DDP-wrapped and regular models
    unwrapped_model = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "step": step,
            "model_state_dict": unwrapped_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "val_loss": val_loss,
        },
        checkpoint_path,
    )

    logger.info(f"Checkpoint saved: {checkpoint_path}")
    return str(checkpoint_path)


def load_checkpoint(
    checkpoint_path: str, model: MacAsr, optimizer: torch.optim.Optimizer = None
) -> Dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into

    Returns:
        Dictionary with checkpoint metadata (step, loss, val_loss)
    """
    checkpoint = torch.load(checkpoint_path)
    unwrapped_model = model.module if hasattr(model, "module") else model
    unwrapped_model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Checkpoint loaded (with optimizer): {checkpoint_path}")
    else:
        logger.info(f"Checkpoint loaded (optimizer reset): {checkpoint_path}")
    return {
        "step": checkpoint["step"] + 2,
        "loss": checkpoint["loss"],
        "val_loss": checkpoint["val_loss"],
    }
