"""
Model and optimizer initialization.
"""

import os
import torch
import wandb
import logging
from typing import Dict, Any, Tuple, Optional, Union
from transformers import Wav2Vec2FeatureExtractor
from huggingface_hub import login

from macasr import MacAsr


logger = logging.getLogger(__name__)


def resolve_token(env_var: str, config: Dict[str, Any], config_key: str) -> str:
    """Resolve a token from environment variable first, then config [auth] section.

    Raises SystemExit if neither source provides the token.
    """
    token = os.getenv(env_var)
    if token:
        logger.info(f"Using {env_var} from environment variable")
        return token

    token = config.get("auth", {}).get(config_key, "")
    if token:
        logger.info(f"Using {config_key} from config [auth] section")
        return token

    logger.error(
        f"{env_var} not set and '{config_key}' not found in config [auth]. "
        f"Set the {env_var} environment variable or add '{config_key}' to [auth] in your config."
    )
    raise SystemExit(1)


def setup_authentication(config: Dict[str, Any]) -> None:
    """Setup HuggingFace and W&B authentication.

    Checks environment variables first (HF_TOKEN, WANDB_KEY),
    then falls back to config [auth] section. Exits if neither is set.
    """
    hf_token = resolve_token("HF_TOKEN", config, "hf_token")
    login(token=hf_token)
    logger.info("Logged into HuggingFace Hub")

    if config.get("wandb", {}).get("enabled", True):
        wandb_key = resolve_token("WANDB_KEY", config, "wandb_key")
        wandb.login(key=wandb_key)
        logger.info("Logged into Weights & Biases")
    else:
        logger.info("Weights & Biases tracking disabled, skipping login")


def initialize_model(
    config: Dict[str, Any], device: torch.device
) -> Tuple[MacAsr, Optional[Wav2Vec2FeatureExtractor], str, torch.dtype]:
    """Initialize the model

    Returns:
        Tuple of (model, processor, eos_token, dtype).
        processor is None for Whisper encoder (not needed),
        Wav2Vec2FeatureExtractor for WavLM.
    """
    model_config = config["model"]
    encoder_name = model_config["encoder"]

    processor = None  # not needed for Whisper, hardcoded

    dtype_str = model_config.get("dtype", "bfloat16")
    dtype = getattr(torch, dtype_str)

    model = MacAsr(
        epsilon=model_config["downsample_factor"],
        device=device,
        dtype=dtype,
        freeze_encoder=model_config["freeze_encoder"],
        freeze_llm=not model_config.get("use_lora", False),
        use_lora=model_config.get("use_lora", False),
        encoder_name=encoder_name,
        model=model_config["llm"],
        connector_type=model_config["connector"],
        num_of_beams=model_config["num_of_beams"],
    )

    eos_token = model.model.eos_token
    logger.info(f"Model initialized successfully on {device} with dtype={dtype_str}")

    return model, processor, eos_token, dtype


def initialize_optimizer(
    model: torch.nn.Module, config: Dict[str, Any]
) -> torch.optim.AdamW:
    """Initialize PyTorch AdamW optimizer for trainable (Connector) parameters."""
    training_config = config["training"]
    opt_cfg = config["optimizer"]

    # Collect only trainable parameters (should be Connector only)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if not trainable_params:
        raise ValueError("No trainable parameters found!")

    # Count trainable parameters
    total_trainable = sum(p.numel() for p in trainable_params)
    logger.info(f"Trainable parameters: {total_trainable:,}")

    for name, p in model.named_parameters():
        if p.requires_grad:
            logger.debug(
                f"Trainable: {name} shape={tuple(p.shape)} numel={p.numel():,}"
            )

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=training_config["learning_rate"],
        betas=(opt_cfg["beta1"], opt_cfg["beta2"]),
        eps=opt_cfg.get("eps", 1e-8),
        weight_decay=opt_cfg.get("weight_decay", 0.01),
        fused=True,
    )

    logger.info(
        f"AdamW optimizer initialized: {len(trainable_params)} param tensors, "
        f"{total_trainable:,} total params, lr={training_config['learning_rate']}"
    )

    return optimizer


def initialize_wandb(config: Dict[str, Any]) -> Optional[Any]:
    """Initialize Weights & Biases tracking."""
    if not config.get("wandb", {}).get("enabled", True):
        logger.info("Weights & Biases tracking disabled")
        return None

    model_config = config["model"]
    training_config = config["training"]

    run = wandb.init(
        project=config["wandb"]["project"],
        config={
            "learning_rate": training_config["learning_rate"],
            "architecture": model_config["encoder"] + model_config["llm"],
            "downsample_factor": model_config["downsample_factor"],
            "dataset": config["dataset"]["name"],
            "batch_size": training_config["batch_size"],
            "frozen_encoder": model_config["freeze_encoder"],
            "used_lora": model_config["use_lora"],
        },
    )

    logger.info(f"Weights & Biases initialized: project={config['wandb']['project']}")
    return run
