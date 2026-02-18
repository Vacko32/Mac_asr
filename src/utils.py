"""
Utility functions for training.
"""

import torch
import logging
import math

### Author: Martin Vaculik (xvaculm00@stud.fit.vutbr.cz)
logger = logging.getLogger(__name__)


def calculate_grad_norm(model: torch.nn.Module) -> float:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm**0.5


def format_number_with_comma(num: float) -> str:
    return f"{int(num):,}"


def setup_logging(log_file: str = "training.log") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )
    return logging.getLogger(__name__)


def debug_log_batch(step, texts, targets, source_text, ref_text, x, wav_mask, dataset_names, tokenizer):
    """Log batch data for debugging."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"DEBUG STEP {step}")
    logger.info(f"{'=' * 60}")
    logger.info(f"Input text[0]: '{texts[0]}'")
    logger.info(f"Target text[0]: '{targets[0]}'")
    logger.info(f"source_text shape: {source_text.shape}")
    logger.info(f"source_text[0]: {source_text[0].tolist()}")
    logger.info(f"source_text[0] decoded: '{tokenizer.decode(source_text[0])}'")
    logger.info(f"ref_text shape: {ref_text.shape}")
    logger.info(f"ref_text[0]: {ref_text[0].tolist()}")
    logger.info(f"ref_text[0] decoded: '{tokenizer.decode(ref_text[0])}'")
    logger.info(f"wav shape: {x.shape}, wav_mask shape: {wav_mask.shape}")
    for i in range(min(len(dataset_names), 4)):
        logger.info(f"batch dataset name: {dataset_names[i]}")


def debug_log_logits(step, logits_full, logits_sliced, ref_text, p_c, tokenizer):
    """Log logit shapes and predictions for debugging."""
    logger.info(f"p_c (prefix length): {p_c}")
    logger.info(f"Full logits shape: {logits_full.shape}")
    logger.info(f"Sliced logits shape (after p_c): {logits_sliced.shape}")
    logger.info(f"ref_text shape: {ref_text.shape}")
    logger.info(
        f"logits seq_len: {logits_sliced.shape[1]}, ref_text seq_len: {ref_text.shape[1]}"
    )
    if logits_sliced.shape[1] != ref_text.shape[1]:
        logger.warning(f"SHAPE MISMATCH! logits seq_len != ref_text seq_len")
    pred_ids = logits_sliced[0].argmax(dim=-1)
    logger.info(f"Predicted token IDs[0]: {pred_ids.tolist()[:20]}...")
    logger.info(f"Target token IDs[0]: {ref_text[0].tolist()[:20]}...")
    logger.info(f"Predicted decoded[0]: '{tokenizer.decode(pred_ids[:20])}'")
    logger.info(f"Target decoded[0]: '{tokenizer.decode(ref_text[0][:20])}'")


def learning_rate_schedule(t, lrmax, lrmin, t_warmup_c, t_cosine_c):
    """
    Cosine learning rate scheduler
    """
    if lrmax < 0 or lrmin < 0:
        raise ValueError("Learning rate cannot be negative number")
    if t < t_warmup_c:
        return t / t_warmup_c * lrmax

    elif t_warmup_c <= t <= t_cosine_c:
        return lrmin + 0.5 * (
            1 + math.cos((math.pi * (t - t_warmup_c)) / (t_cosine_c - t_warmup_c))
        ) * (lrmax - lrmin)
    elif t > t_cosine_c:
        return lrmin
    else:
        raise Exception("Something went wrong with the scheduler")
