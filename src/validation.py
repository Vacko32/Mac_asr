"""
Validation utilities.

Author: Martin Vaculik (xvaculm00@stud.fit.vutbr.cz)
"""

import torch
import torch.distributed as dist
import logging
import numpy as np
from collections import defaultdict
from typing import Dict, Any, List, Tuple
from jiwer import process_words
from tqdm import tqdm
from macasr import MacAsr

logger = logging.getLogger(__name__)


class PerDatasetMetrics:
    """Metrics for each dataset tracker"""

    def __init__(self):
        self.errors: Dict[str, int] = defaultdict(int)
        self.words: Dict[str, int] = defaultdict(int)
        self.samples: Dict[str, int] = defaultdict(int)

    def add_wer(self, dataset_name: str, errors: int, words: int):
        self.errors[dataset_name] += errors
        self.words[dataset_name] += words
        self.samples[dataset_name] += 1

    def get_dataset_names(self) -> List[str]:
        return sorted(self.errors.keys())

    def get_wer(self, dataset_name: str) -> float:
        if self.words[dataset_name] == 0:
            return 0.0
        return self.errors[dataset_name] / self.words[dataset_name]

    def get_totals(self) -> Tuple[int, int]:
        """Return (total_errors, total_words)."""
        return (
            sum(self.errors.values()),
            sum(self.words.values()),
        )


def is_distributed() -> bool:
    return dist.is_initialized() and dist.get_world_size() > 1


def all_reduce_scalar(value: float, op=dist.ReduceOp.SUM) -> float:
    """Reduce a scalar value across all ranks."""
    if not is_distributed():
        return value
    tensor = torch.tensor(value, dtype=torch.float64, device="cuda")
    dist.all_reduce(tensor, op=op)
    return tensor.item()


def compute_wer(reference: str, hypothesis: str) -> Tuple[float, int, int]:
    """
    Compute Word Error Rate between reference and hypothesis using jiwer.
    Returns (wer_score, errors, total_words)
    """

    output = process_words(reference, hypothesis)  # jiwer

    errors = output.substitutions + output.insertions + output.deletions
    total_words = output.hits + output.substitutions + output.deletions
    wer_score = output.wer

    return wer_score, errors, total_words


def validate(
    model: MacAsr,
    val_loader: Any,
    device: torch.device,
    config: Dict[str, Any],
    dtype: torch.dtype = torch.float32,
) -> Tuple[float, Dict[str, Any]]:
    """
    Run validation using generation.

    Returns:
        Tuple of (average WER, per_dataset_metrics) aggregated across all GPUs.
    """

    training_config = config["training"]
    model_config = config["model"]
    rank = dist.get_rank() if is_distributed() else 0
    if rank == 0:
        logger.info("Starting validation...")
    metrics = PerDatasetMetrics()

    val_iter = iter(val_loader)
    model.eval()

    unwrapped_model = model.module if hasattr(model, "module") else model

    with torch.no_grad():
        ### If validation_samples == 0 entire ds
        max_samples = training_config["validation_samples"]
        if max_samples == 0:
            max_samples = len(val_loader)
            if rank == 0:
                logger.info(
                    f"validation_samples=0, running on entire validation set ({max_samples} batches)"
                )

        for j in tqdm(
            range(max_samples),
            desc="Validation",
            disable=(rank != 0),  # Only show progress bar on rank 0
        ):
            try:
                batch = next(val_iter)
            except StopIteration:
                break

            wavs = batch["audio"]["array"]
            texts = batch["text"]
            dataset_names = batch.get("dataset_name", ["unknown"] * len(texts))

            # Prepare audio input
            if model_config["encoder"] == "whisper":
                x = torch.tensor(np.stack(wavs), dtype=dtype, device=device)
                lengths = batch["audio"]["lengths"]
                wav_mask = torch.zeros(
                    x.shape[0], x.shape[1], dtype=torch.bool, device=device
                )
                for i, length in enumerate(lengths):
                    wav_mask[i, :length] = True
            else:
                raise ValueError(f"Unsupported encoder: {model_config['encoder']}")

            generated_texts = unwrapped_model.generate(wav=x, wav_mask=wav_mask)

            ### Log samples - only rank 0
            if rank == 0 and j == 0:
                num_to_print = min(4, len(generated_texts), len(texts))
                for i in range(num_to_print):
                    logger.info(f"Generated: {generated_texts[i][:100]}...")
                    logger.info(f"Reference: {texts[i][:100]}...")

            for ref, hyp, ds_name in zip(texts, generated_texts, dataset_names):
                wer, errors, words = compute_wer(ref, hyp)
                metrics.add_wer(ds_name, errors, words)

    ### Get totals for aggregatio
    total_errors, total_words = metrics.get_totals()

    ### Aggregate metrics via all_reduce
    total_errors = all_reduce_scalar(total_errors)
    total_words = all_reduce_scalar(total_words)

    avg_wer = total_errors / total_words if total_words > 0 else 0.0

    per_dataset_metrics = {}
    dataset_names_list = metrics.get_dataset_names()

    if rank == 0:
        logger.info(f"Validation complete:")
        logger.info(
            f"  Overall WER: {avg_wer:.2%} ({int(total_errors)}/{int(total_words)} errors)"
        )

        if len(dataset_names_list) > 1 or (
            len(dataset_names_list) == 1 and dataset_names_list[0] != "unknown"
        ):
            logger.info("  Per-dataset metrics:")
            for ds_name in dataset_names_list:
                ds_wer = metrics.get_wer(ds_name)
                ds_samples = metrics.samples[ds_name]
                logger.info(f"    {ds_name}: WER={ds_wer:.2%}, samples={ds_samples}")

                per_dataset_metrics[f"val_wer/{ds_name}"] = ds_wer

    return avg_wer, per_dataset_metrics
