"""

Script for Evaluation one or multiple checkpoints
Configs are passed into top of the file (paths)

Results are going to be saved to new file


Usage:
    python evaluation.py ,,,, Uses hardcoded CONFIG_PATHS and CHECKPOINT_PATHS
    python evaluation.py ,,,, --config_path /path/to/config.toml --checkpoint_path /path/to/checkpoint.pt

Author: Martin Vaculik (xvaculm00@stud.fit.vutbr.cz)
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import torch
from model_init import resolve_token
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from validation import PerDatasetMetrics, compute_wer
from data import load_dataset_from_disk, load_dataset_from_download
from config import load_config
from checkpoint import load_checkpoint
from model_init import initialize_model
from data import collate_fn_whisper, MAX_AUDIO_SAMPLES
from normalizer import EnglishNormalizer

CONFIG_PATHS = ["/storage/brno2/home/xvaculm00/OPTIMALIZED/IBT_project/config.toml"]
CHECKPOINT_PATHS = [
    "/storage/brno2/home/xvaculm00/OPTIMALIZED/IBT_project/src/checkpoint/qwen_checkpoint_12345_step_174000.pt", 
    "/storage/brno2/home/xvaculm00/OPTIMALIZED/IBT_project/src/checkpoint/qwen_checkpoint_12345_step_186000.pt",
    "/storage/brno2/home/xvaculm00/OPTIMALIZED/IBT_project/src/checkpoint/qwen_checkpoint_12345_step_144000.pt"
    "/storage/brno2/home/xvaculm00/OPTIMALIZED/IBT_project/src/checkpoint/qwen_checkpoint_12345_step_122000.pt"
]
RESULT_PATH = "/storage/brno2/home/xvaculm00/OPTIMALIZED/IBT_project/src"

DATASET_NAME_FROM_DISK = [
    "/auto/brno2/home/xvaculm00/A_SINGLE_SPEAKER/big_asr_corpus/validation",
]

# Dictionary mapping dataset names to lists of splits to evalua
# Format: {dataset_name: [split1, split2, ...]}
DATASET_NAME_DOWNLOAD = {
    #"openslr/librispeech_asr": ["test.clean", "test.other"],
}
NUM_OF_BATCHES = 64

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def write_metrics(result_path, dataset_name, corpus_wer, avg_wer):
    with open(result_path, "a") as f:
        f.write(f"{dataset_name} corpus_wer={corpus_wer:.4f} avg_wer={avg_wer:.4f}\n")


NUM_WORKERS = 4


def _run_evaluation_loop(model, dataloader, metrics, normalizer, wer_scores_per_ds, desc, device):
    """Run evaluation on a single dataloader, keying metrics by the 'dataset_name' column."""
    sample_count = 0
    pbar = tqdm(dataloader, desc=desc, dynamic_ncols=True)
    with torch.inference_mode():
        for batch in pbar:
            # Non-blocking transfer from pinned CPU memory to GPU via DMA
            audio = batch["audio"]["array"].to(device, dtype=torch.float32, non_blocking=True)
            output_texts = model.generate(audio, batch["text"])
            for ref, hyp, ds_name in zip(batch["text"], output_texts, batch["dataset_name"]):
                ref_normalized = normalizer(ref)
                hyp_normalized = normalizer(hyp)
                wer_score, errors, total_words = compute_wer(
                    ref_normalized, hyp_normalized
                )
                metrics.add_wer(ds_name, errors, total_words)
                wer_scores_per_ds.setdefault(ds_name, []).append(wer_score)
                sample_count += 1
            pbar.set_postfix(samples=sample_count)


def evaluate(model, result_path, device):
    metrics = PerDatasetMetrics()
    normalizer = EnglishNormalizer()
    wer_scores_per_ds = {}

    # Warmup torch.compile (DownSamplerOptimized uses max-autotune which compiles on first call)
    logger.info("Warming up torch.compile (takes a few seconds on first run)...")
    with torch.inference_mode():
        dummy = torch.zeros(1, MAX_AUDIO_SAMPLES, device=device, dtype=torch.float32)
        model.generate(dummy, [""])
    logger.info("Warmup complete.")

    # Evaluate datasets from disk
    for dataset_path in DATASET_NAME_FROM_DISK:
        logger.info(f"Loading dataset from disk: {dataset_path}")
        dataset = load_dataset_from_disk(dataset_path)
        dataloader = DataLoader(
            dataset,
            batch_size=NUM_OF_BATCHES,
            collate_fn=collate_fn_whisper,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
            multiprocessing_context="spawn",
            prefetch_factor=2,
        )
        _run_evaluation_loop(
            model, dataloader, metrics, normalizer, wer_scores_per_ds,
            desc=f"Evaluating {dataset_path}", device=device,
        )

    # Evaluate datasets from download with multiple splits
    for dataset_name, splits in DATASET_NAME_DOWNLOAD.items():
        for split in splits:
            logger.info(f"Loading dataset {dataset_name} with split {split}")
            dataset = load_dataset_from_download(dataset_name, split=split)
            dataloader = DataLoader(
                dataset,
                batch_size=NUM_OF_BATCHES,
                collate_fn=collate_fn_whisper,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                persistent_workers=True,
                multiprocessing_context="spawn",
                prefetch_factor=2,
            )
            _run_evaluation_loop(
                model, dataloader, metrics, normalizer, wer_scores_per_ds,
                desc=f"Evaluating {dataset_name}:{split}", device=device,
            )

    # Write per-dataset metrics (keyed by the 'dataset_name' column)
    for ds_name in sorted(wer_scores_per_ds.keys()):
        corpus_wer = metrics.get_wer(ds_name)
        scores = wer_scores_per_ds[ds_name]
        avg_wer = sum(scores) / len(scores) if scores else 0.0
        logger.info(
            f"{ds_name}  corpus_wer={corpus_wer:.4f}  avg_wer={avg_wer:.4f}  "
            f"samples={len(scores)}"
        )
        write_metrics(result_path, ds_name, corpus_wer, avg_wer)


def evaluate_checkpoint(config_path, checkpoint_path, result_path):
    config = load_config(config_path)
    from huggingface_hub import login
    hf_token = resolve_token("HF_TOKEN", config, "hf_token")
    login(token=hf_token)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _, _ = initialize_model(config, device)
    load_checkpoint(checkpoint_path, model)
    model.eval()
    model.to(device)
    evaluate(model, result_path, device)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one or multiple checkpoints")
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to a single config file (requires --checkpoint_path)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a single checkpoint file (requires --config_path)",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default=None,
        help="Path to save results (default: RESULT_PATH/result_cli.txt)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.config_path and args.checkpoint_path:
        result_path = args.result_path or RESULT_PATH + "/result_cli.txt"
        logger.info(f"Evaluating single checkpoint: {args.checkpoint_path}")
        evaluate_checkpoint(args.config_path, args.checkpoint_path, result_path)
    elif args.config_path or args.checkpoint_path:
        raise ValueError(
            "Both --config_path and --checkpoint_path must be provided together"
        )
    else:
        # Default behavior
        for i in range(len(CONFIG_PATHS)):
            config_path = CONFIG_PATHS[i]
            checkpoint_path = CHECKPOINT_PATHS[i]
            result_path = RESULT_PATH + f"/result_{i}.txt"
            evaluate_checkpoint(config_path, checkpoint_path, result_path)


if __name__ == "__main__":
    print("Starting evaluation script...")
    main()
    print("Done.")
