"""
Author: Martin Vaculik (xvaculm00@stud.fit.vutbr.cz)
Dataset loading and data processing utilities for MAC ASR Framework.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
from datasets import concatenate_datasets, load_dataset, load_from_disk

logger = logging.getLogger(__name__)

# Constants
SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 30
MAX_AUDIO_SAMPLES = MAX_AUDIO_SECONDS * SAMPLE_RATE


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def load_datasets(config: Dict[str, Any]) -> Tuple[Any, Any]:
    """
    Load datasets provided in config
    """

    dataset_config = config["dataset"]
    training_config = config.get("training", {})
    max_val_samples = training_config.get("max_validation_samples", 0)

    rank = get_rank()
    # rank 0 loads
    if rank == 0:
        logger.info(f"Loading datasets length: {len(dataset_config['name'])}")

    datasets_loaded = []
    validation_splits = dataset_config.get("validation_split", [])

    for idx in range(len(dataset_config["name"])):
        has_val_split = idx < len(validation_splits) and validation_splits[idx].strip()

        if dataset_config["load_from_disk"][idx]:
            dataset = load_from_disk(dataset_config["name"][idx])
            val_paths = dataset_config.get("load_from_disk_validation_path", [])
            if idx < len(val_paths) and val_paths[idx].strip():
                dataset_val = load_from_disk(val_paths[idx])
            else:
                dataset_val = None
        else:
            dataset = load_dataset(
                dataset_config["name"][idx], split=dataset_config["split"][idx]
            )
            if has_val_split:
                dataset_val = load_dataset(
                    dataset_config["name"][idx], split=validation_splits[idx]
                )
            else:
                dataset_val = None

        if dataset_config.get("mapping_file") and dataset_config["mapping_file"][idx]:
            raise NotImplementedError("Mapping file not implemented")
        datasets_loaded.append((dataset, dataset_val))

    ### Rename ds and ds_val coll, 'transcript' to text
    ### This needs to be edited in order to process different datasets
    ### So far we support only 'text' and 'audio' colls.
    datasets_loaded = [
        (
            ds.rename_column("transcription", "text")
            if "transcription" in ds.column_names
            else ds,
            ds_val.rename_column("transcription", "text")
            if ds_val and "transcription" in ds_val.column_names
            else ds_val,
        )
        for ds, ds_val in datasets_loaded
    ]

    dataset = None
    dataset_val = None

    for i, (ds, ds_val) in enumerate(datasets_loaded):
        train_columns = ["audio", "text"]  # TODO: Change this can be loaded in config
        if "dataset_name" in ds.column_names:
            train_columns.append("dataset_name")
        ds = ds.select_columns(train_columns)
        dataset = ds if i == 0 else concatenate_datasets([dataset, ds])
        if ds_val is not None:
            # Keep dataset_name column if exists ( U can also add yours )
            val_columns = ["audio", "text"]
            if "dataset_name" in ds_val.column_names:
                val_columns.append("dataset_name")
            ds_val = ds_val.select_columns(val_columns)
            dataset_val = (
                ds_val
                if dataset_val is None
                else concatenate_datasets([dataset_val, ds_val])
            )

    if dataset_val is None:
        if rank == 0:
            logger.warning("No validation datasets loaded")
    else:
        # Shuffle validation dataset
        dataset_val = dataset_val.shuffle(seed=42)

        # Subset to max_validation_samples if configured
        if max_val_samples > 0 and len(dataset_val) > max_val_samples:
            if rank == 0:
                logger.info(
                    f"Subsetting validation: {len(dataset_val)} -> {max_val_samples} samples"
                )
            dataset_val = dataset_val.select(range(max_val_samples))

    dataset = dataset.shuffle(seed=42)
    if rank == 0:
        logger.info("Datasets loaded successfully")
    return dataset, dataset_val


### Used for evaluation
def load_dataset_from_disk(dataset_name: str) -> Any:
    """Load dataset from disk."""
    dataset = load_from_disk(dataset_name)
    if "transcription" in dataset.column_names:
        dataset = dataset.rename_column("transcription", "text")
    columns = ["audio", "text"]
    if "dataset_name" in dataset.column_names:
        columns.append("dataset_name")
    dataset = dataset.select_columns(columns)
    return dataset


def load_dataset_from_download(dataset_name: str, split: str = None) -> Any:
    """Load dataset from download."""
    dataset = load_dataset(dataset_name, split=split)
    if "transcription" in dataset.column_names:
        dataset = dataset.rename_column("transcription", "text")
    columns = ["audio", "text"]
    if "dataset_name" in dataset.column_names:
        columns.append("dataset_name")
    dataset = dataset.select_columns(columns)
    return dataset


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    This function is NOT padding to 30 seconds automatically.
    Used for diffeerent encoders than Whisper.
    """
    wavs = []
    texts = []
    dataset_names = []

    for ex in batch:
        wav = ex["audio"]["array"]
        if len(wav) > MAX_AUDIO_SAMPLES:
            wav = wav[:MAX_AUDIO_SAMPLES]
        wavs.append(wav)
        texts.append(ex["text"])
        dataset_names.append(ex.get("dataset_name", "unknown"))

    return {"audio": {"array": wavs}, "text": texts, "dataset_name": dataset_names}


def collate_fn_whisper(batch: List[Dict]) -> Dict[str, Any]:
    """Collate batch for Whisper"""
    wavs = []
    lengths = []  # Track original lengths for proper attention masking
    texts = []
    dataset_names = []

    for ex in batch:
        wav = np.array(ex["audio"]["array"])
        original_length = len(wav)  # Save length BEFORE padding
        if len(wav) > MAX_AUDIO_SAMPLES:
            wav = wav[:MAX_AUDIO_SAMPLES]
            original_length = MAX_AUDIO_SAMPLES  # Clamp length if truncated
        elif len(wav) < MAX_AUDIO_SAMPLES:
            wav = np.pad(wav, (0, MAX_AUDIO_SAMPLES - len(wav)), mode="constant")
        wavs.append(wav)
        lengths.append(original_length)
        texts.append(ex["text"])
        dataset_names.append(ex.get("dataset_name", "unknown"))

    return {
        "audio": {"array": torch.from_numpy(np.stack(wavs)), "lengths": lengths},
        "text": texts,
        "dataset_name": dataset_names,
    }


###  Deprecated

# def load_text_mapping(json_path: str) -> Dict[str, str]:
#     """Load text mapping from JSONL file."""
#     mapping = {}
#     with open(json_path, "r") as f:
#         for line in f:
#             entry = json.loads(line)
#             mapping[entry["original_text"]] = entry["edited_text"]
#     return mapping


# def create_remap_function(mapping: Dict[str, str]):
#     """Create a remap function for dataset.map()."""
#     def remap_text(example):
#         example["text"] = mapping.get(example["text"], example["text"])
#         return example
#     return remap_text
