"""
Main training loop for MAC ASR Framework.

Author: Martin Vaculik (xvaculm00@stud.fit.vutbr.cz)
"""

import logging
import os
import time
from typing import Dict, Any
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from checkpoint import save_checkpoint, load_checkpoint
from data import load_datasets, collate_fn, collate_fn_whisper
from model_init import initialize_model, initialize_optimizer, initialize_wandb
from utils import calculate_grad_norm, format_number_with_comma, learning_rate_schedule, debug_log_batch, debug_log_logits
from validation import validate
import torch.multiprocessing as mp
logger = logging.getLogger(__name__)


### Profiler Setup 
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

def ddp_setup(rank: int, world_size: int) -> None:
    """Initialize distributed data parallel training."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    torch.set_float32_matmul_precision("high")
    torch.distributed.init_process_group(
        backend="nccl", rank=rank, world_size=world_size
    )


def train(rank: int, config: Dict[str, Any], world_size: int) -> None:
    """Main training function."""
    ddp_setup(rank, world_size)
    # Device setup - use the specific GPU assigned to this rank
    device = torch.device(f"cuda:{rank}")
    logger.info(f"Rank {rank}: Using device: {device}")

    DEBUG = config["training"].get("debug", False)

    DEBUG_LOG_EVERY = config["training"].get("debug_log_every", 0)
    if DEBUG:
        logger.info("=" * 60)
        logger.info(
            f"DEBUG MODE ENABLED - Verbose logging every {DEBUG_LOG_EVERY} steps"
        )
        logger.info("=" * 60)

    ### Initialize
    dataset, dataset_val = load_datasets(config)
    model, processor, eos_token, dtype = initialize_model(config, device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = initialize_optimizer(model, config)

    ### Only initialize wandb on rank 0 to avoid duplicate logging
    ### TODO be careful because logs are only happening on rank 0
    run = initialize_wandb(config) if rank == 0 else None

    ### Extract config
    training_config = config["training"]
    early_stopping_config = config["early_stopping"]
    model_config = config["model"]
    loss_config = config["loss"]
    grad_accum = training_config.get("gradient_accumulation", 1)

    steps_per_epoch = len(dataset) // (
        training_config["batch_size"] * world_size * grad_accum
    )

    total_steps = steps_per_epoch * training_config["epochs"]
    logger.info(
        f"Dataset size: {len(dataset)}, Steps per epoch: {steps_per_epoch}, Total steps: {total_steps}"
    )

    best_val_wer = float("inf")
    no_improve_epochs = 0
    last_ckpt_path = None
    start_step = 0

    ### Resume
    if training_config.get("resume", False):
        resume_path = training_config.get("resume_checkpoint", "")
        if resume_path and os.path.exists(resume_path):
            reset_optimizer = training_config.get("reset_optimizer", False)
            reset_steps = training_config.get("reset_steps", False)
            reset_early_stopping = training_config.get("reset_early_stopping", False)

            checkpoint_data = load_checkpoint(
                resume_path, model, optimizer=None if reset_optimizer else optimizer
            )

            start_step = 0 if reset_steps else checkpoint_data["step"]

            if reset_early_stopping:
                best_val_wer = float("inf")
                logger.info(
                    "Early stopping reset - model will get fresh validation chances"
                )
            else:
                best_val_wer = checkpoint_data.get("val_loss", float("inf"))

            logger.info(
                f"Resumed from step {start_step}, best_val_wer: {best_val_wer:.4f}"
            )
            if reset_optimizer:
                logger.info(
                    f"Using fresh optimizer with LR: {training_config['learning_rate']}"
                )
        else:
            raise ValueError(f"Resume enabled but checkpoint not found: {resume_path}")

    logger.info(
        f"Starting training: {total_steps} total steps (from step {start_step})"
    )

    ### Setup data loaders
    batch_collate_fn = (
        collate_fn_whisper if model_config["encoder"] == "whisper" else collate_fn
    )
    num_workers = training_config.get("num_workers", 4)
    train_loader = DataLoader(
        dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        collate_fn=batch_collate_fn,
        sampler=torch.utils.data.distributed.DistributedSampler(dataset),
    )
    val_loader = None
    if dataset_val is not None:
        val_loader = DataLoader(
            dataset_val,
            batch_size=training_config["batch_size"],
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            collate_fn=batch_collate_fn,
            sampler=torch.utils.data.distributed.DistributedSampler(dataset_val),
        )

    model.train()

    train_iter = iter(train_loader)
    step = start_step

    pbar = tqdm(
        total=total_steps,
        initial=start_step,
        desc="Training",
    )

    pad_token_id = model.module.model.tokenizer.pad_token_id
    logger.debug(f"Using pad_token_id={pad_token_id} as ignore_index for loss")

    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=pad_token_id, label_smoothing=loss_config["label_smoothing"]
    )

    lr_schedule_config = config["lr_schedule"]




    ### Initialize of profiler 
    
    ### ============================================================================================== ###
    ### ============================== TRAINING LOOP ================================================ ###
    ### ============================================================================================== ###
    try:
        logger.info(f"Training loop started")
        while step < total_steps:
            step_start_time = time.time()

            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            ### Process batch
            wavs = batch["audio"]["array"]
            texts = batch["text"]
            dataset_names = batch["dataset_name"]  # can be unknown
            targets2 = [text + eos_token for text in texts]

            ### we assume whisper only
            x = torch.from_numpy(np.stack(wavs)).to(device=device, dtype=dtype)
            lengths = batch["audio"]["lengths"]
            max_len = x.shape[1]
            lengths_t = torch.tensor(lengths, dtype=torch.long, device=device)
            wav_mask = torch.arange(max_len, device=device).unsqueeze(
                0
            ) < lengths_t.unsqueeze(1)

            tokenizer = model.module.model.tokenizer
            source_batch = tokenizer(
                texts, padding=True, return_tensors="pt", add_special_tokens=False
            )
            source_text = source_batch["input_ids"]
            source_mask = source_batch["attention_mask"]

            ref_batch = tokenizer(
                targets2, padding=True, return_tensors="pt", add_special_tokens=False
            )
            ref_text = ref_batch.input_ids.to(device)

            ### Prepend last token from prompt because shift aligment
            last_token_id = model.module.model.last_token_id
            if last_token_id is None:
                raise ValueError(
                    "last_token_id is None. Model must set last_token_id "
                    "(the last token from prompt2) for proper shift alignment."
                )

            ### Prepend last_token_id to source_text and ref_text
            batch_size = source_text.shape[0]

            last_token_coll = torch.full(
                (batch_size, 1), fill_value=last_token_id, dtype=source_text.dtype
            )

            source_text = torch.cat([last_token_coll, source_text], dim=1)
            source_mask = torch.cat(
                [torch.ones(batch_size, 1, dtype=source_mask.dtype), source_mask], dim=1
            )

            y = source_text.to(device)

            text_mask = source_mask.to(device, dtype=torch.bool)

            # DEBUG: Print shapes and sample data every N steps
            if DEBUG and step % DEBUG_LOG_EVERY == 0:
                debug_log_batch(step, texts, targets2, source_text, ref_text, x, wav_mask, dataset_names, tokenizer)

            t_warmup = lr_schedule_config.get("warmup_steps", 0)
            lrmin = lr_schedule_config.get("min_lr", 0.0)

            current_lr = learning_rate_schedule(
                step,
                training_config["learning_rate"],
                lrmin,
                t_warmup,
                total_steps,
            )
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            ### Gradient accumulation setup
            grad_accum_steps = training_config.get("gradient_accumulation", 1)
            is_accumulating = (step + 1) % grad_accum_steps != 0

            ### we zero at start of accumul
            if step % grad_accum_steps == 0:
                optimizer.zero_grad()

            ### Forward pass

            prop = model(y, wav=x, wav_mask=wav_mask, text_mask=text_mask)

            ### p_c -> point of calculation, logits only across our text
            p_c = model.module.model.p_c
            logits = prop.logits[:, p_c:, :]

            # DEBUG: Check shapes before loss
            if DEBUG and step % DEBUG_LOG_EVERY == 0:
                debug_log_logits(step, prop.logits, logits, ref_text, p_c, tokenizer)

            loss = loss_fn(logits.reshape(-1, logits.size(-1)), ref_text.reshape(-1))
            loss_for_logging = loss.item()

            scaled_loss = loss / grad_accum_steps
            if is_accumulating:
                with model.no_sync():
                    scaled_loss.backward()
            else:
                scaled_loss.backward()

            if not is_accumulating:
                # we take step
                if loss_config.get("gradient_clip_norm", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), loss_config["gradient_clip_norm"]
                    )
                optimizer.step()

            step_time = time.time() - step_start_time
            if step % training_config["log_interval"] == 0:
                grad_norm = calculate_grad_norm(model)
                num_tokens = text_mask.sum().item()
                tokens_per_sec = num_tokens / step_time if step_time > 0 else 0
            else:
                grad_norm = 0.0
                tokens_per_sec = 0

            metrics = {
                "grad_norm": grad_norm,
                "tokens_per_sec": tokens_per_sec,
                "step_time": step_time,
            }

            ### Logging
            if step % training_config["log_interval"] == 0:
                log_msg = (
                    f"step: {step}  "
                    f"loss: {loss_for_logging:.4f}  "
                    f"grad_norm: {metrics['grad_norm']:.4f}  "
                    f"tps: {format_number_with_comma(metrics['tokens_per_sec'])}  "
                )
                logger.info(log_msg)

                if run is not None:
                    run.log(
                        {
                            "loss": loss_for_logging,
                            "learning_rate": current_lr,
                            "grad_norm": metrics["grad_norm"],
                            "tokens_per_sec": metrics["tokens_per_sec"],
                            "step_time": metrics["step_time"],
                        }
                    )

                if (
                    val_loader is not None
                    and step % training_config["validation_interval"] == 0
                    and step > 0
                ):
                    val_wer, per_dataset_metrics = validate(
                        model, val_loader, device, config, dtype
                    )
                    if run is not None:
                        wandb_metrics = {
                            "val_wer": val_wer,
                            "step": step,
                        }
                        wandb_metrics.update(per_dataset_metrics)
                        run.log(wandb_metrics)

                    llm_name = model_config["llm"].replace("/", "-")
                    checkpoint_name = (
                        f"{llm_name}_checkpoint_"
                        f"{model_config['model_id']}_step_{step}.pt"
                    )
                    # wer used for early stop
                    if val_wer < best_val_wer:
                        logger.info(
                            f"Validation WER improved: {best_val_wer:.4f} -> {val_wer:.4f}"
                        )
                        best_val_wer = val_wer
                        no_improve_epochs = 0
                        if last_ckpt_path and os.path.exists(last_ckpt_path):
                            os.remove(last_ckpt_path)
                            logger.info(f"Removed old checkpoint: {last_ckpt_path}")
                        last_ckpt_path = save_checkpoint(
                            model,
                            optimizer,
                            step,
                            loss,
                            val_wer,
                            config,
                            checkpoint_name,
                        )
                    else:
                        no_improve_epochs += 1
                        logger.info(
                            f"No improvement for {no_improve_epochs} validations "
                            f"(patience: {early_stopping_config['patience']})"
                        )
                        if no_improve_epochs >= early_stopping_config["patience"]:
                            logger.info(
                                f"Early stopping triggered at step {step} "
                                f"(no improvement for {early_stopping_config['patience']} validations)"
                            )
                            break
                    model.train()

            ### Save checkpoint every N steps
            save_every = training_config.get("save_every", 0)
            if save_every > 0 and step % save_every == 0 and step > 0:
                llm_name = model_config["llm"].replace("/", "-")
                periodic_ckpt_name = (
                    f"{llm_name}_checkpoint_{model_config['model_id']}_step_{step}.pt"
                )
                save_checkpoint(
                    model,
                    optimizer,
                    step,
                    loss,
                    best_val_wer,
                    config,
                    periodic_ckpt_name,
                )
                logger.info(f"Saved periodic checkpoint at step {step}")

            step += 1
            pbar.update(1)

        pbar.close()

        ### save final checkpoint no mather what
        final_checkpoint_name = f"checkpoint_{model_config['model_id']}_final.pt"
        loss = 0.0 # for debug
        save_checkpoint(
            model, optimizer, step, loss, best_val_wer, config, final_checkpoint_name
        )

        if run is not None:
            run.finish()

        logger.info("Training complete!")
        # prof.export_stacks("stacks24.txt", "self_cpu_time_total")
        # with open("profiler4.txt", "w") as f:
        #     f.write("=== MODULE-LEVEL BREAKDOWN ===\n")
        #     f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
        #     f.write("\n\n=== BY INPUT SHAPE + STACK ===\n")
        #     f.write(prof.key_averages(group_by_input_shape=True, group_by_stack_n=5).table(sort_by="cpu_time_total", row_limit=20))
        #     f.write("\n\n=== BY CUDA MEMORY ===\n")
        #     f.write(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_memory_usage", row_limit=20))
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if run is not None:
            run.finish()
        raise e
    finally:
        torch.distributed.destroy_process_group()
           