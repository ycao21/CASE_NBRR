"""
Training script for repurchase prediction model.

Implements training loop with debug analysis following pb_insp_tt patterns.
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .model import RepurchaseModel
from .loss import RankerLoss
from .metrics import compute_batch_metrics, format_metrics


def debug_batch_helper(batch: Dict, logger):
    """
    Log detailed batch statistics for debugging.
    
    Logs shapes, data types, and counts of valid items.
    """
    logger.info("=" * 60)
    logger.info("=== DEBUG BATCH ANALYSIS ===")
    logger.info("=" * 60)

    logger.info("\n--- Input Shapes ---")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            logger.info(f"  {key}: {value.shape} (dtype: {value.dtype})")
        elif hasattr(value, "shape"):
            logger.info(f"  {key}: {value.shape}")
        else:
            logger.info(f"  {key}: {type(value).__name__}")

    # Count valid items (non-padding)
    if "attention_mask" in batch:
        attention_mask = batch["attention_mask"]  # True = valid, False = padding
        batch_size, seq_len = attention_mask.shape
        valid_items_per_sample = attention_mask.sum(dim=1).float()

        logger.info(f"\n--- Item Counts ---")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Sequence length: {seq_len}")
        logger.info(f"  Valid items per sample:")
        logger.info(f"    Mean: {valid_items_per_sample.mean().item():.2f}")
        logger.info(f"    Min: {valid_items_per_sample.min().item():.0f}")
        logger.info(f"    Max: {valid_items_per_sample.max().item():.0f}")
    
    # Count positive labels
    if "labels" in batch:
        labels = batch["labels"]
        attention_mask = batch.get("attention_mask", torch.ones_like(labels, dtype=torch.bool))
        # attention_mask: True = valid, False = padding
        # Mask out padding positions (set them to 0)
        valid_labels = labels.masked_fill(~attention_mask, 0)
        
        positives_per_sample = valid_labels.sum(dim=1).float()
        total_positives = positives_per_sample.sum().item()
        
        logger.info(f"\n--- Label Distribution ---")
        logger.info(f"  Total positives: {total_positives:.0f}")
        logger.info(f"  Positives per sample:")
        logger.info(f"    Mean: {positives_per_sample.mean().item():.2f}")
        logger.info(f"    Min: {positives_per_sample.min().item():.0f}")
        logger.info(f"    Max: {positives_per_sample.max().item():.0f}")
        
        # Samples with no positives
        samples_without_positives = (positives_per_sample == 0).sum().item()
        logger.info(f"  Samples with no positives: {samples_without_positives}")


def compute_debug_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    task_name: str,
    logger,
) -> Dict[str, float]:
    """
    Compute and log detailed debug metrics for logits/labels.
    
    Args:
        logits: Predicted scores [B, N] - scores for positive+negative samples (from loss)
        labels: Ground truth labels [B, N] - 1 for positives, 0 for negatives (from loss)
        attention_mask: NOT USED (kept for API compatibility, shapes don't match)
        task_name: Name for logging (e.g., "Train", "Val")
        logger: Logger instance
        
    Returns:
        Dictionary of computed metrics
    
    Note: Loss function masks padding with -inf in logits, so we infer valid
          positions from non-inf logits instead of using attention_mask.
    """
    if logits is None or logits.numel() == 0:
        logger.info(f"  [{task_name}] No logits available")
        return {}
    
    metrics = {}
    
    # Basic shapes
    batch_size = logits.size(0)
    num_items = logits.size(1)
    logger.info(f"  [{task_name}] Batch size: {batch_size}, Items: {num_items}")
    
    # Get valid mask (non-padding positions, where logits != -inf)
    # The loss function masks out padding with -inf, so we use that
    valid_mask = ~torch.isinf(logits)  # True = valid (not -inf)
    pos_mask = (labels > 0) & valid_mask
    neg_mask = (labels == 0) & valid_mask
    
    # Extract valid scores
    valid_pos_scores = logits[pos_mask]
    valid_neg_scores = logits[neg_mask]
    
    total_valid_pos = valid_pos_scores.numel()
    total_valid_neg = valid_neg_scores.numel()
    
    logger.info(f"  [{task_name}] Valid items - Positives: {total_valid_pos}, Negatives: {total_valid_neg}")
    
    # Per-sample counts
    valid_pos_per_sample = pos_mask.sum(dim=1).float()
    valid_neg_per_sample = neg_mask.sum(dim=1).float()
    
    avg_pos = valid_pos_per_sample.mean().item()
    avg_neg = valid_neg_per_sample.mean().item()
    
    metrics[f"{task_name}_avg_pos_per_sample"] = avg_pos
    metrics[f"{task_name}_avg_neg_per_sample"] = avg_neg
    
    logger.info(f"  [{task_name}] Positives per sample: avg={avg_pos:.2f}, min={valid_pos_per_sample.min().item():.0f}, max={valid_pos_per_sample.max().item():.0f}")
    logger.info(f"  [{task_name}] Negatives per sample: avg={avg_neg:.2f}, min={valid_neg_per_sample.min().item():.0f}, max={valid_neg_per_sample.max().item():.0f}")
    
    # Score statistics
    if valid_pos_scores.numel() > 0:
        pos_mean = valid_pos_scores.mean().item()
        pos_std = valid_pos_scores.std().item()
        pos_min = valid_pos_scores.min().item()
        pos_max = valid_pos_scores.max().item()
        
        metrics[f"{task_name}_pos_score_mean"] = pos_mean
        metrics[f"{task_name}_pos_score_std"] = pos_std
        
        logger.info(f"  [{task_name}] Positive scores: mean={pos_mean:.4f}, std={pos_std:.4f}, range=[{pos_min:.4f}, {pos_max:.4f}]")
    
    if valid_neg_scores.numel() > 0:
        neg_mean = valid_neg_scores.mean().item()
        neg_std = valid_neg_scores.std().item()
        neg_min = valid_neg_scores.min().item()
        neg_max = valid_neg_scores.max().item()
        
        metrics[f"{task_name}_neg_score_mean"] = neg_mean
        metrics[f"{task_name}_neg_score_std"] = neg_std
        
        logger.info(f"  [{task_name}] Negative scores: mean={neg_mean:.4f}, std={neg_std:.4f}, range=[{neg_min:.4f}, {neg_max:.4f}]")
        
        # Compute margin (separation between pos and neg)
        if valid_pos_scores.numel() > 0:
            margin = valid_pos_scores.mean() - valid_neg_scores.mean()
            metrics[f"{task_name}_margin"] = margin.item()
            logger.info(f"  [{task_name}] Score margin (pos_mean - neg_mean): {margin:.4f}")
            
            # Percentage of positives > avg negative
            pos_beats_neg_mean = (valid_pos_scores > valid_neg_scores.mean()).float().mean().item()
            metrics[f"{task_name}_pos_beats_neg_mean"] = pos_beats_neg_mean
            logger.info(f"  [{task_name}] Positives > avg negative: {pos_beats_neg_mean:.2%}")
            
            # Pairwise accuracy (sampled)
            n_samples = min(1000, valid_pos_scores.numel(), valid_neg_scores.numel())
            if n_samples > 0:
                pos_sample = valid_pos_scores[torch.randint(0, valid_pos_scores.numel(), (n_samples,))]
                neg_sample = valid_neg_scores[torch.randint(0, valid_neg_scores.numel(), (n_samples,))]
                pairwise_acc = (pos_sample > neg_sample).float().mean().item()
                metrics[f"{task_name}_pairwise_acc"] = pairwise_acc
                logger.info(f"  [{task_name}] Pairwise accuracy: {pairwise_acc:.2%}")
    
    # Compute Recall@K on this batch
    positives_per_sample = labels.sum(dim=1)
    valid_samples = positives_per_sample > 0
    
    if valid_samples.sum() > 0:
        for k in [1, 5, 10]:
            k_actual = min(k, logits.size(1))
            _, top_k_idx = torch.topk(logits, k=k_actual, dim=1)
            top_k_labels = torch.gather(labels, 1, top_k_idx)
            recall_per_sample = top_k_labels.sum(dim=1) / positives_per_sample.clamp(min=1)
            batch_recall = (recall_per_sample * valid_samples.float()).sum() / valid_samples.sum()
            
            metrics[f"{task_name}_recall@{k}"] = batch_recall.item()
            logger.info(f"  [{task_name}] Recall@{k}: {batch_recall:.4f}")
    
    return metrics


def train_step(
    model: RepurchaseModel,
    batch: Dict,
    criterion: RankerLoss,
    device: torch.device,
    is_debug_batch: bool = False,
) -> Dict:
    """
    Perform one training step (forward + loss computation).
    
    Args:
        model: RepurchaseModel instance
        batch: Batch of data
        criterion: Loss function
        device: Device to run on
        is_debug_batch: Whether to log debug information
        
    Returns:
        Dictionary with loss and debug information
    """
    # Move batch to device
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    if is_debug_batch:
        debug_batch_helper(batch, logger)

    # Forward pass - model expects batch dictionary
    outputs = model(batch)

    # Compute loss - criterion returns a dict
    loss_output = criterion(
        final_scores=outputs["final_scores"],
        item_indices=batch["item_indices"],  # Validated by collate function
        labels=batch["labels"],
        attention_mask=batch["attention_mask"],
        membership=batch.get("membership"),
    )
    
    # Extract loss tensor from dict
    loss = loss_output["loss"]
    logits = loss_output["logits"]
    labels_out = loss_output["labels"]
    
    # Handle NaN loss
    if torch.isnan(loss):
        logger.debug("NaN loss detected - returning zero loss")
        return {
            "loss": torch.tensor(0.0, device=device, requires_grad=True),
            "logits": outputs["final_scores"],
            "labels": batch["labels"],
            "attention_mask": batch["attention_mask"],
        }

    # Debug metrics
    if is_debug_batch:
        logger.info("\n--- Loss ---")
        logger.info(f"  Total Loss: {loss.item():.4f}")
        
        logger.info("\n--- Debug Metrics Analysis ---")
        compute_debug_metrics(
            logits=logits,
            labels=labels_out,
            attention_mask=batch["attention_mask"],
            task_name="Train",
            logger=logger,
        )
        logger.info("=" * 60)

    return {
        "loss": loss,
        "logits": logits,  # [B, T+N] from loss - for debug metrics
        "final_scores": outputs["final_scores"],  # [B, vocab_size] from model - for validation metrics
        "labels": labels_out,  # From loss_output
        "attention_mask": batch["attention_mask"],
    }


class Trainer:
    """
    Trainer for repurchase prediction model.
    
    Handles training loop, validation, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: RepurchaseModel,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: RankerLoss,
        config: Dict,
        device: torch.device,
        checkpoint_dir: str,
        log_dir: str,
        predictions_dir: str = None,
        vocab = None,
        full_config: Dict = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: RepurchaseModel instance
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer instance
            criterion: Loss function (RankerLoss)
            config: Training configuration dictionary
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for TensorBoard logs
            predictions_dir: Directory to save predictions and debug outputs
            vocab: Vocabulary object (for saving in checkpoints)
            full_config: Full configuration dictionary (for saving model/sequence configs)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.device = device
        self.vocab = vocab
        self.full_config = full_config if full_config is not None else config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Predictions directory
        self.predictions_dir = Path(predictions_dir) if predictions_dir else self.checkpoint_dir / "predictions"
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_recall = 0.0
        
        # Training config
        self.num_epochs = config.get("num_epochs", 10)
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
        self.log_interval = config.get("log_interval", 100)
        self.val_interval = config.get("val_interval", 1000)
        self.save_interval = config.get("save_interval", 1000)
        self.debug_steps = config.get("debug_steps", 100)
        self.k_values_train = config.get("k_values_train", [1, 5, 10, 20])
        self.primary_metric = config.get("primary_metric", "Recall@10")
        
        logger.info(f"Trainer initialized:")
        logger.info(f"  Device: {device}")
        logger.info(f"  Num epochs: {self.num_epochs}")
        logger.info(f"  Gradient accumulation steps: {self.gradient_accumulation_steps}")
        logger.info(f"  Debug steps: {self.debug_steps}")
        logger.info(f"  Primary metric: {self.primary_metric}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of average training metrics for the epoch
        """
        self.model.train()
        
        epoch_loss = 0.0
        epoch_steps = 0
        last_grad_norm = 0.0  # Track gradient norm for progress bar
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.num_epochs}",
            total=len(self.train_loader),
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Debug based on gradient steps, not batch index
            # Check if this batch will trigger an optimizer step AND if that step is a debug milestone
            will_update = (batch_idx + 1) % self.gradient_accumulation_steps == 0
            next_global_step = self.global_step + 1 if will_update else self.global_step
            is_debug_batch = will_update and (next_global_step % self.debug_steps == 0)

            # Training step (forward + loss)
            loss_dict = train_step(
                model=self.model,
                batch=batch,
                criterion=self.criterion,
                device=self.device,
                is_debug_batch=is_debug_batch,
            )
            
            loss = loss_dict["loss"]

            # Check for NaN
            if torch.isnan(loss):
                logger.debug(f"NaN loss at epoch {epoch}, batch {batch_idx}")
                continue

            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()

            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping (and capture gradient norm)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get("max_grad_norm", 1.0)
                )
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Store grad norm for progress bar
                last_grad_norm = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
                
                # Log to TensorBoard
                if self.global_step % self.log_interval == 0:
                    self.writer.add_scalar(
                        "Train/Loss",
                        loss.item() * self.gradient_accumulation_steps,
                        self.global_step
                    )
                    self.writer.add_scalar(
                        "Train/GradNorm",
                        grad_norm,
                        self.global_step
                    )
                    self.writer.add_scalar(
                        "Train/LearningRate",
                        self.optimizer.param_groups[0]["lr"],
                        self.global_step
                    )
                
                # Validation
                if self.global_step % self.val_interval == 0:
                    val_metrics = self.validate()
                    self.log_validation_metrics(val_metrics, self.global_step)

                    # Check if best model
                    val_recall = val_metrics.get(self.primary_metric, 0.0)
                    if val_recall > self.best_val_recall:
                        self.best_val_recall = val_recall
                        self.save_checkpoint(is_best=True)
                        logger.info(
                            f"New best model! {self.primary_metric}: {val_recall:.4f}"
                        )
                    
                    self.model.train()
                
                # Save checkpoint
                if self.global_step % self.save_interval == 0:
                    self.save_checkpoint(is_best=False)
            
            # Update progress bar
            epoch_loss += loss.item() * self.gradient_accumulation_steps
            epoch_steps += 1
            progress_bar.set_postfix({
                "loss": f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                "grad": f"{last_grad_norm:.3f}",
                "step": self.global_step,
            })
        
        avg_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0.0
        
        return {"loss": avg_loss}
    
    def validate(self) -> Dict[str, float]:
        """
        Run validation and compute metrics.
        
        Returns:
            Dictionary of validation metrics (including loss)
        """
        self.model.eval()
        
        val_loss = 0.0
        val_batches = 0
        
        # Collect sequence length statistics
        all_seq_lengths = []
        batch_max_lengths = []
        
        # Collect per-batch metrics for aggregation
        batch_metrics_list = []
        batch_sample_counts = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validating", leave=False)):
                # Debug first validation batch
                is_debug = (batch_idx == 0)
                
                # Reuse train_step for consistency
                loss_dict = train_step(
                    model=self.model,
                    batch=batch,
                    criterion=self.criterion,
                    device=self.device,
                    is_debug_batch=is_debug,
                )

                loss = loss_dict["loss"]
                logits = loss_dict["logits"]
                labels = loss_dict["labels"]
                attention_mask = loss_dict["attention_mask"]
                
                # Skip if NaN loss
                if torch.isnan(loss):
                    logger.debug(f"NaN loss in validation batch {batch_idx}")
                    continue

                val_loss += loss.item()
                val_batches += 1
                
                # Collect sequence length statistics
                batch_attention_mask = batch["attention_mask"]  # [B, S]
                # Count valid items per customer (sum along seq dimension)
                seq_lengths = batch_attention_mask.sum(dim=1).cpu().numpy()  # [B]
                all_seq_lengths.extend(seq_lengths.tolist())
                batch_max_lengths.append(batch_attention_mask.shape[1])  # Max padded length
                
                # Debug: Check sequence length distribution in first few batches
                if batch_idx < 5:
                    max_seq = seq_lengths.max()
                    min_seq = seq_lengths.min()
                    mean_seq = seq_lengths.mean()
                    logger.info(f"Batch {batch_idx} seq lengths: min={min_seq}, max={max_seq}, mean={mean_seq:.1f}, shape={batch_attention_mask.shape}")
                    logger.info(f"  Per-customer: {seq_lengths.tolist()}")

                # Get scores for items in this batch
                # final_scores shape: [B, vocab_size]
                # We need to get scores only for items in this batch
                batch_item_indices = batch["item_indices"]  # [B, S], validated by collate
                
                # Extract model's final_scores (not loss logits)
                model_final_scores = loss_dict.get("final_scores")  # [B, vocab_size]
                if model_final_scores is None:
                    logger.debug("No final_scores in loss_dict, skipping metrics")
                    continue
                
                # Gather scores for batch items
                # final_scores: [B, vocab_size] -> gather -> [B, S]
                batch_scores = torch.gather(
                    model_final_scores,
                    dim=1,
                    index=batch_item_indices
                )
                
                # Mask padding positions with -inf
                # attention_mask: True = valid, False = padding
                batch_scores = batch_scores.masked_fill(
                    ~attention_mask,  # Mask padding (not valid)
                    float("-inf")
                )
                
                # DEBUG: Save logits and labels for first batch to verify metrics computation
                if batch_idx == 0:
                    # Save debug tensors to predictions directory
                    debug_save_path = self.predictions_dir / f"debug_val_step_{self.global_step}_epoch_{self.current_epoch}.pt"
                    
                    # Save tensors for manual verification
                    torch.save({
                        'batch_scores': batch_scores.cpu(),           # [B, S] - model scores
                        'labels': batch["labels"].cpu(),              # [B, S] - ground truth labels
                        'attention_mask': attention_mask.cpu(),       # [B, S] - valid positions
                        'item_indices': batch_item_indices.cpu(),     # [B, S] - vocabulary indices
                        'cid': batch['cid'],                          # Customer IDs
                        'seq_lengths': batch_attention_mask.sum(dim=1).cpu().numpy(),  # Actual lengths per sample
                    }, debug_save_path)
                    
                    logger.info(f"Saved debug data to {debug_save_path}")
                    logger.info(f"  batch_scores shape: {batch_scores.shape}")
                    logger.info(f"  labels shape: {batch['labels'].shape}")
                    logger.info(f"  attention_mask shape: {attention_mask.shape}")
                
                # Compute metrics for this batch
                # Use original batch labels (not loss labels which include sampled negatives)
                batch_metrics = compute_batch_metrics(
                    logits=batch_scores,      # [B, S] - scores for actual batch items
                    labels=batch["labels"],   # [B, S] - original labels before negative sampling
                    k_values=self.k_values_train
                )
                
                # Track for aggregation
                num_samples = batch_scores.shape[0]
                batch_metrics_list.append(batch_metrics)
                batch_sample_counts.append(num_samples)
        
        # Log sequence length distribution
        if len(all_seq_lengths) > 0:
            import numpy as np
            all_seq_lengths = np.array(all_seq_lengths)
            logger.info("=" * 60)
            logger.info("Sequence Length Distribution Across All Validation Batches:")
            logger.info(f"  Total customers: {len(all_seq_lengths)}")
            logger.info(f"  Min: {all_seq_lengths.min()}")
            logger.info(f"  Max: {all_seq_lengths.max()}")
            logger.info(f"  Mean: {all_seq_lengths.mean():.1f}")
            logger.info(f"  Median: {np.median(all_seq_lengths):.1f}")
            logger.info(f"  Percentiles:")
            logger.info(f"    25th: {np.percentile(all_seq_lengths, 25):.1f}")
            logger.info(f"    50th: {np.percentile(all_seq_lengths, 50):.1f}")
            logger.info(f"    75th: {np.percentile(all_seq_lengths, 75):.1f}")
            logger.info(f"    90th: {np.percentile(all_seq_lengths, 90):.1f}")
            logger.info(f"    95th: {np.percentile(all_seq_lengths, 95):.1f}")
            logger.info(f"    99th: {np.percentile(all_seq_lengths, 99):.1f}")
            logger.info(f"  Batch max lengths: min={min(batch_max_lengths)}, max={max(batch_max_lengths)}, mean={np.mean(batch_max_lengths):.1f}")
            logger.info("=" * 60)
        
        # Aggregate per-batch metrics (weighted average by sample count)
        total_samples = sum(batch_sample_counts)
        metrics = {}
        
        if len(batch_metrics_list) > 0:
            # Get all metric keys from first batch
            metric_keys = [k for k in batch_metrics_list[0].keys() if k != 'num_samples']
            
            for key in metric_keys:
                # Weighted average across batches
                weighted_sum = sum(
                    batch_metrics[key] * count 
                    for batch_metrics, count in zip(batch_metrics_list, batch_sample_counts)
                )
                metrics[key] = weighted_sum / total_samples if total_samples > 0 else 0.0
            
            metrics["num_samples"] = total_samples
        
        # Add validation loss to metrics
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
        metrics["val_loss"] = avg_val_loss
        
        return metrics
    
    def log_validation_metrics(self, metrics: Dict[str, float], step: int):
        """Log validation metrics to TensorBoard and console."""
        # TensorBoard
        for name, value in metrics.items():
            if name != "num_samples":
                self.writer.add_scalar(f"Val/{name}", value, step)
        
        # Console
        formatted = format_metrics(metrics, prefix="")
        logger.info(f"Validation @ step {step}: {formatted}")
    
    def save_checkpoint(self, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
        """
        # Extract critical configs for inference reproducibility
        model_config = self.full_config.get("model", {}).copy()
        sequence_config = self.full_config.get("sequence", {}).copy()

        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_recall": self.best_val_recall,
            "model_config": model_config,
            "sequence_config": sequence_config,
            # Full config kept for reference/debugging
            "config": self.full_config,
        }
        
        # Add vocabulary info if available
        if self.vocab is not None:
            checkpoint["vocab"] = {
                "item_to_idx": self.vocab.item_to_idx,
                "idx_to_item": self.vocab.idx_to_item,
                "vocab_size": self.vocab._vocab_size,
            }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)
        logger.info(f"Saved latest checkpoint to {latest_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint to resume training.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_recall = checkpoint["best_val_recall"]
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"  Epoch: {self.current_epoch}")
        logger.info(f"  Global step: {self.global_step}")
        logger.info(f"  Best val recall: {self.best_val_recall:.4f}")
    
    def train(self, resume_from: Optional[str] = None):
        """
        Main training loop.
        
        Args:
            resume_from: Optional path to checkpoint to resume from
        """
        # Resume from checkpoint if provided
        if resume_from:
            self.load_checkpoint(resume_from)
            start_epoch = self.current_epoch + 1
        else:
            start_epoch = 1
        
        logger.info("Starting training...")
        logger.info(f"  Total epochs: {self.num_epochs}")
        logger.info(f"  Starting from epoch: {start_epoch}")
        
        for epoch in range(start_epoch, self.num_epochs + 1):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_metrics = self.train_epoch(epoch)
            
            # Log epoch metrics
            logger.info(
                f"Epoch {epoch} completed: "
                f"Avg Loss = {train_metrics['loss']:.4f}"
            )
            self.writer.add_scalar("Train/EpochLoss", train_metrics["loss"], epoch)
            
            # Run validation at end of epoch
            val_metrics = self.validate()
            self.log_validation_metrics(val_metrics, self.global_step)

            # Check if best model
            val_recall = val_metrics.get(self.primary_metric, 0.0)
            if val_recall > self.best_val_recall:
                self.best_val_recall = val_recall
                self.save_checkpoint(is_best=True)
                logger.info(
                    f"New best model! {self.primary_metric}: {val_recall:.4f}"
                )
            
            # Save latest checkpoint
            self.save_checkpoint(is_best=False)
        
        # Training complete
        logger.info("Training completed!")
        logger.info(f"Best {self.primary_metric}: {self.best_val_recall:.4f}")
        self.writer.close()


def setup_optimizer(model: nn.Module, config: Dict) -> torch.optim.Optimizer:
    """
    Setup optimizer based on config.
    
    Args:
        model: Model to optimize
        config: Configuration dictionary
        
    Returns:
        Optimizer instance
    """
    optimizer_type = config.get("optimizer", "adam").lower()
    learning_rate = config.get("learning_rate", 1e-3)
    weight_decay = config.get("weight_decay", 0.0)
    
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_type == "sgd":
        momentum = config.get("momentum", 0.9)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    logger.info(f"Optimizer: {optimizer_type}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Weight decay: {weight_decay}")
    
    return optimizer


def setup_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Setup learning rate scheduler based on config.
    
    Args:
        optimizer: Optimizer instance
        config: Configuration dictionary
        
    Returns:
        Scheduler instance or None
    """
    scheduler_type = config.get("scheduler", None)
    
    if scheduler_type is None:
        return None
    
    if scheduler_type == "step":
        step_size = config.get("scheduler_step_size", 1000)
        gamma = config.get("scheduler_gamma", 0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
    elif scheduler_type == "cosine":
        T_max = config.get("scheduler_t_max", 10000)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
        )
    elif scheduler_type == "plateau":
        patience = config.get("scheduler_patience", 3)
        factor = config.get("scheduler_factor", 0.5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=patience,
            factor=factor,
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    logger.info(f"Scheduler: {scheduler_type}")
    
    return scheduler
