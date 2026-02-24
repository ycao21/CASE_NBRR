import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataloader import (
    VocabularyBuilder,
    DataDownloader,
    create_dataloader,
)
from .model import RepurchaseModel
from .metrics import compute_batch_metrics
from .utils import (
    load_config, 
    setup_device, 
    download_from_gcs, 
    get_dir_name
)


def load_model_and_vocab(checkpoint_path: str, config: Dict[str, Any], device: torch.device):
    """
    Load trained model and vocabulary from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration dictionary
        device: Device to load model on
        
    Returns:
        Tuple of (model, vocab)
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract vocab from checkpoint
    vocab = VocabularyBuilder()
    vocab.item_to_idx = checkpoint['vocab']['item_to_idx']
    vocab.idx_to_item = checkpoint['vocab']['idx_to_item']
    vocab._vocab_size = checkpoint['vocab']['vocab_size']
    
    logger.info(f"Loaded vocabulary: {vocab._vocab_size} items")
    
    # Create model
    model_config = checkpoint.get('model_config')
    if model_config:
        logger.info("Model config found in checkpoint. Using it to construct model architecture.")
        model = RepurchaseModel(
            vocab_size=vocab._vocab_size,
            embedding_dim=model_config['embedding_dim'],
            cnn_output_dim=model_config['cnn_output_dim'],
            hidden_dim=model_config['hidden_dim'],
            dropout=model_config['dropout'],
            membership_dim=model_config['membership_dim'],
            cnn_kernel_sizes=model_config['cnn_kernel_sizes'],
            set_phi_type=model_config['set_phi_type'],
            perm_eq_num_stacks=model_config.get('perm_eq_num_stacks', 2),
            num_heads=model_config.get('num_heads', 4),
            num_inds=model_config.get('num_inds', 32),
            category_pooling_type=model_config.get('category_pooling_type', 'pma'),
            pma_num_heads=model_config.get('pma_num_heads', 4),
            scoring_mode=model_config.get('scoring_mode', 'both'),
        )
    else:
        logger.warning("Model config didn't save in the checkpoint. Using current config for model architecture.")
        model = RepurchaseModel(
            vocab_size=vocab._vocab_size,
            embedding_dim=config['model']['embedding_dim'],
            cnn_output_dim=config['model']['cnn_output_dim'],
            hidden_dim=config['model']['hidden_dim'],
            dropout=config['model']['dropout'],
            membership_dim=config['model']['membership_dim'],
            cnn_kernel_sizes=config['model']['cnn_kernel_sizes'],
            set_phi_type=config['model']['set_phi_type'],
            perm_eq_num_stacks=config['model'].get('perm_eq_num_stacks', 2),
            num_heads=config['model'].get('num_heads', 4),
            num_inds=config['model'].get('num_inds', 32),
            category_pooling_type=config['model'].get('category_pooling_type', 'pma'),
            pma_num_heads=config['model'].get('pma_num_heads', 4),
            scoring_mode=config['model'].get('scoring_mode', 'both'),
        )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully (epoch {checkpoint['epoch']}, step {checkpoint['global_step']})")
    
    return model, vocab


def inference_batch(
    model: RepurchaseModel,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """
    Run inference on a batch.
    
    Args:
        model: Trained model
        batch: Batch of data
        device: Device to run on
        
    Returns:
        final_scores: [B, vocab_size] scores for all items
    """
    # Move batch to device
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    
    with torch.no_grad():
        outputs = model(batch)
        final_scores = outputs["final_scores"]  # [B, vocab_size]

    return final_scores


def evaluate_model(
    model: RepurchaseModel,
    test_loader: DataLoader,
    vocab: VocabularyBuilder,
    device: torch.device,
    k_values_recall: List[int] = [10, 50, 100, 500],
    k_values_precision: List[int] = [1, 5, 10, 20],
    save_predictions: bool = True,
    top_k_predictions: int = 500
) -> Dict[str, Any]:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        vocab: Vocabulary
        device: Device
        k_values_recall: K values for Recall@K
        k_values_precision: K values for Precision@K
        save_predictions: Whether to save per-sample predictions
        
    Returns:
        Dictionary with aggregate metrics and per-sample results
    """
    logger.info("Starting evaluation...")
    
    model.eval()
    
    all_sample_metrics = []
    all_predictions = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Get model predictions
            final_scores = inference_batch(model, batch, device)  # [B, vocab_size]
            
            # Get batch data
            item_indices = batch["item_indices"]  # [B, S]
            labels = batch["labels"]  # [B, S]
            attention_mask = batch["attention_mask"]  # [B, S]
            cids = batch["cid"]
            
            batch_size = final_scores.shape[0]
            
            # Gather scores for batch items (same approach as train.py)
            # final_scores: [B, vocab_size] -> gather -> [B, S]
            batch_scores = torch.gather(
                final_scores,
                dim=1,
                index=item_indices
            )
            
            # Mask padding positions with -inf
            # attention_mask: True = valid, False = padding
            batch_scores = batch_scores.masked_fill(
                ~attention_mask,  # Mask padding (not valid)
                float("-inf")
            )
            
            # Compute metrics for this batch using k_values
            k_values = list(set(k_values_recall + k_values_precision))
            batch_metrics = compute_batch_metrics(
                logits=batch_scores,      # [B, S] - scores for actual batch items
                labels=labels,            # [B, S] - ground truth labels
                k_values=k_values
            )
            
            # Store batch metrics
            all_sample_metrics.append(batch_metrics)
            
            # Save prediction details if requested
            if save_predictions:
                # Save predictions for each sample in batch
                for b in range(batch_size):
                    # Get valid items for this customer
                    valid_mask = attention_mask[b]
                    valid_item_indices = item_indices[b][valid_mask]  # [num_valid]
                    valid_labels = labels[b][valid_mask]  # [num_valid]
                    valid_scores = batch_scores[b][valid_mask]  # [num_valid]
                    
                    # Get ground truth positives (items with label=1)
                    positive_mask = valid_labels == 1
                    ground_truth_indices = valid_item_indices[positive_mask]  # Vocab indices
                    ground_truth_items = [vocab.idx_to_item.get(idx.item(), "<UNK>") for idx in ground_truth_indices]
                    
                    # Rank items by score (highest first)
                    sorted_indices = torch.argsort(valid_scores, descending=True)
                    ranked_vocab_indices = valid_item_indices[sorted_indices]
                    ranked_items = [vocab.idx_to_item.get(idx.item(), "<UNK>") for idx in ranked_vocab_indices]
                    
                    # Save prediction with both indices and item IDs
                    all_predictions.append({
                        'cid': cids[b].item() if isinstance(cids[b], torch.Tensor) else cids[b],
                        'ground_truth_items': ground_truth_items,  # Already a Python list
                        'top_500_prediction_items': ranked_items[:top_k_predictions], 
                        'top_500_prediction_scores': valid_scores[sorted_indices][:top_k_predictions].cpu().tolist(), 
                        'num_ground_truth': len(ground_truth_indices),
                        'num_candidates': len(ranked_vocab_indices),
                    })
    
    # Aggregate metrics across all batches (weighted average by sample count)
    logger.info(f"Evaluated {len(all_sample_metrics)} batches")
    
    aggregate_metrics = {}
    if len(all_sample_metrics) > 0:
        # Get all metric keys from first batch (exclude 'num_samples')
        metric_keys = [k for k in all_sample_metrics[0].keys() if k != 'num_samples']
        
        # Weighted average across batches
        total_samples = sum(batch_metrics.get('num_samples', 0) for batch_metrics in all_sample_metrics)
        
        for key in metric_keys:
            # Weighted sum
            weighted_sum = sum(
                batch_metrics[key] * batch_metrics.get('num_samples', 0)
                for batch_metrics in all_sample_metrics
            )
            aggregate_metrics[key] = weighted_sum / total_samples if total_samples > 0 else 0.0
        
        aggregate_metrics['num_samples'] = total_samples
    else:
        aggregate_metrics['num_samples'] = 0
    
    logger.info(f"Total samples evaluated: {aggregate_metrics.get('num_samples', 0)}")
    
    return {
        'aggregate_metrics': aggregate_metrics,
        'per_sample_predictions': all_predictions if save_predictions else [],
    }


def save_results(results: Dict[str, Any], output_path: str):
    """Save evaluation results to JSON file."""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        else:
            return obj
    
    results_native = convert_to_native(results)
    
    with open(output_path, 'w') as f:
        json.dump(results_native, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained repurchase prediction model")
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint file (.pt)"
    )
    parser.add_argument(
        "--test_data_path", 
        type=str, 
        default=None,
        help="Override test data path")    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results (JSON). Default: eval_results_{timestamp}.json"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for evaluation (default: use config value)"
    )
    parser.add_argument(
        "--no_save_predictions",
        action="store_true",
        help="Don't save per-sample predictions (only aggregate metrics)"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional tag for evaluation output directory"
    )
    
    args = parser.parse_args()

    # Download checkpoint from GCS if needed
    checkpoint_path = args.checkpoint
    experiments_base = "/home/yanan_cao_walmart_com/work/PB_cnn_sab_spf/cnn_st_spf_nbr/experiments"
    eval_dir_name = get_dir_name(prefix="eval", tag=args.tag)
    eval_run_dir = None

    if args.checkpoint.startswith("gs://"):
        eval_run_dir = os.path.join(experiments_base, eval_dir_name)
        os.makedirs(eval_run_dir, exist_ok=True)
        checkpoint_path = os.path.join(eval_run_dir, os.path.basename(args.checkpoint))
        download_from_gcs(args.checkpoint, checkpoint_path)
        logger.info(f"Downloaded checkpoint to {checkpoint_path}")

    # Setup logging (console first)
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    # Consistent timestamp for this run
    # Determine log file location and output path for results
    if args.output:
        output_dir = os.path.dirname(os.path.abspath(args.output))
        log_file = os.path.join(output_dir, f"{eval_dir_name}.log")
        output_path = args.output
    else:
        experiment_dir = Path(checkpoint_path).parent
        # Always use the parent of checkpoint_dir as the experiment directory
        predictions_dir = experiment_dir / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        log_file = str(experiment_dir / f"{eval_dir_name}.log")
        output_path = str(predictions_dir / f"{eval_dir_name}_results.json")

    # Add file logging
    logger.add(
        log_file,
        level=args.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="500 MB",
        retention="10 days",
        compression="zip"
    )
    logger.info(f"Logging to file: {log_file}")
    
    # Load config
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config, Path(checkpoint_path).parent)
    
    # Setup device
    device = setup_device()
    logger.info(f"Using device: {device}")
    
    # Load model and vocabulary
    model, vocab = load_model_and_vocab(checkpoint_path, config, device)
    
    # Get test data path from config
    if args.test_data_path:
        logger.info(f"Overriding test data path with CLI argument: {args.test_data_path}")
        config['data']['test_path'] = args.test_data_path
    test_path = config['data']['test_path']
    test_pattern = config['data'].get('test_pattern', '*.parquet')
    max_test_files = config['data'].get('max_test_files', None)
    
    logger.info(f"Loading test data from {test_path}")
    
    # Get test files
    downloader = DataDownloader(temp_dir=config['data']['temp_dir'])
    test_files = downloader.list_files(test_path, test_pattern, max_test_files)
    logger.info(f"Found {len(test_files)} test files")
    
    # Create test dataloader    
    test_loader, _ = create_dataloader(
        files=test_files,
        vocab=vocab,
        config=config,
        split="test",
    )
    
    logger.info(f"Created test dataloader: {len(test_loader.dataset)} samples, {len(test_loader)} batches")
    
    # Run evaluation
    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        vocab=vocab,
        device=device,
        save_predictions=not args.no_save_predictions,
        top_k_predictions=500
    )
    
    # Print aggregate metrics
    logger.info("=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Num samples evaluated: {results['aggregate_metrics']['num_samples']}")
    logger.info("")
    logger.info("Recall Metrics:")
    for k in [10, 50, 100, 500]:
        key = f"Recall@{k}"
        if key in results['aggregate_metrics']:
            logger.info(f"  {key}: {results['aggregate_metrics'][key]:.4f}")
    logger.info("")
    logger.info("Precision Metrics:")
    for k in [1, 5, 10, 20]:
        key = f"Precision@{k}"
        if key in results['aggregate_metrics']:
            logger.info(f"  {key}: {results['aggregate_metrics'][key]:.4f}")
    logger.info("")
    if 'MRR' in results['aggregate_metrics']:
        logger.info(f"MRR: {results['aggregate_metrics']['MRR']:.4f}")
    if 'NDCG@10' in results['aggregate_metrics']:
        logger.info(f"NDCG@10: {results['aggregate_metrics']['NDCG@10']:.4f}")
    if 'NDCG@50' in results['aggregate_metrics']:
        logger.info(f"NDCG@50: {results['aggregate_metrics']['NDCG@50']:.4f}")
    logger.info("=" * 80)
    
    save_results(results, output_path)
    
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
