import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import RepurchaseModel
from .utils import (
    load_config, 
    setup_device, 
    download_from_gcs, 
    upload_directory_to_gcs,
    get_dir_name,
    set_seed
)
from .dataloader import (
    VocabularyBuilder, 
    DataDownloader
)
from .inference_dataloader import create_inference_dataloader

def load_model_and_vocab(checkpoint_path: str, config: Dict[str, Any], device: torch.device):
    """
    Load trained model and vocabulary from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration dictionary (fallback if checkpoint doesn't have model_config)
        device: Device to load model on
        
    Returns:
        Tuple of (model, vocab, sequence_config)
        - model: Loaded model in eval mode
        - vocab: Vocabulary from checkpoint
        - sequence_config: Sequence settings from checkpoint (for validation)
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    logger.info(f"Checkpoint: {checkpoint.keys()}")
    
    # Extract vocab from checkpoint
    vocab = VocabularyBuilder()
    vocab.item_to_idx = checkpoint['vocab']['item_to_idx']
    vocab.idx_to_item = checkpoint['vocab']['idx_to_item']
    vocab._vocab_size = checkpoint['vocab']['vocab_size']
    
    logger.info(f"Loaded vocabulary: {vocab._vocab_size} items")
    
    # Create model
    model_config = checkpoint.get('model_config')
    logger.info(f"Model config: {model_config}")
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
            cnn_layer_enabled=model_config.get('cnn_layer_enabled', True),
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
            cnn_layer_enabled=config['model'].get('cnn_layer_enabled', True),
            scoring_mode=config['model'].get('scoring_mode', 'both'),
        )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully (epoch {checkpoint['epoch']}, step {checkpoint['global_step']})")
    
    # Return sequence config for validation (explicit, not from full config)
    sequence_config = checkpoint.get('sequence_config')
    if not sequence_config:
        # Fallback to full config for backward compatibility with old checkpoints
        logger.warning("Checkpoint does not have explicit 'sequence_config'. Extracting from full config.")
        sequence_config = checkpoint.get('config', {}).get('sequence', {})
    
    return model, vocab, sequence_config


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


def inference_model(
    model: RepurchaseModel,
    inference_loader: DataLoader,
    vocab: VocabularyBuilder,
    device: torch.device,
    top_k: int = 500
) -> List[Dict[str, Any]]:
    """
    Inference model on inference set.
    
    Args:
        model: Trained model
        inference_loader: Inference data loader
        vocab: Vocabulary
        device: Device
        top_k: Number of top predictions to return
        
    Returns:
        List of inference results, e.g. [{"cid": ..., "top_500_prediction_items": [...], ...}]
    """
    logger.info("Starting evaluation...")
    
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(inference_loader, desc="Inferencing")):
            # Get model predictions
            final_scores = inference_batch(model, batch, device)  # [B, vocab_size]
            
            # Get batch data
            item_indices = batch["item_indices"]  # [B, S]
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
            
            for b in range(batch_size):
                # Get valid items for this customer
                valid_mask = attention_mask[b]
                valid_item_indices = item_indices[b][valid_mask]  # [num_valid]
                valid_scores = batch_scores[b][valid_mask]  # [num_valid]
                
                # Rank items by score (highest first)
                sorted_indices = torch.argsort(valid_scores, descending=True)
                ranked_vocab_indices = valid_item_indices[sorted_indices]
                ranked_items = [vocab.idx_to_item.get(idx.item(), "<UNK>") for idx in ranked_vocab_indices]
                
                # Save prediction with both indices and item IDs
                predictions.append({
                    'cid': cids[b].item() if isinstance(cids[b], torch.Tensor) else cids[b],
                    'top_500_prediction_items': ranked_items[:top_k], 
                    'top_500_prediction_scores': valid_scores[sorted_indices][:top_k].cpu().tolist(), 
                    'num_candidates': len(ranked_vocab_indices),
                })
    return predictions


def save_predictions(predictions: List[Dict[str, Any]], output_path: str):
    output_path = os.path.join(output_path, "predictions.parquet")
    df = pd.DataFrame(predictions)
    df.to_parquet(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Run inference with trained repurchase model")
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
        "--inference_data_path", 
        type=str, 
        default=None,
        help="Override inference data path (reuse test_data_path in config)")    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results (JSON)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for evaluation (default: use config value)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Number of top predictions to return per customer (default: use config value)"
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
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run in local mode (skip GCS uploads, use local paths)"
    )
    parser.add_argument(
        "--vertex_config",
        type=str,
        default=None,
        help="Path to vertex config YAML file (optional, for infrastructure settings)"
    )
    
    args = parser.parse_args()

    # Setup logging (console only initially)
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    # Setup job directory
    inference_dir_name = get_dir_name(prefix="inference", tag=args.tag)
    cleanup_dir = None  # Track directory to clean up later

    if args.local:
        # For local runs, use experiments/ directory in project root
        project_root =  '/home/yanan_cao_walmart_com/work/PB_cnn_sab_spf/cnn_st_spf_nbr'
        job_dir = os.path.join(project_root, "experiments", inference_dir_name)
    else:
        # For Vertex AI, create experiments structure in temp directory for consistent layout
        cleanup_dir = tempfile.mkdtemp()
        job_dir = os.path.join(cleanup_dir, "experiments", inference_dir_name)

    os.makedirs(job_dir, exist_ok=True)

    # Download checkpoint from GCS if needed
    checkpoint_path = args.checkpoint
    if args.checkpoint.startswith("gs://"):
        checkpoint_local_path = os.path.join(job_dir, os.path.basename(args.checkpoint))
        download_from_gcs(args.checkpoint, checkpoint_local_path)
        logger.info(f"Downloaded checkpoint from {args.checkpoint} to {checkpoint_local_path}")
        checkpoint_path = checkpoint_local_path

    # Add file logging now that job_dir exists
    log_file = os.path.join(job_dir, "inference.log")
    logger.add(
        log_file,
        level=args.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="500 MB",
        retention="10 days",
        compression="zip"
    )
    logger.info(f"Inference directory: {job_dir}")
    logger.info(f"Logging to file: {log_file}")
    
    # Load config
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config, job_dir)
    
    # Setup artifact URI for Vertex AI
    artifact_base_uri = None
    if not args.local and args.vertex_config:
        vertex_config = load_config(args.vertex_config, job_dir)
        if "vertex" in vertex_config:
            staging_bucket = vertex_config["vertex"].get("staging_bucket")
            if staging_bucket:
                job_id = os.environ.get("CLOUD_ML_JOB_ID")
                if job_id:
                    artifact_base_uri = f"{staging_bucket}/inference-job-output/{job_id}"
                else:
                    artifact_base_uri = f"{staging_bucket}/inference-job-output/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                logger.info(f"Artifact base URI: {artifact_base_uri}")
    
    # Apply CLI overrides
    if args.batch_size:
        config["batch"]["batch_size"] = args.batch_size
    if args.top_k:
        if "inference" not in config:
            config["inference"] = {}
        config["inference"]["top_k"] = args.top_k
    
    # Set random seed for reproducibility
    seed = config.get("seed", 42)
    set_seed(seed)
    
    # Save config to job folder for reproducibility
    config_save_path = os.path.join(job_dir, "config.yaml")
    with open(config_save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Saved config to {config_save_path}")

    # Create predictions directory
    predictions_dir = os.path.join(job_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Setup device
    device = setup_device()
    logger.info(f"Using device: {device}")
    
    # Load model and vocabulary
    model, vocab, checkpoint_sequence_config = load_model_and_vocab(checkpoint_path, config, device)
    
    # Use sequence config from checkpoint if available (source of truth)
    if checkpoint_sequence_config:
        # Override inference.yaml with checkpoint values
        config['sequence']['max_sequence_length'] = checkpoint_sequence_config.get('max_sequence_length')
        config['sequence']['truncate_strategy'] = checkpoint_sequence_config.get('truncate_strategy')
        
        logger.info(
            f"Using sequence settings from checkpoint: "
            f"max_length={config['sequence']['max_sequence_length']}, "
            f"strategy={config['sequence']['truncate_strategy']}"
        )
    else:
        logger.warning(
            "Checkpoint does not contain sequence_config. "
            f"Using inference.yaml: max_length={config['sequence']['max_sequence_length']}, "
            f"strategy={config['sequence']['truncate_strategy']}"
        )
    
    # Get inference data path from config
    if args.inference_data_path:
        logger.info(f"Overriding inference data path with CLI argument: {args.inference_data_path}")
        config['data']['test_path'] = args.inference_data_path
    inference_path = config['data']['test_path']
    inference_pattern = config['data'].get('test_pattern', '*.parquet')
    max_inference_files = config['data'].get('max_test_files', None)
    
    logger.info("=" * 60)
    logger.info("Configuration Summary")
    logger.info("=" * 60)
    logger.info(f"Vocabulary size: {vocab._vocab_size}")
    logger.info(f"Inference data: {inference_path}")
    logger.info(f"Batch size: {config['batch']['batch_size']}")
    logger.info(f"Top-K predictions: {config.get('inference', {}).get('top_k', 500)}")
    logger.info(f"Device: {device}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info("=" * 60)
    
    logger.info(f"Loading inference data from {inference_path}")
    
    # Get inference files
    downloader = DataDownloader(temp_dir=config['data']['temp_dir'])
    inference_files = downloader.list_files(inference_path, inference_pattern, max_inference_files)
    logger.info(f"Found {len(inference_files)} inference files")
    
    if len(inference_files) == 0:
        logger.error(f"No inference files found at {inference_path} with pattern {inference_pattern}")
        sys.exit(1)
    
    # Create inference dataloader    
    inference_loader = create_inference_dataloader(
        files=inference_files,
        vocab=vocab,
        config=config,
    )
        
    logger.info(f"Created inference dataloader: {len(inference_loader.dataset)} samples, {len(inference_loader)} batches")

    # Get top_k from config
    top_k = config.get('inference', {}).get('top_k', 500)
    logger.info(f"Returning top {top_k} predictions per customer")

    # Run inference
    logger.info("=" * 60)
    logger.info("Starting inference...")
    logger.info("=" * 60)
    
    try:
        predictions = inference_model(model, inference_loader, vocab, device, top_k=top_k)
        save_predictions(predictions, predictions_dir)
        logger.info("Inference completed successfully!")
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise
    
    # Upload entire job directory to GCS (if not local mode)
    if not args.local and artifact_base_uri:
        logger.info("=" * 60)
        logger.info("Uploading entire inference directory to GCS...")
        logger.info(f"Source: {job_dir}")
        logger.info(f"Destination: {artifact_base_uri}")
        logger.info("=" * 60)
        
        try:
            upload_directory_to_gcs(job_dir, artifact_base_uri)
            logger.info("=" * 60)
            logger.info(f"✅ All artifacts uploaded to gs://{artifact_base_uri}")
            logger.info("=" * 60)
        except Exception as e:
            logger.error(f"❌ Failed to upload artifacts: {e}")
            logger.error("Keeping local files for inspection...")
            raise

    # Clean up (skip in local mode for inspection)
    if not args.local and cleanup_dir:
        logger.info("Cleaning up temporary directory...")
        os.system(f"rm -rf {cleanup_dir}")
    else:
        logger.info(f"Local mode: outputs saved in {job_dir}")
        logger.info(f"  Predictions: {predictions_dir}")
    
    logger.info("=" * 60)
    logger.info("Inference completed successfully!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()