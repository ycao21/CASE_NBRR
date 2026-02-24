import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List, Union

import torch
from loguru import logger

from src.dataloader import (
    VocabularyBuilder,
    load_and_split_data,
    create_dataloader,
)
from src.loss import RankerLoss, BCELoss
from src.model import RepurchaseModel
from src.train import Trainer, setup_scheduler
from src.utils import (
    download_from_gcs,
    upload_directory_to_gcs,
    load_config,
    setup_device,
    setup_directories,
    get_dir_name,
    set_seed,
)


# ============================================================================
# CONFIG & SETUP
# ============================================================================

def load_or_build_vocab(
    config: Dict[str, Any],
    train_files: list,
    vocab_dir: str
) -> VocabularyBuilder:
    """
    Load pre-built vocabulary or build from training files.
    
    Args:
        config: Full configuration dictionary
        train_files: List of training file paths
        vocab_dir: Directory to save vocabulary files
    
    Returns:
        VocabularyBuilder instance
    """
    vocab_path = config["data"].get("vocab_path")
    
    if vocab_path:
        # Load pre-built vocabulary
        logger.info(f"Loading pre-built vocabulary from {vocab_path}")
        
        if vocab_path.startswith("gs://"):
            local_vocab_path = os.path.join(vocab_dir, "item_to_idx.json")
            download_from_gcs(vocab_path, local_vocab_path)
            vocab_path = local_vocab_path
        
        with open(vocab_path, "r") as f:
            item_to_idx = json.load(f)
        
        # Create VocabularyBuilder from loaded vocab
        vocab = VocabularyBuilder()
        vocab.item_to_idx = {int(k): v for k, v in item_to_idx.items()}
        vocab.idx_to_item = {v: int(k) for k, v in item_to_idx.items()}
        vocab._vocab_size = len(vocab.item_to_idx) + 1  # +1 for padding
        
        logger.info(f"Loaded vocabulary: {vocab._vocab_size - 1} items (+1 padding)")
    else:
        # Build vocabulary from training files
        logger.info("Building vocabulary from training files...")
        vocab = VocabularyBuilder()
        
        column_mapping = config["data"].get("column_mapping")
        vocab_stats = vocab.build_from_files(train_files, column_mapping)
        
        logger.info(f"Built vocabulary: {vocab_stats['vocab_size']} items")
        logger.info(f"  Total item occurrences: {vocab_stats['total_item_occurrences']}")
        logger.info(f"  Unique items in data: {vocab_stats['unique_items_in_data']}")
    
    # Save vocabulary to experiment folder
    vocab_save_path = os.path.join(vocab_dir, "item_to_idx.json")
    with open(vocab_save_path, "w") as f:
        json.dump(vocab.item_to_idx, f, indent=2)
    logger.info(f"Saved vocabulary to {vocab_save_path}")
    
    # Also save reverse mapping for convenience
    idx_to_item_path = os.path.join(vocab_dir, "idx_to_item.json")
    with open(idx_to_item_path, "w") as f:
        json.dump(vocab.idx_to_item, f, indent=2)
    logger.info(f"Saved reverse vocabulary to {idx_to_item_path}")
    
    return vocab


# ============================================================================
# SETUP FUNCTIONS
# ============================================================================

def setup_dataloaders(
    config: Dict[str, Any],
    vocab: VocabularyBuilder,
    train_files: List[str],
    val_files: List[str]
) -> Tuple[Any, Any]:  # Returns (train_loader, val_loader)
    """
    Create train and validation dataloaders.
    
    Args:
        config: Configuration dictionary
        vocab: Vocabulary builder with item mappings
        train_files: List of training file paths (already downloaded)
        val_files: List of validation file paths (already downloaded)
    """
    logger.info("Setting up dataloaders...")
    
    logger.info(f"  Train files: {len(train_files)}")
    logger.info(f"  Val files: {len(val_files)}")

    # Create train dataloader
    train_loader, _ = create_dataloader(
        config=config,
        split="train",
        files=train_files,
        vocab=vocab,
    )
    
    # Create val dataloader
    val_loader, _ = create_dataloader(
        config=config,
        split="val",
        files=val_files,
        vocab=vocab,
    )
    
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    
    return train_loader, val_loader


def setup_model(config: Dict[str, Any], vocab_size: int, device: torch.device) -> RepurchaseModel:
    """Create and initialize model."""
    logger.info("Creating model...")
    
    model = RepurchaseModel(
        vocab_size=vocab_size,
        embedding_dim=config["model"]["embedding_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        cnn_output_dim=config["model"]["cnn_output_dim"],
        cnn_layer_enabled=config["model"]["cnn_layer_enabled"],
        cnn_kernel_sizes=config["model"]["cnn_kernel_sizes"],
        membership_dim=config["model"]["membership_dim"],
        num_heads=config["model"]["num_heads"],
        num_inds=config["model"]["num_inds"],
        dropout=config["model"]["dropout"],
        category_pooling_type=config["model"].get("category_pooling_type", "average"),
        pma_num_heads=config["model"].get("pma_num_heads", 4),
        set_phi_type=config["model"].get("set_phi_type", "deep_sets"),
        perm_eq_num_stacks=config["model"].get("perm_eq_num_stacks", 1),
        scoring_mode=config["model"].get("scoring_mode", "both"),
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    return model


def setup_criterion(config: Dict[str, Any]) -> Union[RankerLoss, BCELoss]:
    """Create loss function."""
    logger.info("Creating loss function...")
    
    if config["loss"]["type"] == "RankerLoss":
        criterion = RankerLoss(
            margin=config["loss"]["margin"],
            sampling_strategy=config["loss"]["sampling_strategy"],
            neg_sample_ratio=config["loss"]["neg_sample_ratio"],
            frequency_window_days=config["loss"].get("frequency_window_days", 28),
            strategy_weights=config["loss"].get("strategy_weights"),
        )
    elif config["loss"]["type"] == "BCELoss":
        criterion = BCELoss(
            sampling_strategy=config["loss"]["sampling_strategy"],
            neg_sample_ratio=config["loss"]["neg_sample_ratio"],
            frequency_window_days=config["loss"].get("frequency_window_days", 28),
            strategy_weights=config["loss"].get("strategy_weights"),
            pos_weight=config["loss"].get("pos_weight"),
            reduction=config["loss"].get("reduction", "mean"),
        )
    else:
        raise ValueError(f"Invalid loss type: {config['loss']['type']}")
    return criterion

def setup_optimizer_and_scheduler(
    model: RepurchaseModel,
    config: Dict[str, Any]
) -> Tuple[torch.optim.Optimizer, Optional[Any]]:
    """Create optimizer and learning rate scheduler."""
    logger.info("Creating optimizer...")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["optimizer"]["learning_rate"],
        weight_decay=config["optimizer"].get("weight_decay", 0.01),
    )
    
    scheduler = None
    if "scheduler" in config and config["scheduler"].get("type"):
        logger.info(f"Creating scheduler: {config['scheduler']['type']}")
        scheduler = setup_scheduler(optimizer, config["scheduler"])
    
    return optimizer, scheduler


def setup_trainer(
    model: RepurchaseModel,
    criterion: RankerLoss,
    optimizer: torch.optim.Optimizer,
    train_loader: Any,
    val_loader: Any,
    device: torch.device,
    dirs: Dict[str, str],
    config: Dict[str, Any],
    vocab: Any,
    scheduler: Optional[Any] = None
) -> Trainer:
    """Create trainer instance."""
    logger.info("Creating trainer...")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        config=config["training"],  # Pass training section for training params
        device=device,
        checkpoint_dir=dirs["checkpoints"],
        log_dir=dirs["tensorboard"],
        predictions_dir=dirs["predictions"],
        vocab=vocab,
        full_config=config,  # Pass full config for checkpoint saving
    )
    
    return trainer


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description="Train RepurchaseModel with CNN + Set Transformer"
    )
    # Config file
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML file (local or gs://)"
    )
    parser.add_argument(
        "--vertex_config",
        type=str,
        default=None,
        help="Path to vertex config YAML file (optional, for infrastructure settings)"
    )
    # Execution mode
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run in local mode (skip GCS uploads, use local paths)"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    # Data overrides (minimal - vocab is auto-built)
    parser.add_argument(
        "--train_data_path", 
        type=str, 
        help="Override training data path")
    # Training params (can override config)
    parser.add_argument(
        "--batch_size", 
        type=int, 
        help="Override batch size")
    parser.add_argument(
        "--num_epochs", 
        type=int, 
        help="Override number of epochs")
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        help="Override learning rate")
    parser.add_argument(
        "--resume_from", 
        type=str, 
        help="Path to checkpoint to resume from")

    # Optional tag for experiment output directory
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional tag for experiment output directory"
    )

    # Parse args
    args = parser.parse_args()

    # Setup logging (console only initially)
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    # Setup job directory
    run_dir_name = get_dir_name(prefix="run", tag=args.tag)
    cleanup_dir = None  # Track directory to clean up later

    if args.local:
        # For local runs, use experiments/ directory in project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        job_dir = os.path.join(project_root, "experiments", run_dir_name)
    else:
        # For Vertex AI, create experiments structure in temp directory for consistent layout
        cleanup_dir = tempfile.mkdtemp()
        job_dir = os.path.join(cleanup_dir, "experiments", run_dir_name)

    os.makedirs(job_dir, exist_ok=True)
    
    # Add file logging now that job_dir exists
    log_file = os.path.join(job_dir, "train.log")
    logger.add(
        log_file,
        level=args.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="500 MB",  # Rotate when file reaches 500 MB
        retention="10 days",  # Keep logs for 10 days
        compression="zip"  # Compress rotated logs
    )
    logger.info(f"Experiment directory: {job_dir}")
    logger.info(f"Logging to file: {log_file}")
    
    # Load training config
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
            artifact_base_uri = f"{staging_bucket}/training-job-output/{job_id}"
        else:
            artifact_base_uri = f"{staging_bucket}/training-job-output/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Artifact base URI: {artifact_base_uri}")

    # Apply CLI overrides
    if args.train_data_path:
        config["data"]["train_path"] = args.train_data_path
    # if args.test_data_path:
    #     config["data"]["test_path"] = args.test_data_path
    # if args.vocab_path:
    #     config["data"]["vocab_path"] = args.vocab_path
    if args.batch_size:
        config["batch"]["batch_size"] = args.batch_size
    if args.num_epochs:
        config["training"]["num_epochs"] = args.num_epochs
    if args.learning_rate:
        config["optimizer"]["learning_rate"] = args.learning_rate
    
    # Set random seed for reproducibility
    seed = config.get("seed", 42)
    deterministic = config.get("deterministic", True)  # Default True for full reproducibility
    set_seed(seed, deterministic=deterministic)
    
    # Setup directories
    dirs = setup_directories(job_dir, args.local)
    
    # Save config to experiment folder for reproducibility
    import yaml
    config_save_path = os.path.join(job_dir, "config.yaml")
    with open(config_save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Saved config to {config_save_path}")
    
    # Setup device (auto-detect)
    device = setup_device()
    
    # Get file splits first (needed for vocab building)
    logger.info("Loading and splitting data files...")
    train_files, val_files = load_and_split_data(config)
    
    # Load or build vocabulary (saves to dirs["vocab"])
    vocab = load_or_build_vocab(config, train_files, dirs["vocab"])
    vocab_size = vocab._vocab_size
    
    logger.info("=" * 60)
    logger.info("Configuration Summary")
    logger.info("=" * 60)
    logger.info(f"Vocabulary size: {vocab_size}")
    logger.info(f"Training data: {config['data']['train_path']}")
    logger.info(f"Batch size: {config['batch']['batch_size']}")
    logger.info(f"Epochs: {config['training']['num_epochs']}")
    logger.info(f"Learning rate: {config['optimizer']['learning_rate']}")
    logger.info(f"Random seed: {seed} (deterministic={deterministic})")
    logger.info(f"Loss type: {config['loss']['type']}")
    logger.info(f"Negative sampling strategy: {config['loss']['sampling_strategy']}")
    logger.info(f"Negative sample ratio: {config['loss']['neg_sample_ratio']}")
    if config['loss'].get('strategy_weights'):
        logger.info(f"Strategy weights (only when mixed strategy is used): ")
        logger.info(f"  {config['loss']['strategy_weights']}")
    logger.info(f"Device: {device}")
    logger.info("=" * 60)

    # Setup components
    train_loader, val_loader = setup_dataloaders(config, vocab, train_files, val_files)
    model = setup_model(config, vocab_size, device)
    criterion = setup_criterion(config)
    optimizer, scheduler = setup_optimizer_and_scheduler(model, config)
    trainer = setup_trainer(
        model, criterion, optimizer, train_loader, val_loader,
        device, dirs, config, vocab, scheduler
    )
    
    # Train
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)
    
    try:
        trainer.train(resume_from=args.resume_from)
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    # Upload entire experiment directory to GCS (if not local mode)
    if not args.local and artifact_base_uri:
        logger.info("=" * 60)
        logger.info("Uploading entire experiment directory to GCS...")
        logger.info(f"Source: {job_dir}")
        logger.info(f"Destination: gs://{artifact_base_uri}")
        logger.info("=" * 60)
        
        try:
            upload_directory_to_gcs(job_dir, artifact_base_uri)
            logger.info("=" * 60)
            logger.info(f"✅ All artifacts uploaded to gs://{artifact_base_uri}")
            logger.info("=" * 60)
        except Exception as e:
            logger.error(f"❌ Failed to upload artifacts: {e}")
            logger.error("Keeping local files for inspection...")
            # Don't delete job_dir if upload fails
            raise

    # Clean up (skip in local mode for inspection)
    if not args.local and cleanup_dir:
        logger.info("Cleaning up temporary directory...")
        os.system(f"rm -rf {cleanup_dir}")
    else:
        logger.info(f"Local mode: outputs saved in {job_dir}")
        logger.info(f"  Checkpoints: {dirs['checkpoints']}")
        logger.info(f"  TensorBoard: {dirs['tensorboard']}")
    
    logger.info("=" * 60)
    logger.info("Runner completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
