"""
Utility functions for the repurchase prediction model.
"""

import logging
import os
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from google.cloud import storage
from datetime import datetime

# GCS logger
gcs_logger = logging.getLogger(__name__)


def masked_mean(
    x: torch.Tensor, 
    attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute mean over sequence dimension while ignoring padded positions.
    
    This is useful for pooling variable-length sequences where some positions
    are padding and should not contribute to the mean.
    
    Args:
        x: Input tensor of shape [batch_size, seq_len, dim]
        attention_mask: Boolean mask of shape [batch_size, seq_len]
                       - True: valid position (include in mean)
                       - False: padding position (exclude from mean)
    
    Returns:
        pooled: Tensor of shape [batch_size, dim] containing mean over valid positions
    
    Example:
        >>> x = torch.randn(2, 5, 128)
        >>> mask = torch.tensor([[True, True, True, False, False],
        ...                      [True, True, True, True, False]])
        >>> pooled = masked_mean(x, mask)
        >>> pooled.shape
        torch.Size([2, 128])
        
    Notes:
        - If all positions are masked (all False), uses denominator of 1.0 to avoid NaN
        - Equivalent to: mean(x[mask]) for each batch element
    """
    # Convert boolean mask to float and expand for broadcasting
    mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, S, 1]
    
    # Zero out masked positions
    x_masked = x * mask_expanded  # [B, S, D]
    
    # Compute sum over sequence dimension
    sum_x = x_masked.sum(dim=1)  # [B, D]
    
    # Count valid positions per batch element
    count = mask_expanded.sum(dim=1).clamp(min=1.0)  # [B, 1] - at least 1 to avoid division by zero
    
    # Compute mean
    mean_x = sum_x / count  # [B, D]
    
    return mean_x


def masked_sum(
    x: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute sum over sequence dimension while ignoring padded positions.
    
    Args:
        x: Input tensor of shape [batch_size, seq_len, dim]
        attention_mask: Boolean mask of shape [batch_size, seq_len]
                       - True: valid position (include in sum)
                       - False: padding position (exclude from sum)
    
    Returns:
        pooled: Tensor of shape [batch_size, dim] containing sum over valid positions
    
    Example:
        >>> x = torch.randn(2, 5, 128)
        >>> mask = torch.tensor([[True, True, True, False, False],
        ...                      [True, True, True, True, False]])
        >>> pooled = masked_sum(x, mask)
        >>> pooled.shape
        torch.Size([2, 128])
    """
    # Convert boolean mask to float and expand
    mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, S, 1]
    
    # Zero out masked positions and sum
    x_masked = x * mask_expanded
    sum_x = x_masked.sum(dim=1)  # [B, D]
    
    return sum_x


def masked_max(
    x: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute max over sequence dimension while ignoring padded positions.
    
    Args:
        x: Input tensor of shape [batch_size, seq_len, dim]
        attention_mask: Boolean mask of shape [batch_size, seq_len]
                       - True: valid position (include in max)
                       - False: padding position (exclude from max)
    
    Returns:
        pooled: Tensor of shape [batch_size, dim] containing max over valid positions
    
    Example:
        >>> x = torch.randn(2, 5, 128)
        >>> mask = torch.tensor([[True, True, True, False, False],
        ...                      [True, True, True, True, False]])
        >>> pooled = masked_max(x, mask)
        >>> pooled.shape
        torch.Size([2, 128])
        
    Notes:
        - Masked positions are set to -inf before taking max
        - If all positions are masked, returns zeros instead of -inf
    """
    # Set masked positions to -inf so they don't affect max
    x_masked = x.clone()
    # Expand mask for broadcasting: [B, S] -> [B, S, 1]
    mask_expanded = attention_mask.unsqueeze(-1)
    
    # Set padded positions (False) to -inf
    x_masked[~mask_expanded.expand_as(x_masked)] = float('-inf')
    
    # Take max
    max_x = x_masked.max(dim=1)[0]  # [B, D]
    
    # Handle case where all positions are masked (would result in -inf)
    max_x = torch.where(torch.isinf(max_x), torch.zeros_like(max_x), max_x)
    
    return max_x


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        num_params: Total number of trainable parameters
    
    Example:
        >>> from src.model import RepurchaseModel
        >>> model = RepurchaseModel(vocab_size=10000, embedding_dim=256)
        >>> num_params = count_parameters(model)
        >>> print(f"Model has {num_params:,} parameters")
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(use_cuda: bool = True) -> torch.device:
    """
    Get the appropriate device (CPU or CUDA).
    
    Args:
        use_cuda: Whether to use CUDA if available
    
    Returns:
        device: torch.device object
    
    Note:
        For training, prefer setup_device() which auto-detects and logs.
        This function is kept for backward compatibility and fine-grained control.
    """
    if use_cuda and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def setup_device() -> torch.device:
    """
    Setup device for training (auto-detect with logging).
    
    Follows pb_insp_tt approach: auto-detect GPU availability.
    Always uses best available device (no config parameter).
    
    Returns:
        torch.device instance (cuda if available, else cpu)
    
    Example:
        >>> device = setup_device()
        Using device: cuda
        >>> print(device)
        cuda
    
    Notes:
        - Automatically uses CUDA if available
        - Falls back to CPU if CUDA not available
        - Logs the selected device
        - Recommended for training scripts
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gcs_logger.info(f"Using device: {device}")
    return device


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed: Random seed value
        deterministic: If True, enables deterministic behavior in CUDNN
                      (may reduce performance but ensures full reproducibility)
    
    Example:
        >>> set_seed(42)
        >>> # All random operations will now be reproducible
        
        >>> # For full determinism (slower but fully reproducible)
        >>> set_seed(42, deterministic=True)
    
    Notes:
        - Sets seeds for: random, numpy, torch, torch.cuda
        - deterministic=True impacts performance but ensures reproducibility
        - Call this at the start of training/inference scripts
        - Recommended seed values: 42, 0, 1337, etc.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        # For full reproducibility (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        gcs_logger.info(f"Set random seed to {seed} with deterministic=True (may impact performance)")
    else:
        gcs_logger.info(f"Set random seed to {seed}")


def setup_directories(job_dir: str, local_mode: bool = False) -> Dict[str, str]:
    """
    Setup directory structure for training/inference jobs.
    
    Args:
        job_dir: Base directory for job outputs (e.g., experiments/run_20260202_123456)
        local_mode: If True, uses more readable structure (unused, kept for compatibility)
    
    Returns:
        Dictionary of directory paths with keys:
            - job_dir: Base directory
            - checkpoints: Model checkpoint directory
            - tensorboard: TensorBoard logs directory
            - logs: Training logs directory
            - vocab: Vocabulary files directory
            - predictions: Model predictions directory
            - artifacts: Additional artifacts directory
    
    Example:
        >>> dirs = setup_directories('experiments/run_20260202_123456')
        >>> print(dirs['checkpoints'])
        experiments/run_20260202_123456/checkpoints
    
    Notes:
        - Creates all directories with exist_ok=True
        - Logs the base directory path
        - Safe to call multiple times (idempotent)
    """
    dirs = {
        "job_dir": job_dir,
        "checkpoints": os.path.join(job_dir, "checkpoints"),
        "tensorboard": os.path.join(job_dir, "tensorboard"),
        "logs": os.path.join(job_dir, "logs"),
        "vocab": os.path.join(job_dir, "vocab"),
        "predictions": os.path.join(job_dir, "predictions"),
        "artifacts": os.path.join(job_dir, "artifacts"),
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    gcs_logger.info(f"Created directory structure in {job_dir}")
    return dirs


# ============================================================================
# GCS UTILITIES
# ============================================================================

def parse_gcs_uri(gcs_uri: str) -> Tuple[str, str]:
    """
    Parse GCS URI into bucket name and blob path.
    
    Args:
        gcs_uri: Full GCS URI (e.g., 'gs://bucket-name/path/to/file')
    
    Returns:
        Tuple of (bucket_name, blob_path)
    
    Example:
        >>> bucket, path = parse_gcs_uri('gs://my-bucket/data/file.parquet')
        >>> bucket
        'my-bucket'
        >>> path
        'data/file.parquet'
    """
    parts = gcs_uri.replace("gs://", "").split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def download_from_gcs(gcs_uri: str, local_path: str) -> str:
    """
    Download a file from GCS to local path.
    
    Args:
        gcs_uri: GCS URI (gs://bucket/path/to/file)
        local_path: Local destination path
    
    Returns:
        local_path: Path to downloaded file
    
    Example:
        >>> local_file = download_from_gcs(
        ...     'gs://my-bucket/config.yaml',
        ...     '/tmp/config.yaml'
        ... )
    """
    client = storage.Client()
    bucket_name, blob_path = parse_gcs_uri(gcs_uri)
    
    gcs_logger.info(f"Downloading from GCS: {gcs_uri}")
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)
    gcs_logger.info(f"Downloaded to: {local_path}")
    
    return local_path


def upload_to_gcs(local_path: str, gcs_uri: str) -> None:
    """
    Upload a local file to GCS.
    
    Args:
        local_path: Path to local file
        gcs_uri: Destination GCS URI
    
    Example:
        >>> upload_to_gcs(
        ...     '/tmp/model.pt',
        ...     'gs://my-bucket/models/model.pt'
        ... )
    """
    client = storage.Client()
    bucket_name, blob_path = parse_gcs_uri(gcs_uri)
    
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
    
    gcs_logger.info(f"Uploaded {local_path} to {gcs_uri}")


def upload_directory_to_gcs(local_directory: str, gcs_uri: str) -> None:
    """
    Upload entire directory to GCS recursively.
    
    Args:
        local_directory: Path to local directory
        gcs_uri: Destination GCS base URI
    
    Example:
        >>> upload_directory_to_gcs(
        ...     '/tmp/checkpoints',
        ...     'gs://my-bucket/models/experiment_1/checkpoints'
        ... )
    
    Notes:
        - Preserves directory structure
        - Uploads all files recursively
        - Logs each file upload
    """
    client = storage.Client()
    bucket_name, gcs_path = parse_gcs_uri(gcs_uri.rstrip("/"))
    bucket = client.bucket(bucket_name)
    
    for root, _, files in os.walk(local_directory):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, local_directory)
            gcs_file_path = os.path.join(gcs_path, relative_path)
            
            blob = bucket.blob(gcs_file_path)
            blob.upload_from_filename(local_file_path)
            gcs_logger.info(f"Uploaded {local_file_path} to gs://{bucket_name}/{gcs_file_path}")


def load_config(config_path: str, job_dir: str) -> Dict[str, Any]:
    """
    Load configuration from local path or GCS URI.
    
    Args:
        config_path: Path to YAML config file (local or gs://)
        job_dir: Local directory for temporary downloads
    
    Returns:
        config: Configuration dictionary
    
    Example:
        >>> # Load from local path
        >>> config = load_config('configs/train.yaml', '/tmp')
        
        >>> # Load from GCS
        >>> config = load_config('gs://bucket/configs/train.yaml', '/tmp')
    
    Notes:
        - Automatically downloads from GCS if path starts with 'gs://'
        - Returns parsed YAML as dictionary
    """
    if config_path.startswith("gs://"):
        local_config_path = os.path.join(job_dir, "config.yaml")
        download_from_gcs(config_path, local_config_path)
        config_path = local_config_path
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    gcs_logger.info(f"Loaded config from {config_path}")
    return config


def get_dir_name(prefix="run", tag=None, timestamp=None):
    """
    Returns a directory name like run_YYYYMMDD_HHMMSS[_tag]
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_suffix = f"_{tag}" if tag else ""
    return f"{prefix}_{timestamp}{tag_suffix}"
