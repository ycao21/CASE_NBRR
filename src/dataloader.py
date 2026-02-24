import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger
from torch.utils.data import DataLoader, Dataset


class VocabularyBuilder:
    """
    Maps real item_ids to embedding indices.
    
    Index 0 is reserved for:
    - Padding (during batching for variable-length sequences)
    - Unknown/cold-start items (items not in training vocabulary)
    
    Real items from training data are mapped to indices starting from 1.
    """
    
    UNKNOWN_IDX = 0  # Used for padding and unknown items
    FIRST_ITEM_IDX = 1  # Real items start here
    
    def __init__(self):
        """Initialize vocabulary builder."""
        self.item_to_idx: Dict[int, int] = {}  # Only real item_ids
        self.idx_to_item: Dict[int, int] = {}  # Only real items (index 0 not included)
        self._vocab_size: int = 1  # Start at 1 (index 0 is reserved for padding)
    
    def build_from_files(self, parquet_files: List[str], column_mapping: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Build vocabulary from parquet files. Includes all unique items.
        
        Args:
            parquet_files: List of parquet file paths
            column_mapping: Optional dict mapping expected to actual column names
                          e.g., {'item_ids': 'item_id_arr'}
            
        Returns:
            Dictionary with vocabulary statistics
        """
        logger.info(f"Building vocabulary from {len(parquet_files)} parquet files...")
        
        # Get actual column name for item_ids
        item_col = column_mapping.get('item_ids', 'item_ids') if column_mapping else 'item_ids'
        
        # Collect all unique item IDs
        unique_items = set()
        total_items_seen = 0
        
        for file_path in parquet_files:
            logger.debug(f"Scanning {file_path}...")
            df = pd.read_parquet(file_path)
            
            for item_list in df[item_col]:
                if isinstance(item_list, (list, np.ndarray)):
                    unique_items.update(item_list)
                    total_items_seen += len(item_list)
        
        # Assign indices to all unique items (sorted for reproducibility)
        # Start from index 1 (index 0 is reserved for padding)
        next_idx = self.FIRST_ITEM_IDX
        for item_id in sorted(unique_items):
            self.item_to_idx[int(item_id)] = next_idx
            self.idx_to_item[next_idx] = int(item_id)
            next_idx += 1
        
        # vocab_size includes padding index (0) + all real items
        self._vocab_size = len(self.item_to_idx) + 1
        
        stats = {
            'vocab_size': self._vocab_size,
            'num_items': len(self.item_to_idx),
            'unique_items_in_data': len(unique_items),
            'total_item_occurrences': total_items_seen,
        }
        
        logger.info(f"Vocabulary built: {stats['vocab_size']} indices "
                   f"({stats['num_items']} real items, index 0 for padding/unknown)")
        
        return stats
    
    def get_index(self, item_id: int) -> int:
        """
        Get vocabulary index for an item_id.
        
        Args:
            item_id: Original item ID
            
        Returns:
            Vocabulary index (1+) if found, 0 if item not in vocabulary
        """
        return self.item_to_idx.get(int(item_id), self.UNKNOWN_IDX)
    
    def has_item(self, item_id: int) -> bool:
        """
        Check if item_id exists in vocabulary.
        
        Args:
            item_id: Original item ID
            
        Returns:
            True if item is in vocabulary, False otherwise
        """
        return int(item_id) in self.item_to_idx
    
    def get_item_id(self, idx: int) -> Optional[int]:
        """
        Get original item_id from vocabulary index.
        
        Args:
            idx: Vocabulary index
            
        Returns:
            Original item ID if found, None if index is 0 (unknown/padding) or not in vocabulary
        """
        if idx == self.UNKNOWN_IDX:
            return None  # Index 0 doesn't map to a real item
        return self.idx_to_item.get(idx, None)
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self._vocab_size
    
    def save(self, save_path: str):
        """
        Save vocabulary to JSON file.
        
        Args:
            save_path: Path to save vocabulary
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        vocab_data = {
            'item_to_idx': {str(k): v for k, v in self.item_to_idx.items()},
            'idx_to_item': {str(k): v for k, v in self.idx_to_item.items()},
            'vocab_size': self._vocab_size,
            'num_items': len(self.item_to_idx),
            'note': 'Index 0 is used for padding and unknown items. Real items start at index 1.',
        }
        
        with open(save_path, 'w') as f:
            json.dump(vocab_data, f, indent=2)
        
        logger.info(f"Vocabulary saved to {save_path}")
    
    @classmethod
    def load(cls, load_path: str) -> 'VocabularyBuilder':
        """
        Load vocabulary from JSON file.
        
        Args:
            load_path: Path to vocabulary file
            
        Returns:
            VocabularyBuilder instance
        """
        logger.info(f"Loading vocabulary from {load_path}...")
        
        with open(load_path, 'r') as f:
            vocab_data = json.load(f)
        
        vocab = cls()
        
        # Load mappings (convert string keys back to int)
        vocab.item_to_idx = {
            int(k): v for k, v in vocab_data['item_to_idx'].items()
        }
        vocab.idx_to_item = {
            int(k): int(v) for k, v in vocab_data['idx_to_item'].items()
        }
        vocab._vocab_size = vocab_data['vocab_size']
        
        logger.info(f"Vocabulary loaded: {vocab._vocab_size} tokens")
        return vocab


class DataDownloader:
    """Handles downloading data from GCS and managing local files."""
    
    def __init__(self, temp_dir: str = "/tmp/repurchase_data"):
        """
        Initialize data downloader.
        
        Args:
            temp_dir: Directory for temporary downloads
        """
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)
    
    @staticmethod
    def is_gcs_path(path: str) -> bool:
        """Check if path is a GCS path."""
        return path.startswith('gs://')
    
    def list_files(
        self, 
        path: str, 
        pattern: str = "*.parquet",
        max_files: Optional[int] = None
    ) -> List[str]:
        """
        List files matching pattern from local or GCS path.
        
        Args:
            path: Local or GCS path
            pattern: File pattern to match
            max_files: Maximum number of files to return
            
        Returns:
            List of file paths
        """
        if self.is_gcs_path(path):
            files = self._list_gcs_files(path, pattern)
        else:
            files = self._list_local_files(path, pattern)
        
        # Sort for reproducibility
        files = sorted(files)
        
        # Limit number of files if specified
        if max_files is not None and max_files > 0:
            logger.info(f"Limiting to {max_files} files (found {len(files)})")
            files = files[:max_files]
        
        # with open('files.txt', 'w') as f:
        #     for file in files:
        #         f.write('\''+file + '\',\n')

        return files
    
    def _list_local_files(self, path: str, pattern: str) -> List[str]:
        """List files from local filesystem."""
        path_obj = Path(path)
        
        if not path_obj.exists():
            logger.warning(f"Path does not exist: {path}")
            return []
        
        if path_obj.is_file():
            return [str(path_obj)]
        
        # Directory: glob for pattern
        files = list(path_obj.glob(pattern))
        logger.info(f"Found {len(files)} local files matching {pattern} in {path}")
        return [str(f) for f in files]
    
    def _list_gcs_files(self, path: str, pattern: str) -> List[str]:
        """List files from GCS."""
        try:
            from google.cloud import storage
        except ImportError:
            raise ImportError(
                "google-cloud-storage is required for GCS support. "
                "Install with: pip install google-cloud-storage"
            )
        
        # Parse GCS path: gs://bucket/prefix
        path_parts = path.replace('gs://', '').split('/', 1)
        bucket_name = path_parts[0]
        prefix = path_parts[1] if len(path_parts) > 1 else ''
        
        # Ensure prefix ends with / if it's a directory
        if prefix and not prefix.endswith('/'):
            prefix += '/'
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # List blobs with prefix
        blobs = bucket.list_blobs(prefix=prefix)
        
        # Filter by pattern (simple glob matching)
        pattern_suffix = pattern.replace('*', '')
        files = [
            f'gs://{bucket_name}/{blob.name}'
            for blob in blobs
            if blob.name.endswith(pattern_suffix)
        ]
        
        logger.info(f"Found {len(files)} GCS files matching {pattern} in {path}")
        return files
    
    def download_files(self, gcs_files: List[str]) -> List[str]:
        """
        Download files from GCS to local temp directory.
        
        Args:
            gcs_files: List of GCS file paths
            
        Returns:
            List of local file paths
        """
        try:
            from google.cloud import storage
        except ImportError:
            raise ImportError(
                "google-cloud-storage is required for GCS support. "
                "Install with: pip install google-cloud-storage"
            )
        
        client = storage.Client()
        local_files = []
        
        logger.info(f"Downloading {len(gcs_files)} files from GCS...")
        
        for gcs_path in gcs_files:
            # Parse GCS path
            path_parts = gcs_path.replace('gs://', '').split('/', 1)
            bucket_name = path_parts[0]
            blob_name = path_parts[1]
            
            # Create local path
            local_path = os.path.join(self.temp_dir, os.path.basename(blob_name))
            
            # Download
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(local_path)
            
            local_files.append(local_path)
            logger.debug(f"Downloaded {gcs_path} -> {local_path}")
        
        logger.info(f"Downloaded {len(local_files)} files to {self.temp_dir}")
        return local_files
    
    def cleanup(self):
        """Remove temporary download directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")


class RepurchaseDataset(Dataset):
    """
    Dataset for repurchase prediction.
    
    Loads data from parquet files lazily (files are concatenated but individual
    rows are accessed on-demand via __getitem__).
    """
    
    def __init__(
        self,
        parquet_files: List[str],
        vocab: VocabularyBuilder,
        max_sequence_length: int = 512,
        truncate_strategy: str = "recent",
        column_mapping: Dict[str, str] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            parquet_files: List of parquet file paths
            vocab: Vocabulary builder instance
            max_sequence_length: Maximum sequence length (for truncation)
            truncate_strategy: "recent" (keep most recent) or "random"
            column_mapping: Optional dict mapping expected to actual column names
        """
        self.vocab = vocab
        self.max_sequence_length = max_sequence_length
        self.truncate_strategy = truncate_strategy
        self.column_mapping = column_mapping or {}
        
        # Load all parquet files (lazy loading - pandas reads efficiently)
        self.data = self._load_parquet_files(parquet_files)
        
        logger.info(f"Dataset initialized with {len(self.data)} samples")
        logger.info(f"  max_sequence_length: {self.max_sequence_length}")
        logger.info(f"  truncate_strategy: {self.truncate_strategy}")
    
    def _load_parquet_files(self, parquet_files: List[str]) -> pd.DataFrame:
        """Load and concatenate parquet files."""
        dfs = []
        
        for file_path in parquet_files:
            logger.debug(f"Loading {file_path}...")
            df = pd.read_parquet(file_path)
            dfs.append(df)
        
        if not dfs:
            raise ValueError("No parquet files loaded!")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(combined_df)} samples from {len(parquet_files)} files")
        
        return combined_df
    
    def _truncate_sequence(
        self, 
        item_ids: np.ndarray, 
        labels: np.ndarray, 
        membership: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Truncate sequences to max_sequence_length.
        
        For "recent" strategy: Sorts items by their most recent purchase date
        (determined by the largest index with value 1 in the membership vector),
        then keeps the top N most recent items.
        
        Args:
            item_ids: Item ID array
            labels: Label array (same order as item_ids)
            membership: Membership array [seq_len, 364] where larger index = more recent date
            
        Returns:
            Truncated (item_ids, labels, membership)
        """
        seq_len = len(item_ids)
        
        # ALWAYS log truncation for debugging
        if seq_len > self.max_sequence_length:
            logger.debug(f"Truncating sequence: {seq_len} -> {self.max_sequence_length} (strategy: {self.truncate_strategy})")
        
        if seq_len <= self.max_sequence_length:
            return item_ids, labels, membership
        
        if self.truncate_strategy == "recent":
            # Sort items by recency based on membership vectors
            # For each item, find the SMALLEST index where membership[i][j] == 1
            # This gives us the most recent purchase date for each item (smaller index = more recent)
            last_purchase_indices = []
            for i in range(seq_len):
                # Find all indices where this item was purchased (value == 1)
                purchase_dates = np.where(membership[i] == 1)[0]
                if len(purchase_dates) > 0:
                    # Get the smallest index (most recent)
                    last_purchase_indices.append(purchase_dates[0])
                else:
                    # No purchases found, assign large value (will be sorted to end)
                    logger.debug(f"an occurence of Item ID {item_ids[i]} has no purchases in membership vector.")
                    last_purchase_indices.append(365)
            
            last_purchase_indices = np.array(last_purchase_indices)
            
            # Sort indices by last purchase date in ascending order (smaller index = more recent)
            sorted_positions = np.argsort(last_purchase_indices)
            
            # Reorder all arrays by recency
            item_ids = item_ids[sorted_positions]
            labels = labels[sorted_positions]
            membership = membership[sorted_positions]
            
            # Keep the first max_sequence_length items (most recent)
            item_ids = item_ids[:self.max_sequence_length]
            labels = labels[:self.max_sequence_length]
            membership = membership[:self.max_sequence_length]
        elif self.truncate_strategy == "random":
            # Random sampling
            indices = np.random.choice(
                seq_len, 
                size=self.max_sequence_length, 
                replace=False
            )
            indices = np.sort(indices)  # Keep chronological order
            item_ids = item_ids[indices]
            labels = labels[indices]
            membership = membership[indices]
        else:
            raise ValueError(f"Unknown truncate_strategy: {self.truncate_strategy}")
        
        return item_ids, labels, membership
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with item_indices, labels, membership, cid
            Note: Unknown items (not in vocabulary) are mapped to index 0
        """
        row = self.data.iloc[idx]
        
        # Get actual column names
        item_col = self.column_mapping.get('item_ids', 'item_ids')
        label_col = self.column_mapping.get('labels', 'labels')
        membership_col = self.column_mapping.get('membership', 'membership')
        
        # Get raw data
        item_ids = np.array(row[item_col], dtype=np.int64)
        labels = np.array(row[label_col], dtype=np.int32)
        membership_raw = row[membership_col]
        if not isinstance(membership_raw, list):
            membership_raw = list(membership_raw)
        membership = np.array(membership_raw, dtype=np.float32)  # Shape: [seq_len, 365]
                
        # Truncate if needed
        orig_len = len(item_ids)
        item_ids, labels, membership = self._truncate_sequence(
            item_ids, labels, membership
        )
        
        # Verify truncation worked
        if len(item_ids) > self.max_sequence_length:
            logger.error(f"TRUNCATION FAILED! Before: {orig_len}, After: {len(item_ids)}, Max: {self.max_sequence_length}")
        
        # Convert item_ids to vocabulary indices
        # Known items → index 1+
        # Unknown items → index 0 (same as padding)
        item_indices = np.array([
            self.vocab.get_index(item_id) 
            for item_id in item_ids
        ], dtype=np.int64)
        
        # VALIDATION: Check if all items are unknown (mapped to 0)
        # This causes issues in attention mechanism (all keys masked → NaN)
        num_valid_items = np.sum(item_indices != 0)
        if num_valid_items == 0:
            logger.debug(
                f"Sample with CID {row['cid']} has ALL items unknown (not in vocabulary). "
                f"Original items: {len(item_ids)}, All mapped to index 0. "
                f"This sample will be DROPPED during collation."
            )
            # Mark this sample as invalid by setting a flag
            return {
                'cid': row['cid'],
                'item_indices': item_indices,
                'labels': labels,
                'membership': membership,
                'seq_length': len(item_indices),
                '_invalid': True,  # Flag for collate_fn to drop
            }
        
        return {
            'cid': row['cid'],
            'item_indices': item_indices,
            'labels': labels,
            'membership': membership,
            'seq_length': len(item_indices),
            '_invalid': False,
        }


class RepurchaseCollate:
    """Collate function for batching with padding."""
    
    def __init__(
        self,
        pad_item_idx: int = 0,
        pad_label_value: int = -100,
        membership_padding_value: float = 0.0,
    ):
        """
        Initialize collate function.
        
        Args:
            pad_item_idx: Padding index for item_indices (0)
            pad_label_value: Padding value for labels (-100 for PyTorch ignore_index)
            membership_padding_value: Padding value for membership features (0.0, same as non-member)
                Stored as float since membership is feature data, not indices
        """
        self.pad_item_idx = pad_item_idx
        self.pad_label_value = pad_label_value
        self.membership_padding_value = membership_padding_value
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch with padding.
        
        Args:
            batch: List of samples from dataset
            
        Returns:
            Batched and padded tensors
        """
        # Filter out invalid samples (all items unknown/not in vocabulary)
        original_batch_size = len(batch)
        batch = [sample for sample in batch if not sample.get('_invalid', False)]
        
        if len(batch) < original_batch_size:
            num_dropped = original_batch_size - len(batch)
            logger.debug(
                f"Dropped {num_dropped}/{original_batch_size} samples from batch "
                f"(all items unknown/not in vocabulary)"
            )
        
        # Handle edge case: entire batch was invalid
        if len(batch) == 0:
            logger.error(
                "Entire batch was invalid (all samples had only unknown items). "
                "Returning empty batch with minimal structure."
            )
            # Return a minimal valid batch structure to avoid crashes
            return {
                'cid': [],
                'item_indices': torch.zeros((0, 1), dtype=torch.long),
                'labels': torch.zeros((0, 1), dtype=torch.long),
                'membership': torch.zeros((0, 1, 1), dtype=torch.float32),
                'attention_mask': torch.zeros((0, 1), dtype=torch.bool),
                'seq_length': [],
            }
        
        # Validate that all samples have correct keys
        # This ensures item_ids were converted to item_indices in Dataset.__getitem__
        for i, sample in enumerate(batch):
            if 'item_indices' not in sample:
                raise KeyError(
                    f"Sample {i} missing 'item_indices'. "
                    "Expected 'item_indices' (vocabulary indices), not 'item_ids'. "
                    "Conversion from item_ids to item_indices should happen in Dataset.__getitem__()."
                )
        
        batch_size = len(batch)
        
        # Find max sequence length in this batch
        seq_lengths_list = [sample['seq_length'] for sample in batch]
        max_len = max(seq_lengths_list)
        
        # Debug: log batch shape info
        logger.debug(f"Collating batch: size={batch_size}, seq_lengths={seq_lengths_list}, max_len={max_len}")
        
        # Get membership feature dimension
        membership_dim = batch[0]['membership'].shape[1]
        
        # Initialize padded tensors
        item_indices = torch.full(
            (batch_size, max_len), 
            self.pad_item_idx, 
            dtype=torch.long
        )
        labels = torch.full(
            (batch_size, max_len), 
            self.pad_label_value, 
            dtype=torch.long
        )
        membership = torch.full(
            (batch_size, max_len, membership_dim), 
            self.membership_padding_value, 
            dtype=torch.float32  # Store as float (it's feature data, not indices)
        )
        attention_mask = torch.zeros(
            (batch_size, max_len), 
            dtype=torch.bool
        )
        
        # Collect CIDs and sequence lengths
        cids = []
        seq_lengths = []
        
        # Fill tensors
        for i, sample in enumerate(batch):
            seq_len = sample['seq_length']
            
            item_indices[i, :seq_len] = torch.from_numpy(sample['item_indices'])
            labels[i, :seq_len] = torch.from_numpy(sample['labels'])
            membership[i, :seq_len, :] = torch.from_numpy(sample['membership'])
            attention_mask[i, :seq_len] = True  # True for valid positions
            
            cids.append(sample['cid'])
            seq_lengths.append(seq_len)
            
        attention_mask = attention_mask & (item_indices != 0)
        
        return {
            'cid': cids,
            'item_indices': item_indices,       # [B, max_len], long
            'labels': labels,                   # [B, max_len], long
            'membership': membership,           # [B, max_len, 364], float32 (binary 0/1 features)
            'attention_mask': attention_mask,   # [B, max_len], bool - True for valid, False for padding/item_id=0
            'seq_lengths': torch.tensor(seq_lengths, dtype=torch.long),  # [B]
        }


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config YAML file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_and_split_data(
    config: Dict,
    downloader: Optional[DataDownloader] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Get file lists for train/val/test splits.
    
    **Memory efficient**: Splits at the FILE level, not sample level.
    This avoids loading all data into memory and scales to arbitrarily large datasets.
    
    Args:
        config: Training configuration dictionary
        downloader: Optional DataDownloader instance
        
    Returns:
        Tuple of (train_files, val_files, test_files)
        - train_files: List of training file paths
        - val_files: List of validation file paths (split from train files)
        - test_files: List of test file paths (if test_path exists, else empty list)
        
    Example:
        >>> train_files, val_files, test_files = load_and_split_data(config)
        >>> # If 10 files with val_split_ratio=0.2: 8 files for train, 2 for val
        >>> # Minimum 1 file guaranteed for validation
    """
    data_config = config['data']
    
    # Initialize downloader if needed
    if downloader is None:
        downloader = DataDownloader(temp_dir=data_config['temp_dir'])
    
    # Get training files
    train_path = data_config['train_path']
    train_pattern = data_config.get('train_pattern', '*.parquet')
    max_train_files = data_config.get('max_train_files', None)
    
    all_train_files = downloader.list_files(train_path, train_pattern, max_train_files)
    logger.info(f"Found {len(all_train_files)} training files")
    
    if not all_train_files:
        raise ValueError(f"No training files found at {train_path}")
    
    # Download if GCS
    if data_config['download_from_gcs'] and downloader.is_gcs_path(all_train_files[0]):
        all_train_files = downloader.download_files(all_train_files)
    
        
    train_files = [all_train_files[0]]
    val_files = [all_train_files[1]]

    logger.info(f"hardcoded train and val files:")
    logger.info(f"  Train files: {train_files}")
    logger.info(f"  Val files: {val_files}")

    # Split files into train/val
    # val_split_ratio = data_config.get('val_split_ratio', 0.2)
    # split_seed = data_config.get('split_seed', 42)
    
    # Shuffle files with seed for reproducibility
    # import random
    # rng = random.Random(split_seed)
    # shuffled_files = all_train_files.copy()
    # rng.shuffle(shuffled_files)
    
    # Calculate split - ensure at least 1 file for validation
    # total_files = len(shuffled_files)
    # num_val_files = max(1, int(total_files * val_split_ratio))  # At least 1 file
    # num_train_files = total_files - num_val_files
    
    # logger.info(f"File split (seed={split_seed}):")
    # logger.info(f"  Total files: {total_files}")
    # logger.info(f"  Train files: {num_train_files}")
    # logger.info(f"  Val files: {num_val_files}")
    
    # # Get test files (if they exist) NOT USED DURING TRAINING
    # test_files = []
    # test_path = data_config.get('test_path')
    # if test_path:
    #     test_pattern = data_config.get('test_pattern', '*.parquet')
    #     max_test_files = data_config.get('max_test_files', None)
    #     test_files = downloader.list_files(test_path, test_pattern, max_test_files)
        
    #     if test_files and data_config['download_from_gcs'] and downloader.is_gcs_path(test_files[0]):
    #         test_files = downloader.download_files(test_files)
        
    #     logger.info(f"  Test files: {len(test_files)}")
    
    return train_files, val_files #, test_files


def create_dataloader(
    config: Dict,
    split: str,
    files: List[str],
    vocab: Optional[VocabularyBuilder] = None,
) -> Tuple[DataLoader, Optional[VocabularyBuilder]]:
    """
    Create a single dataloader for a specific split (train/val/test).
    
    **Memory efficient**: Works with file lists from load_and_split_data().
    Data is loaded lazily by the dataset, not all at once.
    
    Args:
        config: Training configuration dictionary (config['training'])
        split: Split name ('train', 'val', or 'test')
        files: List of parquet file paths for this split
        vocab: Pre-built vocabulary. If None and split='train', builds from files.
               If None and split!='train', raises error.
        
    Returns:
        Tuple of (dataloader, vocab_builder)
        - dataloader: PyTorch DataLoader for the split
        - vocab_builder: VocabularyBuilder (only if split='train' and vocab was None, else returns input vocab)
        
    Example:
        >>> # Get file splits first (memory efficient)
        >>> train_files, val_files, test_files = load_and_split_data(config)
        >>> 
        >>> # Create train loader and build vocabulary
        >>> train_loader, vocab = create_dataloader(config, 'train', train_files, vocab=None)
        >>> 
        >>> # Create val loader using the same vocabulary
        >>> val_loader, _ = create_dataloader(config, 'val', val_files, vocab=vocab)
        >>> 
        >>> # Optionally create test loader later
        >>> test_loader, _ = create_dataloader(config, 'test', test_files, vocab=vocab)
    """
    if split not in ['train', 'val', 'test']:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
    
    # Validate inputs
    if vocab is None and split != 'train':
        raise ValueError(f"Vocabulary must be provided for {split} split. Build it from training data first.")
    
    if not files:
        raise ValueError(f"No files provided for {split} split")
    
    # Extract configs
    data_config = config['data']
    vocab_config = config['vocabulary']
    seq_config = config['sequence']
    batch_config = config['batch']
    
    # Get column mapping (for data with different column names)
    column_mapping = data_config.get('column_mapping', None)
    
    # Build or use vocabulary (only for train split)
    vocab_was_none = vocab is None
    if vocab is None:
        # Build from training files
        logger.info(f"Building vocabulary from {len(files)} training files...")
        vocab = VocabularyBuilder()
        stats = vocab.build_from_files(files, column_mapping=column_mapping)
        
        # Save vocabulary if path specified
        save_path = vocab_config.get('save_vocab_path')
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            vocab.save(save_path)
            logger.info(f"Vocabulary saved to {save_path}")
    
    logger.info(f"Using vocabulary with {vocab.vocab_size} items")
    
    # Create dataset
    dataset = RepurchaseDataset(
        vocab=vocab,
        parquet_files=files,
        max_sequence_length=seq_config['max_sequence_length'],
        truncate_strategy=seq_config['truncate_strategy'],
        column_mapping=column_mapping,
    )
    
    # Create collate function
    collate_fn = RepurchaseCollate(
        pad_item_idx=VocabularyBuilder.UNKNOWN_IDX,
        pad_label_value=seq_config['label_padding_value'],
        membership_padding_value=seq_config['membership_padding_value'],
    )
    
    # Determine shuffle
    if split == 'train':
        shuffle = batch_config.get('shuffle_train', True)
    else:
        shuffle = batch_config.get(f'shuffle_{split}', False)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_config['batch_size'],
        shuffle=shuffle,
        num_workers=batch_config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=batch_config.get('pin_memory', True),
        prefetch_factor=batch_config.get('prefetch_factor', 2) if batch_config['num_workers'] > 0 else None,
    )
    
    logger.info(f"{split.upper()} DataLoader created: {len(dataset)} samples, {len(dataloader)} batches")
    
    # Return vocab only if we built it
    return dataloader, vocab if vocab_was_none else vocab


if __name__ == "__main__":
    """Test the dataloader with sample config."""
    import sys
    
    # Setup logging
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    # Test with config
    config_path = "../configs/config.yaml"
    
    try:
        # Load config
        config = load_config(config_path)
        train_config = config['training']
        
        # Split files (memory efficient)
        logger.info("Splitting files...")
        train_files, val_files = load_and_split_data(train_config)
        logger.info(f"Train: {len(train_files)} files, Val: {len(val_files)} files")
        
        # Create train dataloader and build vocabulary
        logger.info("\nCreating train dataloader...")
        train_loader, vocab = create_dataloader(train_config, 'train', train_files, vocab=None)
        
        # Create val dataloader using the same vocabulary
        logger.info("\nCreating val dataloader...")
        val_loader, _ = create_dataloader(train_config, 'val', val_files, vocab=vocab)
        
        # Test batch
        logger.info("\n" + "="*60)
        logger.info("Testing batch loading...")
        logger.info("="*60)
        
        batch = next(iter(train_loader))
        
        logger.info(f"\nBatch shapes:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"  {key}: {value.shape}, dtype={value.dtype}")
            elif isinstance(value, list):
                logger.info(f"  {key}: list of {len(value)} items")
        
        logger.info(f"\nSample statistics:")
        logger.info(f"  Batch size: {batch['item_indices'].shape[0]}")
        logger.info(f"  Max sequence length: {batch['item_indices'].shape[1]}")
        logger.info(f"  Valid items: {batch['attention_mask'].sum().item()}")
        logger.info(f"  Avg sequence length: {batch['seq_lengths'].float().mean().item():.1f}")
        
        # Label distribution
        valid_labels = batch['labels'][batch['attention_mask']]
        logger.info(f"\nLabel distribution (valid positions only):")
        logger.info(f"  Positive (1): {(valid_labels == 1).sum().item()}")
        logger.info(f"  Negative (0): {(valid_labels == 0).sum().item()}")
        logger.info(f"  Positive ratio: {(valid_labels == 1).float().mean().item():.3f}")
        
        logger.info("\n" + "="*60)
        logger.info("Dataloader test completed successfully!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
