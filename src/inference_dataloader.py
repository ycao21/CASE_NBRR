from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from loguru import logger

from .dataloader import VocabularyBuilder  # Import your vocab class

class InferenceRepurchaseDataset(Dataset):
    """
    Dataset for inference (no labels).
    Loads item_ids and membership features only.
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
        membership: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Truncate sequences to max_sequence_length.
        
        For "recent" strategy: Sorts items by their most recent purchase date
        (determined by the largest index with value 1 in the membership vector),
        then keeps the top N most recent items.
        
        Args:
            item_ids: Item ID array
            membership: Membership array [seq_len, 364] where larger index = more recent date
            
        Returns:
            Truncated (item_ids, membership)
        """
        seq_len = len(item_ids)
        
        # ALWAYS log truncation for debugging
        if seq_len > self.max_sequence_length:
            logger.warning(f"Truncating sequence: {seq_len} -> {self.max_sequence_length} (strategy: {self.truncate_strategy})")
        
        if seq_len <= self.max_sequence_length:
            return item_ids, membership
        
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
                    logger.warning(f"an occurence of Item ID {item_ids[i]} has no purchases in membership vector.")
                    last_purchase_indices.append(365)
            
            last_purchase_indices = np.array(last_purchase_indices)
            
            # Sort indices by last purchase date in ascending order (smaller index = more recent)
            sorted_positions = np.argsort(last_purchase_indices)
            
            # Reorder all arrays by recency
            item_ids = item_ids[sorted_positions]
            membership = membership[sorted_positions]
            
            # Keep the first max_sequence_length items (most recent)
            item_ids = item_ids[:self.max_sequence_length]
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
            membership = membership[indices]
        else:
            raise ValueError(f"Unknown truncate_strategy: {self.truncate_strategy}")
        
        return item_ids, membership
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with item_indices, membership, cid
            Note: Unknown items (not in vocabulary) are mapped to index 0
        """
        row = self.data.iloc[idx]
        
        # Get actual column names
        item_col = self.column_mapping.get('item_ids', 'item_ids')
        membership_col = self.column_mapping.get('membership', 'membership')
        
        # Get raw data
        item_ids = np.array(row[item_col], dtype=np.int64)
        membership_raw = row[membership_col]
        if not isinstance(membership_raw, list):
            membership_raw = list(membership_raw)
        # print(membership_raw)
        membership = np.array(membership_raw, dtype=np.float32)  # Shape: [seq_len, 365]
                
        # Truncate if needed
        orig_len = len(item_ids)
        item_ids, membership = self._truncate_sequence(
            item_ids, membership
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
            logger.warning(
                f"Sample with CID {row['cid']} has ALL items unknown (not in vocabulary). "
                f"Original items: {len(item_ids)}, All mapped to index 0. "
                f"This sample will be DROPPED during collation."
            )
            # Mark this sample as invalid by setting a flag
            return {
                'cid': row['cid'],
                'item_indices': item_indices,
                'membership': membership,
                'seq_length': len(item_indices),
                '_invalid': True,  # Flag for collate_fn to drop
            }
        
        return {
            'cid': row['cid'],
            'item_indices': item_indices,
            'membership': membership,
            'seq_length': len(item_indices),
            '_invalid': False,
        }


class InferenceRepurchaseCollate:
    """Collate function for batching with padding."""
    
    def __init__(
        self,
        pad_item_idx: int = 0,
        membership_padding_value: float = 0.0,
    ):
        """
        Initialize collate function.
        
        Args:
            pad_item_idx: Padding index for item_indices (0)
            membership_padding_value: Padding value for membership features (0.0, same as non-member)
                Stored as float since membership is feature data, not indices
        """
        self.pad_item_idx = pad_item_idx
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
            logger.warning(
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
            membership[i, :seq_len, :] = torch.from_numpy(sample['membership'])
            attention_mask[i, :seq_len] = True  # True for valid positions
            
            cids.append(sample['cid'])
            seq_lengths.append(seq_len)

        attention_mask = attention_mask & (item_indices != 0)
        
        return {
            'cid': cids,
            'item_indices': item_indices,       # [B, max_len], long
            'membership': membership,           # [B, max_len, 364], float32 (binary 0/1 features)
            'attention_mask': attention_mask,   # [B, max_len], bool - True for valid, False for padding/item_id=0
            'seq_lengths': torch.tensor(seq_lengths, dtype=torch.long),  # [B]
        }

def create_inference_dataloader(
    config,
    files,
    vocab,
):
    """
    Create inference dataloader (no labels).
    Raises error if vocab is not provided.
    """
    if vocab is None:
        raise ValueError("Vocabulary must be provided for inference. Do not build vocab during inference.")
    data_config = config['data']
    seq_config = config['sequence']
    batch_config = config['batch']
    column_mapping = data_config.get('column_mapping', None)

    dataset = InferenceRepurchaseDataset(
        vocab=vocab,
        parquet_files=files,
        max_sequence_length=seq_config['max_sequence_length'],
        truncate_strategy=seq_config['truncate_strategy'],
        column_mapping=column_mapping,
    )
    collate_fn = InferenceRepurchaseCollate(
        pad_item_idx=VocabularyBuilder.UNKNOWN_IDX,
        membership_padding_value=seq_config['membership_padding_value'],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_config['batch_size'],
        shuffle=False,
        num_workers=batch_config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=batch_config.get('pin_memory', True),
        prefetch_factor=batch_config.get('prefetch_factor', 2) if batch_config['num_workers'] > 0 else None,
    )
    logger.info(f"Inference DataLoader created: {len(dataset)} samples, {len(dataloader)} batches")

    return dataloader