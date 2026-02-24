import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from loguru import logger

# temporal cnn
from .temporal_cnn import TemporalCNN

# set transformer
from .set_transformer import SetTransformerEncoder, PMA

# deep set
from .deep_sets import PermEqMeanStack, SetRho


class CategoryEmbedding(nn.Module):
    """
    Embedding layer for binary membership categories (0/1).
    
    Maps each 0/1 value in the membership feature to a learned embedding vector.
    This allows the model to learn distinct representations for "purchased" vs 
    "not purchased" on each day.
    
    Args:
        num_categories: Number of categories (2 for binary {0, 1})
        embedding_dim: Dimension of embedding vectors
        
    Input Shape:
        [batch, seq_len, num_days] with integer values in {0, 1}
        
    Output Shape:
        [batch, seq_len, num_days, embedding_dim]
        
    Example:
        >>> category_emb = CategoryEmbedding(num_categories=2, embedding_dim=256)
        >>> membership = torch.randint(0, 2, (32, 50, 365))  # batch=32, items=50, days=365
        >>> embs = category_emb(membership)
        >>> print(embs.shape)  # torch.Size([32, 50, 365, 256])
    """
    
    def __init__(self, num_categories: int = 2, embedding_dim: int = 256):
        super().__init__()
        self.num_categories = num_categories
        self.embedding_dim = embedding_dim
        
        # Embedding layer for {0, 1} categories
        self.embedding = nn.Embedding(num_categories, embedding_dim)
        
    def forward(self, membership: torch.Tensor) -> torch.Tensor:
        """
        Embed membership categories.
        
        Args:
            membership: [B, S, D] tensor with values in {0, 1}
                       B = batch size, S = sequence length, D = num days
        
        Returns:
            embeddings: [B, S, D, E] embedded representations
                       E = embedding_dim
        """
        # Ensure membership is long type for embedding lookup
        membership = membership.long()
        
        # Embed each value: [B, S, D] -> [B, S, D, E]
        return self.embedding(membership)


class RepurchaseModel(nn.Module):
    """
    Repurchase prediction model combining CNN and Set Transformer.
    
    This model predicts item repurchase behavior by:
    1. Embedding items and learning from temporal membership patterns
    2. Combining item embeddings with CNN-extracted temporal features
    3. Encoding sequences with Set Transformer for permutation-invariant representations
    4. Creating global representations through Set Rho
    
    Architecture Flow:
    ┌─────────────────────────────────────────────────────┐
    │ Input: item_indices, membership, attention_mask     │
    └─────────────────────────────────────────────────────┘
                         │
                         ├─────────────────┬──────────────────┐
                         ▼                 ▼                  ▼
                  Item Embedding    MembershipCNN    CategoryEmbedding
                  [B,S,emb_dim]    [B,S,cnn_dim]    [B,S,365,emb_dim]
                         │                 │                  │
                         └────────┬────────┘                  │
                                  ▼                           ▼
                    Concat: [B,S,emb_dim+cnn_dim]     Sum+Pool: [B,emb_dim]
                                  │                           │
                                  ▼                           ▼
                        SetTransformerEncoder            SetRho
                           [B,S,hidden_dim]           [B,hidden_dim]
                                  │                           │
                                  └─────────┬─────────────────┘
                                            ▼
                                   Scoring/Prediction
    
    Args:
        vocab_size: Vocabulary size (includes padding at index 0)
        embedding_dim: Item embedding dimension
        cnn_output_dim: CNN output dimension
        hidden_dim: Set encoder hidden dimension
        num_heads: Number of attention heads (only for set_phi_type="set_transformer")
        num_inds: Number of induced points for ISAB (only for set_phi_type="set_transformer")
        dropout: Dropout rate
        membership_dim: Length of membership vector (e.g., 365 days)
        cnn_kernel_sizes: Optional dict of CNN kernel sizes per scale
        category_pooling: Global representation method ("average" or "pma")
                         - "average": Average pooling + SetRho (2-layer MLP)
                         - "pma": PMA attention pooling (replaces average + SetRho)
        pma_num_heads: Number of attention heads for PMA (only used if category_pooling="pma")
        set_phi_type: Type of set encoder ("set_transformer" or "perm_eq_mean")
        perm_eq_num_stacks: Number of PermEqMean layers (only used if set_phi_type="perm_eq_mean")
        cnn_layer_enabled: Whether to use temporal CNN (True) or projection baseline (False)
        cnn_output_dim: Output dimension of CNN layer
        scoring_mode: Scoring mechanism ablation mode
                     - "both": Use both intrinsic and compatibility scores (full model)
                     - "intrinsic_only": Use only item-specific intrinsic scores
                     - "compatibility_only": Use only global-item compatibility scores
        
    Example:
        >>> model = RepurchaseModel(vocab_size=2632, embedding_dim=256)
        >>> batch = {
        ...     'item_indices': torch.randint(0, 2632, (32, 50)),
        ...     'membership': torch.randint(0, 2, (32, 50, 365)).float(),
        ...     'attention_mask': torch.ones(32, 50).bool()
        ... }
        >>> output = model(batch)
        >>> print(output['item_representations'].shape)  # torch.Size([32, 50, 256])
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_inds: int = 32,
        dropout: float = 0.1,
        membership_dim: int = 365,
        cnn_kernel_sizes: Optional[Dict[str, int]] = None,
        category_pooling_type: str = "average",  # "average" or "pma"
        pma_num_heads: int = 4,
        set_phi_type: str = "st_encoder",  # "st_encoder" or "perm_eq_mean"
        perm_eq_num_stacks: int = 3,  # Only used if set_phi_type="perm_eq_mean"
        cnn_layer_enabled: bool = True,
        cnn_output_dim: int = 128,
        scoring_mode: str = "both",  # "both", "intrinsic_only", "compatibility_only"

    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.cnn_output_dim = cnn_output_dim
        self.hidden_dim = hidden_dim
        self.membership_dim = membership_dim
        self.category_pooling = category_pooling_type
        self.set_phi_type = set_phi_type
        self.cnn_layer_enabled = cnn_layer_enabled
        self.scoring_mode = scoring_mode
        
        # Validate scoring_mode
        valid_modes = ["both", "intrinsic_only", "compatibility_only"]
        if scoring_mode not in valid_modes:
            raise ValueError(
                f"Invalid scoring_mode: {scoring_mode}. Must be one of {valid_modes}"
            )
        
        logger.info(f"Initializing RepurchaseModel:")
        logger.info(f"  Vocabulary size: {vocab_size}")
        logger.info(f"  Embedding dim: {self.embedding_dim}")
        logger.info(f"  CNN output dim: {self.cnn_output_dim}")
        logger.info(f"  Hidden dim: {hidden_dim}")
        logger.info(f"  Membership dim: {membership_dim}")
        logger.info(f"  Scoring mode: {scoring_mode}")
        
        # 1. Item Embedding (padding_idx=0 for padding and unknown items)
        self.item_embedding = nn.Embedding(
            vocab_size, self.embedding_dim, padding_idx=0
        )
        logger.info(f"  Item embedding: {vocab_size} -> {self.embedding_dim}")
        
        # 2. Category Embedding (for membership 0/1 values)
        self.category_embedding = CategoryEmbedding(
            num_categories=2,  # {0, 1}
            embedding_dim=self.embedding_dim
        )
        logger.info(f"  Category embedding: 2 classes -> {self.embedding_dim}")
        
        # 3. Temporal CNN (on membership vectors)
        if self.cnn_layer_enabled:
            self.membership_cnn = TemporalCNN(
                output_dim=cnn_output_dim,
                num_days=membership_dim,
                kernel_sizes=cnn_kernel_sizes,
                dropout=dropout
            )
            logger.info(f"  Temporal CNN: {membership_dim} days -> {cnn_output_dim}")
        else:
            self.membership_cnn = None
            self.cnn_replace_proj = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim + cnn_output_dim),
                nn.ELU(),
                nn.Linear(embedding_dim + cnn_output_dim, embedding_dim + cnn_output_dim)
            )
            logger.info(f"  Temporal CNN: disabled")
            logger.info(f"  CNN Replacement Projection layer: {embedding_dim} -> {embedding_dim + cnn_output_dim}")
        
        # 4. Set Phi Encoder (Set Transformer or PermEqMean)
        # Input: item_emb + cnn_features
        if set_phi_type == "perm_eq_mean":
            self.set_phi = PermEqMeanStack(
                in_dim=embedding_dim + cnn_output_dim,
                out_dim=hidden_dim,
                num_stacks=perm_eq_num_stacks,
                dropout=dropout
            )
            logger.info(f"  Set Phi (PermEqMean): {embedding_dim + cnn_output_dim} -> {hidden_dim} ({perm_eq_num_stacks} stacks)")
        else:  # set_transformer
            self.set_phi = SetTransformerEncoder(
                dim_input=embedding_dim + cnn_output_dim,
                dim_hidden=hidden_dim,
                num_heads=num_heads,
                num_inds=num_inds,
                dropout=dropout
            )
            logger.info(f"  Set Phi (SetTransformer): {embedding_dim + cnn_output_dim} -> {hidden_dim}")
        
        # 5. Category Pooling + Global Representation
        # Pooling options: "pma" (learned attention) or "average" (masked mean)
        # Both use SetRho for final projection
        
        if category_pooling_type == "pma":
            # PMA: Learned attention pooling
            self.category_pooler = PMA(
                dim=embedding_dim,
                num_heads=pma_num_heads,
                num_seeds=1,  # Single pooled output
                ln=True,
                squeeze=True  # Squeeze to [B, D] instead of [B, 1, D]
            )
            logger.info(f"  Category Pooling: PMA with {pma_num_heads} heads")
        else:
            # Average pooling (masked mean)
            self.category_pooler = self.masked_mean # use masked mean function instead in forward pass
            logger.info(f"  Category Pooling: Average (masked mean)")
        
        # SetRho: Projects pooled representation to hidden_dim (used by both)
        self.set_rho = SetRho(
                in_dim=embedding_dim,
                out_dim=embedding_dim, 
                dropout=dropout
            )
        logger.info(f"  Set Rho Projection: {embedding_dim} -> {embedding_dim}")
        
        # 6. Scoring Components
        # Category scorer: Maps item representations to scalar scores
        self.category_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        logger.info(f"  Category Scorer: {hidden_dim} -> 1 (intrinsic scores)")
        
        # Combination weights: [vocab_size, 2] for [compat_weight, intrinsic_weight]
        # Initialize with [0.5, 0.5] for balanced combination
        self.combination_weights = nn.Parameter(
            torch.full((vocab_size, 2), 0.5, dtype=torch.float32)
        )
        logger.info(f"  Combination Weights: [{vocab_size}, 2] (learnable)")
        
        # 7. Apply weight initialization
        self.apply(self._init_weights)
        logger.info("Model initialization complete")
    
    # @staticmethod
    def masked_mean(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute mean over valid positions (ignoring padding).
        
        Args:
            x: [B, S, D] input tensor
            attention_mask: [B, S] boolean mask (True = valid, False = padding)
        
        Returns:
            mean: [B, D] averaged over valid positions
        """
        mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, S, 1]
        x_masked = x * mask_expanded
        sum_x = x_masked.sum(dim=1)
        count = mask_expanded.sum(dim=1).clamp(min=1.0)
        return sum_x / count
    
    def compute_scores(
        self,
        item_representations: torch.Tensor,  # [B, S, hidden_dim]
        global_representation: torch.Tensor,  # [B, hidden_dim]
        item_indices: torch.Tensor,  # [B, S] - original item indices
        attention_mask: torch.Tensor  # [B, S] - boolean mask
    ) -> torch.Tensor:
        """
        Compute final repurchase scores by combining intrinsic and compatibility scores.
        
        This implements the dual-score mechanism with ablation support:
        1. Intrinsic scores: Item-specific scores from enriched representations
        2. Compatibility scores: Global-item compatibility via dot product
        3. Weighted combination: Learnable weights balance both signals
        
        Ablation modes (controlled by self.scoring_mode):
        - "both": Full model combining intrinsic + compatibility (default)
        - "intrinsic_only": Only use item-specific scores (tests local signal)
        - "compatibility_only": Only use global-item matching (tests global signal)
        
        Args:
            item_representations: [B, S, hidden_dim] - enriched item representations from set_phi
            global_representation: [B, hidden_dim] - global set representation from set_rho
            item_indices: [B, S] - original item indices (vocabulary indices)
            attention_mask: [B, S] - boolean mask (True=valid, False=padding)
        
        Returns:
            final_scores: [B, vocab_size] - repurchase scores for all items
        
        Example:
            >>> model = RepurchaseModel(vocab_size=1000, hidden_dim=256)
            >>> item_reps = torch.randn(32, 50, 256)  # 32 customers, up to 50 items
            >>> global_rep = torch.randn(32, 256)
            >>> indices = torch.randint(0, 1000, (32, 50))
            >>> mask = torch.ones(32, 50).bool()
            >>> scores = model.compute_scores(item_reps, global_rep, indices, mask)
            >>> print(scores.shape)  # torch.Size([32, 1000])
        """
        batch_size, seq_len, hidden_dim = item_representations.shape
        
        # ===== Step 1: Intrinsic Scores =====
        # Compute per-item scores from enriched representations
        i_scores = self.category_scorer(item_representations).squeeze(-1)  # [B, S]
        
        # ===== Step 2: Compatibility Scores =====
        # Dot product of global representation with all item embeddings
        # global_representation: [B, hidden_dim]
        # item_embedding.weight: [vocab_size, embedding_dim]
        # Need to project global_rep back to embedding space for scoring
        
        # Get embedding weights: [vocab_size, embedding_dim]
        embedding_weights = self.item_embedding.weight  # [vocab_size, embedding_dim]
        
        # Compute compatibility scores: [B, vocab_size]
        global_scores = torch.matmul(
            global_representation,  # [B, hidden_dim]
            embedding_weights.t()  # [hidden_dim, vocab_size]
        )  # [B, vocab_size]
        
        # ===== Step 3: Scatter Intrinsic Scores =====
        # Place intrinsic scores into full vocabulary tensor
        # Use scatter_add for gradient-safe operation
        
        # Mask padded items (set padding scores to 0)
        i_scores_masked = i_scores * attention_mask.float()  # [B, S]
        
        # Scatter with gradient tracking
        intrinsic_scores_all = torch.zeros(
            batch_size, self.vocab_size,
            dtype=torch.float32,
            device=item_representations.device
        ).scatter_add_(
            dim=1, 
            index=item_indices,      # [B, S]
            src=i_scores_masked      # [B, S]
        )  # [B, vocab_size]
        
        # ===== Step 4: Create Category Mask =====
        # Binary mask indicating which items are in each customer's history
        # Use scatter_add for gradient-safe operation
        categories_mask = torch.zeros(
            batch_size, self.vocab_size,
            dtype=torch.float32,
            device=item_representations.device
        ).scatter_add_(
            dim=1,
            index=item_indices,           # [B, S]
            src=attention_mask.float()    # [B, S] - Place 1.0 for valid items
        )  # [B, vocab_size]
        
        # ===== Step 5: Combination Weights =====
        # combination_weights: [vocab_size, 2] -> [compat_weight, intrinsic_weight]
        compat_weights = self.combination_weights[:, 0]  # [vocab_size]
        intrinsic_weights = self.combination_weights[:, 1]  # [vocab_size]
        
        # Expand to batch dimension
        compat_weights = compat_weights.unsqueeze(0)  # [1, vocab_size]
        intrinsic_weights = intrinsic_weights.unsqueeze(0)  # [1, vocab_size]
        
        # Mask intrinsic weights: only apply to purchased items
        intrinsic_weights = intrinsic_weights * categories_mask  # [B, vocab_size]
        
        # ===== Step 6: Weighted Combination (with Ablation Support) =====
        weighted_compat = global_scores * compat_weights  # [B, vocab_size]
        weighted_intrinsic = intrinsic_scores_all * intrinsic_weights  # [B, vocab_size]
        
        # Apply scoring mode ablation
        if self.scoring_mode == "both":
            # Full model: combine both intrinsic and compatibility scores
            final_scores = weighted_compat + weighted_intrinsic  # [B, vocab_size]
        elif self.scoring_mode == "intrinsic_only":
            # Ablation: only use intrinsic scores (item-specific)
            # final_scores = weighted_intrinsic  # [B, vocab_size]
            final_scores = intrinsic_scores_all  # [B, vocab_size]
        elif self.scoring_mode == "compatibility_only":
            # Ablation: only use compatibility scores (global-item matching)
            # final_scores = weighted_compat  # [B, vocab_size]
            final_scores = global_scores  # [B, vocab_size]
        else:
            raise ValueError(f"Invalid scoring_mode: {self.scoring_mode}")
        
        # # ===== Step 7: Mask Padding/Unknown Index =====
        # # CRITICAL: Set padding index (0) to -inf to prevent it from being selected
        # # This ensures padding/unknown items never get high scores
        # final_scores[:, 0] = float("-inf")
        
        return final_scores
    
    def _init_weights(self, module):
        """
        Initialize weights following pb_insp_tt pattern.
        
        Args:
            module: PyTorch module to initialize
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for repurchase prediction.
        
        Args:
            batch: Dictionary with keys:
                - 'item_indices': [B, S] - item vocabulary indices
                - 'membership': [B, S, D] - binary membership vectors (D days)
                - 'attention_mask': [B, S] - True for valid positions
                - 'labels' (optional): [B, S] - repurchase labels (0/1)
        
        Returns:
            Dictionary with:
                - 'item_representations': [B, S, hidden_dim] - enriched item representations
                - 'global_representation': [B, hidden_dim] - global set representation
                - 'item_embeddings': [B, S, embedding_dim] - raw item embeddings
                - 'cnn_features': [B, S, cnn_dim] - temporal CNN features
                - 'category_embeddings': [B, S, D, embedding_dim] - category embeddings
                - 'predictions': None (PLACEHOLDER for future implementation)
        """
        # Extract inputs
        item_indices = batch['item_indices']  # [B, S]
        membership = batch['membership']      # [B, S, D] where D = membership_dim
        attention_mask = batch['attention_mask']  # [B, S] - True for valid positions
        
        B, S = item_indices.shape
        D = membership.shape[2]  # Should be membership_dim (365 or 364)
        
        # Validate indices before embedding lookup
        if item_indices.min() < 0:
            raise ValueError(
                f"Negative item indices found! Min: {item_indices.min()}, "
                f"Max: {item_indices.max()}, vocab_size: {self.vocab_size}"
            )
        if item_indices.max() >= self.vocab_size:
            raise ValueError(
                f"Item indices out of range! Max index: {item_indices.max()}, "
                f"vocab_size: {self.vocab_size}. Valid range is [0, {self.vocab_size-1}]"
            )
        
        # 1. Item Embeddings + CNN
        
        # 1a. Get item embeddings
        item_embs = self.item_embedding(item_indices)  # [B, S, emb_dim]
        
        # 1b. Get CNN features from membership
        # TemporalCNN expects [B, S, D] float and returns [B, S, cnn_dim]
        if self.cnn_layer_enabled:
            cnn_features = self.membership_cnn(membership)  # [B, S, cnn_dim]
            combined_embs = torch.cat([item_embs, cnn_features], dim=-1)  # [B, S, emb_dim+cnn_dim]
        else:
            cnn_features = None  # No CNN features when disabled
            combined_embs = self.cnn_replace_proj(item_embs)
        
        # 1d. Pass through Set Transformer Encoder
        item_representations = self.set_phi(
            combined_embs, attention_mask
        )  # [B, S, hidden_dim]
        
        # 2. Category Embeddings + Set Rho
        
        # Validate membership values (should be binary: 0 or 1)
        if membership.min() < 0 or membership.max() > 1:
            raise ValueError(
                f"Invalid membership values! Expected binary (0 or 1) values only. "
                f"Got range: [{membership.min()}, {membership.max()}]. "
                f"Check that membership_padding_value in config is set to 0"
            )
        
        # 2a. Get category embeddings for membership (0/1 values)
        # Convert to long for embedding lookup (treats 0/1 as indices)
        membership_indices = membership.long()
        category_embs = self.category_embedding(membership_indices)  # [B, S, D, emb_dim]
        
        # 2b. Sum across the days dimension
        category_summed = category_embs.sum(dim=2)  # [B, S, emb_dim]
        
        # 2c. Pool to global representation
        category_pooled = self.category_pooler(category_summed, attention_mask)  # [B, emb_dim]
        
        # Project pooled representation to hidden_dim (SetRho used by both)
        global_representation = self.set_rho(category_pooled)  # [B, 1, hidden_dim] or [B, hidden_dim]
        
        # # If SetRho outputs 3D (e.g., from PMA), squeeze to 2D
        # if global_representation.ndim == 3:
        #     global_representation = global_representation.squeeze(1)  # [B, 1, D] → [B, D]
        
        # 3. Final Scoring/Prediction
        # Combine item_representations and global_representation
        # to produce final repurchase scores for all items in vocabulary
        final_scores = self.compute_scores(
            item_representations=item_representations,  # [B, S, hidden_dim]
            global_representation=global_representation,  # [B, hidden_dim]
            item_indices=item_indices,  # [B, S]
            attention_mask=attention_mask  # [B, S]
        )  # [B, vocab_size]
        
        # Return all intermediate and final outputs
        return {
            'item_representations': item_representations,      # [B, S, hidden_dim]
            'global_representation': global_representation,    # [B, hidden_dim]
            'item_embeddings': item_embs,                     # [B, S, embedding_dim]
            'cnn_features': cnn_features,                     # [B, S, cnn_dim] or None
            'category_embeddings': category_embs,             # [B, S, D, embedding_dim]
            'final_scores': final_scores,                     # [B, vocab_size] - repurchase scores
        }
    
    def get_item_embedding(self, item_indices: torch.Tensor) -> torch.Tensor:
        """
        Helper method: Get embeddings for item indices.
        
        Args:
            item_indices: [B, S] or [B] tensor of item indices
        
        Returns:
            embeddings: [B, S, embedding_dim] or [B, embedding_dim]
        """
        return self.item_embedding(item_indices)
    
    def encode_items(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Helper method: Encode items to get representations (without predictions).
        
        Useful for getting item representations during inference or evaluation.
        
        Args:
            batch: Dictionary with 'item_indices', 'membership', 'attention_mask'
        
        Returns:
            item_representations: [B, S, hidden_dim]
        """
        output = self.forward(batch)
        return output['item_representations']
    
    def get_global_representation(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Helper method: Get global set representation.
        
        Args:
            batch: Dictionary with 'item_indices', 'membership', 'attention_mask'
        
        Returns:
            global_representation: [B, hidden_dim]
        """
        output = self.forward(batch)
        return output['global_representation']