import torch
import torch.nn as nn
from typing import Optional


class PermEqMean(nn.Module):
    """
    Permutation Equivariant Mean Layer.
    
    This layer creates a permutation-equivariant transformation by:
    1. Computing the mean across the sequence (permutation invariant)
    2. Projecting both the input and the mean
    3. Subtracting the projected mean from the projected input
    
    The subtraction of the mean ensures permutation equivariance:
    - If you permute the input sequence, the output is permuted identically
    - The mean is the same regardless of order
    
    This is computationally cheaper than attention mechanisms while still
    capturing set structure.
    
    Args:
        out_dim: Output dimension
        
    Input Shape:
        [batch, seq_len, in_dim] or [batch, seq_len, in_dim] with optional weights
        
    Output Shape:
        [batch, seq_len, out_dim]
        
    Example:
        >>> layer = PermEqMean(out_dim=128)
        >>> x = torch.randn(32, 50, 128)
        >>> out = layer(x)
        >>> print(out.shape)  # torch.Size([32, 50, 128])
    """
    
    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim
        
        # Gamma: projects input features
        self.gamma = nn.Linear(out_dim, out_dim, bias=True)
        
        # Lambda: projects mean features (no bias to maintain zero-centering)
        self.lambda_proj = nn.Linear(out_dim, out_dim, bias=False)
    
    def forward(
        self, 
        x: torch.Tensor, 
        weights: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [B, S, D] input tensor
            weights: [B, S] optional weights for weighted mean (if provided, should sum to 1)
            attention_mask: [B, S] boolean mask (True = valid, False = padding)
        
        Returns:
            [B, S, D] transformed tensor with mean subtracted
        """
        B, S, D = x.shape
        
        # Compute mean (handle padding if attention_mask provided)
        if weights is not None:
            # Weighted mean (weights should be normalized to sum to 1)
            xm = (x * weights.unsqueeze(-1)).sum(dim=1, keepdim=True)  # [B, 1, D]
        elif attention_mask is not None:
            # Masked mean (only average over valid positions)
            mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, S, 1]
            xm = (x * mask_expanded).sum(dim=1, keepdim=True)  # [B, 1, D]
            count = mask_expanded.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1, 1]
            xm = xm / count
        else:
            # Simple mean
            xm = x.mean(dim=1, keepdim=True)  # [B, 1, D]
        
        # Project mean
        xm = self.lambda_proj(xm)  # [B, 1, D]
        
        # Project input
        x_proj = self.gamma(x)  # [B, S, D]
        
        # Subtract projected mean from projected input
        # This makes the layer permutation equivariant
        output = x_proj - xm  # [B, S, D]
        
        return output


class PermEqMeanStack(nn.Module):
    """
    Stack of Permutation Equivariant Mean Layers with ELU activation.
    
    Stacks multiple PermEqMean layers to create deeper permutation-equivariant
    representations. Each layer is followed by ELU activation.
    
    This provides a simpler alternative to Set Transformer:
    - Fewer parameters (no attention mechanism)
    - Faster computation (O(D²) vs O(S²D) for attention)
    - Still permutation equivariant
    - Good for cases where attention may be overkill
    
    Args:
        in_dim: Input dimension
        out_dim: Output dimension (same for all layers)
        num_stacks: Number of PermEqMean layers to stack (default: 3)
        dropout: Dropout rate (default: 0.1)
        
    Example:
        >>> encoder = PermEqMeanStack(in_dim=384, out_dim=256, num_stacks=3)
        >>> x = torch.randn(32, 50, 384)
        >>> mask = torch.ones(32, 50).bool()
        >>> out = encoder(x, mask)
        >>> print(out.shape)  # torch.Size([32, 50, 256])
    """
    
    def __init__(
        self, 
        in_dim: int,
        out_dim: int, 
        num_stacks: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        assert num_stacks > 0, "num_stacks must be positive"
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_stacks = num_stacks
        
        # Input projection if dimensions don't match
        if in_dim != out_dim:
            self.input_proj = nn.Linear(in_dim, out_dim)
        else:
            self.input_proj = None
        
        # Stack of PermEqMean layers
        self.stacks = nn.ModuleList([
            PermEqMean(out_dim) for _ in range(num_stacks)
        ])
        
        # Activation
        self.activation = nn.ELU()
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through stacked PermEqMean layers.
        
        Args:
            x: [B, S, in_dim] input tensor
            attention_mask: [B, S] boolean mask (True = valid, False = padding)
            weights: [B, S] optional weights for weighted mean
        
        Returns:
            [B, S, out_dim] encoded tensor
        """
        # Project input if needed
        if self.input_proj is not None:
            x = self.input_proj(x)
        
        # Normalize weights if provided
        if weights is not None:
            if attention_mask is not None:
                # Zero out weights for padded positions
                weights = weights.masked_fill(~attention_mask, 0.0)
            
            # Normalize to sum to 1 per sequence
            weight_sum = weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
            weights = weights / weight_sum
        
        # Pass through stacked layers
        for layer in self.stacks:
            x = layer(x, weights=weights, attention_mask=attention_mask)
            x = self.activation(x)
            x = self.dropout(x)
        
        return x


# class SetPhi(nn.Module):
#     """
#     Set Phi layer: Element-wise encoding for set elements.
    
#     Implements a permutation-equivariant transformation by applying
#     the same MLP to each element independently:
#         φ(x_i) for each element x_i in the set
    
#     This is the simplest form of permutation equivariance:
#     - No aggregation (unlike PermEqMean which subtracts mean)
#     - No attention (unlike SetTransformer)
#     - Just element-wise MLP
    
#     Useful as a preprocessing step before aggregation or attention.
    
#     Based on the Deep Sets φ (phi) function.
    
#     Args:
#         in_dim: Input dimension
#         out_dim: Output dimension
#         hidden_dim: Hidden dimension for MLP (default: same as out_dim)
#         dropout: Dropout rate (default: 0.1)
#         num_layers: Number of MLP layers (default: 2)
        
#     Input Shape:
#         [batch, seq_len, in_dim] - set of elements
        
#     Output Shape:
#         [batch, seq_len, out_dim] - transformed elements
        
#     Example:
#         >>> set_phi = SetPhi(in_dim=128, out_dim=256, hidden_dim=256)
#         >>> x = torch.randn(32, 50, 128)
#         >>> out = set_phi(x)
#         >>> print(out.shape)  # torch.Size([32, 50, 256])
#     """
    
#     def __init__(
#         self, 
#         in_dim: int, 
#         out_dim: int,
#         hidden_dim: Optional[int] = None,
#         dropout: float = 0.1,
#         num_layers: int = 2
#     ):
#         super().__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.hidden_dim = hidden_dim or out_dim
#         self.num_layers = num_layers
        
#         assert num_layers >= 1, "num_layers must be at least 1"
        
#         # Build MLP layers
#         layers = []
#         current_dim = in_dim
        
#         for i in range(num_layers - 1):
#             # Hidden layers with activation and dropout
#             layers.extend([
#                 nn.Linear(current_dim, self.hidden_dim),
#                 nn.ELU(),
#                 nn.Dropout(dropout)
#             ])
#             current_dim = self.hidden_dim
        
#         # Final layer (no activation after last layer)
#         layers.append(nn.Linear(current_dim, out_dim))
        
#         self.phi = nn.Sequential(*layers)
    
#     def forward(
#         self, 
#         x: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None
#     ) -> torch.Tensor:
#         """
#         Apply element-wise encoding to set elements.
        
#         Args:
#             x: [B, S, in_dim] input set elements
#             attention_mask: [B, S] optional mask (not used in computation,
#                            kept for API compatibility with other encoders)
        
#         Returns:
#             [B, S, out_dim] encoded elements
#         """
#         # Apply same MLP to each element independently
#         # This is permutation equivariant because we apply the same function
#         # to each element regardless of position
#         return self.phi(x)


class SetRho(nn.Module):
    """
    Set Rho layer: Projects global set representation.
    
    Implements a two-layer MLP with residual connection:
        rho1(x) -> dropout -> rho2(x)
    
    This is used to transform the pooled set representation into a 
    global representation suitable for scoring.
    
    Based on the TensorFlow SetRhos implementation from the reference model.
    
    Args:
        in_dim: Input dimension
        out_dim: Output dimension
        dropout: Dropout rate (default: 0.1)
        
    Input Shape:
        [batch, in_dim] - pooled set representation
        
    Output Shape:
        [batch, out_dim] - projected global representation
        
    Example:
        >>> set_rho = SetRho(in_dim=256, out_dim=256, dropout=0.1)
        >>> pooled = torch.randn(32, 256)
        >>> global_rep = set_rho(pooled)
        >>> print(global_rep.shape)  # torch.Size([32, 256])
    """
    
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # First projection: 2-layer MLP (rho1)
        self.rho1 = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ELU(),
            nn.Linear(out_dim, out_dim)
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Final projection (rho2)
        self.rho2 = nn.Linear(out_dim, out_dim)
        
    def forward(self, pooled: torch.Tensor, training: bool = None) -> torch.Tensor:
        """
        Project pooled set representation.
        
        Args:
            pooled: [B, in_dim] pooled set representation
            training: Whether in training mode (for dropout). If None, uses self.training
        
        Returns:
            projected: [B, out_dim] projected representation
        """
        # Use module's training state if not explicitly provided
        if training is None:
            training = self.training
            
        # First projection
        x = self.rho1(pooled)
        
        # Dropout (only in training)
        if training:
            x = self.dropout(x)
        
        # Final projection
        x = self.rho2(x)
        
        return x


# For backward compatibility / convenience
def build_perm_eq_mean_stack(
    in_dim: int,
    out_dim: int,
    num_stacks: int = 3,
    dropout: float = 0.1
) -> nn.Module:
    """
    Factory function to build PermEqMeanStack.
    
    Args:
        in_dim: Input dimension
        out_dim: Output dimension
        num_stacks: Number of layers
        dropout: Dropout rate
    
    Returns:
        PermEqMeanStack module
    """
    return PermEqMeanStack(in_dim, out_dim, num_stacks, dropout)




# class PermEqMax(nn.Module):
#     """
#     Permutation Equivariant Max Layer.
    
#     Similar to PermEqMean but uses max pooling instead of mean pooling.
#     This captures the most prominent features rather than average features.
    
#     The max operation is permutation invariant, so the output remains
#     permutation equivariant through the subtraction.
    
#     Useful when:
#     - Sparse patterns are important (e.g., one dominant item)
#     - Want to focus on maximum activations
#     - Robust to outliers in different ways than mean
    
#     Args:
#         out_dim: Output dimension
        
#     Example:
#         >>> layer = PermEqMax(out_dim=128)
#         >>> x = torch.randn(32, 50, 128)
#         >>> out = layer(x)
#         >>> print(out.shape)  # torch.Size([32, 50, 128])
#     """
    
#     def __init__(self, out_dim: int):
#         super().__init__()
#         self.out_dim = out_dim
        
#         # Gamma: projects input features
#         self.gamma = nn.Linear(out_dim, out_dim, bias=True)
        
#         # Lambda: projects max features (no bias to maintain centering)
#         self.lambda_proj = nn.Linear(out_dim, out_dim, bias=False)
    
#     def forward(
#         self, 
#         x: torch.Tensor, 
#         weights: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None
#     ) -> torch.Tensor:
#         """
#         Forward pass using max pooling.
        
#         Args:
#             x: [B, S, D] input tensor
#             weights: [B, S] optional weights (not used for max, kept for API compatibility)
#             attention_mask: [B, S] boolean mask (True = valid, False = padding)
        
#         Returns:
#             [B, S, D] transformed tensor with max subtracted
#         """
#         B, S, D = x.shape
        
#         # Compute max (handle padding if attention_mask provided)
#         if attention_mask is not None:
#             # Masked max (only consider valid positions)
#             # Set padded positions to -inf so they don't affect max
#             mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, S, 1]
#             x_masked = x.clone()
#             x_masked[~attention_mask] = float('-inf')
#             xm = x_masked.max(dim=1, keepdim=True)[0]  # [B, 1, D]
#             # Handle case where all positions are padded (max would be -inf)
#             xm = torch.where(torch.isinf(xm), torch.zeros_like(xm), xm)
#         else:
#             # Simple max
#             xm = x.max(dim=1, keepdim=True)[0]  # [B, 1, D]
        
#         # Project max
#         xm = self.lambda_proj(xm)  # [B, 1, D]
        
#         # Project input
#         x_proj = self.gamma(x)  # [B, S, D]
        
#         # Subtract projected max from projected input
#         output = x_proj - xm  # [B, S, D]
        
#         return output




# class PermEqMaxStack(nn.Module):
#     """
#     Stack of Permutation Equivariant Max Layers with ELU activation.
    
#     Similar to PermEqMeanStack but uses max pooling instead of mean pooling.
#     Good for capturing dominant features in sparse patterns.
    
#     Args:
#         in_dim: Input dimension
#         out_dim: Output dimension (same for all layers)
#         num_stacks: Number of PermEqMax layers to stack (default: 3)
#         dropout: Dropout rate (default: 0.1)
        
#     Example:
#         >>> encoder = PermEqMaxStack(in_dim=384, out_dim=256, num_stacks=3)
#         >>> x = torch.randn(32, 50, 384)
#         >>> mask = torch.ones(32, 50).bool()
#         >>> out = encoder(x, mask)
#         >>> print(out.shape)  # torch.Size([32, 50, 256])
#     """
    
#     def __init__(
#         self, 
#         in_dim: int,
#         out_dim: int, 
#         num_stacks: int = 3,
#         dropout: float = 0.1
#     ):
#         super().__init__()
#         assert num_stacks > 0, "num_stacks must be positive"
        
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.num_stacks = num_stacks
        
#         # Input projection if dimensions don't match
#         if in_dim != out_dim:
#             self.input_proj = nn.Linear(in_dim, out_dim)
#         else:
#             self.input_proj = None
        
#         # Stack of PermEqMax layers
#         self.stacks = nn.ModuleList([
#             PermEqMax(out_dim) for _ in range(num_stacks)
#         ])
        
#         # Activation
#         self.activation = nn.ELU()
        
#         # Dropout
#         self.dropout = nn.Dropout(dropout)
    
#     def forward(
#         self, 
#         x: torch.Tensor, 
#         attention_mask: Optional[torch.Tensor] = None,
#         weights: Optional[torch.Tensor] = None
#     ) -> torch.Tensor:
#         """
#         Forward pass through stacked PermEqMax layers.
        
#         Args:
#             x: [B, S, in_dim] input tensor
#             attention_mask: [B, S] boolean mask (True = valid, False = padding)
#             weights: [B, S] optional weights (not used for max)
        
#         Returns:
#             [B, S, out_dim] encoded tensor
#         """
#         # Project input if needed
#         if self.input_proj is not None:
#             x = self.input_proj(x)
        
#         # Pass through stacked layers
#         for layer in self.stacks:
#             x = layer(x, weights=weights, attention_mask=attention_mask)
#             x = self.activation(x)
#             x = self.dropout(x)
        
#         return x


# class DeepSetsPooling(nn.Module):
#     """
#     Classic Deep Sets architecture: ρ(agg(φ(x_i)))
    
#     This is the canonical Deep Sets formula for permutation-invariant functions:
#     - φ (phi): element-wise encoding
#     - agg: permutation-invariant aggregation (sum/mean/max)
#     - ρ (rho): aggregated encoding
    
#     Unlike PermEqMean/PermEqMax which are equivariant (set → set),
#     this is invariant (set → vector), so it's used for global pooling.
    
#     Args:
#         in_dim: Input dimension
#         hidden_dim: Hidden dimension for phi and rho networks
#         out_dim: Output dimension
#         pooling: Aggregation method ("sum", "mean", or "max")
#         dropout: Dropout rate
        
#     Example:
#         >>> pooler = DeepSetsPooling(in_dim=256, hidden_dim=128, out_dim=256, pooling="mean")
#         >>> x = torch.randn(32, 50, 256)
#         >>> mask = torch.ones(32, 50).bool()
#         >>> out = pooler(x, mask)
#         >>> print(out.shape)  # torch.Size([32, 256])
#     """
    
#     def __init__(
#         self,
#         in_dim: int,
#         hidden_dim: int,
#         out_dim: int,
#         pooling: str = "mean",
#         dropout: float = 0.1
#     ):
#         super().__init__()
#         assert pooling in ["sum", "mean", "max"], f"pooling must be sum/mean/max, got {pooling}"
        
#         self.in_dim = in_dim
#         self.hidden_dim = hidden_dim
#         self.out_dim = out_dim
#         self.pooling = pooling
        
#         # φ (phi): element-wise encoding
#         self.phi = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim),
#             nn.ELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ELU(),
#         )
        
#         # ρ (rho): aggregated encoding
#         self.rho = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, out_dim)
#         )
    
#     def forward(
#         self, 
#         x: torch.Tensor, 
#         attention_mask: Optional[torch.Tensor] = None
#     ) -> torch.Tensor:
#         """
#         Forward pass: ρ(agg(φ(x)))
        
#         Args:
#             x: [B, S, in_dim] input tensor
#             attention_mask: [B, S] boolean mask (True = valid, False = padding)
        
#         Returns:
#             [B, out_dim] pooled representation
#         """
#         B, S, D = x.shape
        
#         # φ: element-wise encoding
#         x_encoded = self.phi(x)  # [B, S, hidden_dim]
        
#         # Aggregation (permutation invariant)
#         if self.pooling == "sum":
#             if attention_mask is not None:
#                 mask_expanded = attention_mask.unsqueeze(-1).float()
#                 x_agg = (x_encoded * mask_expanded).sum(dim=1)  # [B, hidden_dim]
#             else:
#                 x_agg = x_encoded.sum(dim=1)
        
#         elif self.pooling == "mean":
#             if attention_mask is not None:
#                 mask_expanded = attention_mask.unsqueeze(-1).float()
#                 x_sum = (x_encoded * mask_expanded).sum(dim=1)
#                 count = mask_expanded.sum(dim=1).clamp(min=1)
#                 x_agg = x_sum / count  # [B, hidden_dim]
#             else:
#                 x_agg = x_encoded.mean(dim=1)
        
#         else:  # max
#             if attention_mask is not None:
#                 x_masked = x_encoded.clone()
#                 x_masked[~attention_mask] = float('-inf')
#                 x_agg = x_masked.max(dim=1)[0]  # [B, hidden_dim]
#                 # Handle all-padded sequences
#                 x_agg = torch.where(torch.isinf(x_agg), torch.zeros_like(x_agg), x_agg)
#             else:
#                 x_agg = x_encoded.max(dim=1)[0]
        
#         # ρ: aggregated encoding
#         output = self.rho(x_agg)  # [B, out_dim]
        
#         return output