import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from loguru import logger

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, key_padding_mask: Optional[torch.Tensor] = None):
        """
        Args:
            Q: Query tensor [B, S_q, D]
            K: Key/Value tensor [B, S_k, D]
            key_padding_mask: [B, S_k] boolean mask where True means "ignore this position"
        """
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        # Compute attention scores
        scores = Q_.bmm(K_.transpose(1,2)) / math.sqrt(self.dim_V)  # [B*num_heads, S_q, S_k]
        
        # Apply mask if provided
        if key_padding_mask is not None:
            # Expand mask for multi-head attention
            # key_padding_mask: [B, S_k] -> [B*num_heads, 1, S_k]
            mask_expanded = key_padding_mask.repeat(self.num_heads, 1).unsqueeze(1)
            # Set masked positions to -inf so they become 0 after softmax
            scores = scores.masked_fill(mask_expanded, float('-inf'))
        
        # Softmax with numerical stability fix
        # If all positions are masked (-inf), softmax returns NaN. Handle this case:
        A = torch.softmax(scores, dim=2)
        
        # Check for NaN and warn (indicates all keys were masked for some queries)
        if torch.isnan(A).any():
            logger.debug(
                "MAB attention produced NaN values (all keys masked for some queries). "
                "Replacing with zeros. This may indicate data quality issues or incorrect masking."
            )
        
        # Replace NaN with 0 (happens when all keys are masked)
        A = torch.nan_to_num(A, nan=0.0)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


# class SAB(nn.Module):
#     def __init__(self, dim_in, dim_out, num_heads, ln=False):
#         super(SAB, self).__init__()
#         self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

#     def forward(self, X):
#         return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X, attention_mask: Optional[torch.Tensor] = None):
        """
        Args:
            X: Input tensor [B, S, D]
            attention_mask: [B, S] boolean mask where True = valid, False = padding
        
        Returns:
            Output tensor [B, S, D]
        """
        key_padding_mask = None
        if attention_mask is not None:
            # Invert mask: attention_mask (True=valid) -> key_padding_mask (True=ignore)
            key_padding_mask = ~attention_mask
        
        # First MAB: induced points attend to input X
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, key_padding_mask=key_padding_mask)
        
        # Second MAB: input X attends to induced points (no masking needed for H)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False, squeeze=True):
        super(PMA, self).__init__()
        self.num_seeds = num_seeds
        self.squeeze = squeeze
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X, attention_mask: Optional[torch.Tensor] = None):
        """
        Args:
            X: Input tensor [B, S, D]
            attention_mask: [B, S] boolean mask where True = valid, False = padding
        
        Returns:
            Output tensor [B, num_seeds, D]
        """
        key_padding_mask = None
        if attention_mask is not None:
            # Invert mask: attention_mask (True=valid) -> key_padding_mask (True=ignore)
            key_padding_mask = ~attention_mask  # [B, S] bool

        out = self.mab(self.S.repeat(X.size(0), 1, 1), X, key_padding_mask=key_padding_mask)

        if self.squeeze: 
            if self.num_seeds == 1:
                out = out.squeeze(1)  # [B, 1, D] -> [B, D]
            else:
                raise ValueError(f"num_seeds must be 1 if squeeze is True, but got {self.num_seeds}")
        return out


# class SetTransformer(nn.Module):
#     def __init__(self, dim_input, num_outputs, dim_output,
#             num_inds=32, dim_hidden=128, num_heads=4, ln=False):
#         super(SetTransformer, self).__init__()
#         self.enc = nn.Sequential(
#                 ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
#                 ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
#         self.dec = nn.Sequential(
#                 PMA(dim_hidden, num_heads, num_outputs, ln=ln),
#                 SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
#                 SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
#                 nn.Linear(dim_hidden, dim_output))

#     def forward(self, X):
#         return self.dec(self.enc(X))



class SetTransformerEncoder(nn.Module):
    """
    Set Transformer encoder using Induced Set Attention Blocks (ISAB).
    
    This is a simple wrapper around the encoder part of the existing SetTransformer.
    We only need the encoder (ISAB layers) to preserve the full sequence, not the
    decoder which would reduce it to a fixed output size via PMA.
    
    Architecture:
        ISAB -> Dropout -> ISAB -> Dropout
    
    Args:
        dim_input: Input dimension (item_emb_dim + cnn_dim)
        dim_hidden: Hidden dimension for attention
        num_heads: Number of attention heads
        num_inds: Number of induced points for ISAB
        dropout: Dropout rate
        layer_norm: Whether to use layer normalization
        
    Example:
        >>> encoder = SetTransformerEncoder(dim_input=384, dim_hidden=256, num_heads=4, num_inds=32)
        >>> x = torch.randn(32, 50, 384)
        >>> out = encoder(x)
        >>> print(out.shape)  # torch.Size([32, 50, 256])
    """
    
    def __init__(
        self,
        dim_input: int,
        dim_hidden: int,
        num_heads: int = 4,
        num_inds: int = 32,
        dropout: float = 0.1,
        layer_norm: bool = True
    ):
        super().__init__()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        
        # Use ISAB blocks directly (not Sequential so we can pass mask)
        self.isab1 = ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=layer_norm)
        self.dropout1 = nn.Dropout(dropout)
        self.isab2 = ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=layer_norm)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Encode item sequence with set transformer.
        
        Args:
            x: [B, S, D] input embeddings
            attention_mask: [B, S] boolean mask (True = valid, False = padding)
        
        Returns:
            encoded: [B, S, dim_hidden] encoded representations
        """
        # Pass attention_mask through ISAB layers
        x = self.isab1(x, attention_mask=attention_mask)
        x = self.dropout1(x)
        x = self.isab2(x, attention_mask=attention_mask)
        x = self.dropout2(x)
        return x

# class SetTransformerDecoder(nn.Module):
#     """
#     Set Transformer decoder using Pooled Multihead Attention (PMA).
    
#     Pools a variable-length set into a fixed number of output vectors using
#     learnable seed vectors and attention. This is permutation invariant and
#     provides an alternative to simple mean/max pooling.
    
#     Architecture:
#         PMA (pooling) -> Optional SAB processing -> Linear projection
    
#     The PMA uses learnable seed vectors to query the input set and aggregate
#     information via attention, producing a fixed-size output regardless of
#     input sequence length.
    
#     Args:
#         dim_input: Input dimension from encoder
#         dim_output: Output dimension
#         num_outputs: Number of output vectors (typically 1 for global representation)
#         num_heads: Number of attention heads (default: 4)
#         layer_norm: Whether to use layer normalization (default: True)
#         num_sab_layers: Number of SAB layers after PMA (default: 0 for simple decoder)
        
#     Input Shape:
#         [batch, seq_len, dim_input] - encoded set elements
        
#     Output Shape:
#         [batch, num_outputs, dim_output] - pooled representation
        
#     Example:
#         >>> # Simple decoder (just PMA + linear)
#         >>> decoder = SetTransformerDecoder(dim_input=256, dim_output=512, num_outputs=1)
#         >>> x = torch.randn(32, 50, 256)
#         >>> out = decoder(x)
#         >>> print(out.shape)  # torch.Size([32, 1, 512])
#         >>> global_repr = out.squeeze(1)  # [32, 512]
        
#         >>> # Decoder with SAB processing
#         >>> decoder = SetTransformerDecoder(dim_input=256, dim_output=512, num_outputs=1, num_sab_layers=2)
#         >>> out = decoder(x)
#         >>> print(out.shape)  # torch.Size([32, 1, 512])
#     """
    
#     def __init__(
#         self,
#         dim_input: int,
#         dim_output: int,
#         num_outputs: int = 1,
#         num_heads: int = 4,
#         layer_norm: bool = True,
#         num_sab_layers: int = 0
#     ):
#         super().__init__()
#         self.dim_input = dim_input
#         self.dim_output = dim_output
#         self.num_outputs = num_outputs
#         self.num_sab_layers = num_sab_layers
        
#         # # PMA: Pool sequence to fixed num_outputs
#         # self.pma = PMA(
#         #     dim=dim_input,
#         #     num_heads=num_heads,
#         #     num_seeds=num_outputs,
#         #     ln=layer_norm
#         # )
        
#         # Optional SAB layers for processing pooled outputs
#         if num_sab_layers > 0:
#             # Note: SAB is commented out in original code, so we'll use MAB instead
#             # SAB(X) = MAB(X, X)
#             self.sab_layers = nn.ModuleList([
#                 MAB(dim_input, dim_input, dim_input, num_heads, ln=layer_norm)
#                 for _ in range(num_sab_layers)
#             ])
#         else:
#             self.sab_layers = None
        
#         # Final projection to output dimension
#         self.output_proj = nn.Linear(dim_input, dim_output)
    
#     def forward(
#         self, 
#         x: torch.Tensor, 
#         attention_mask: torch.Tensor = None
#     ) -> torch.Tensor:
#         """
#         Pool and decode set representation.
        
#         Args:
#             x: [B, S, dim_input] encoded set elements
#             attention_mask: [B, S] boolean mask (True = valid, False = padding)
        
#         Returns:
#             out: [B, num_outputs, dim_output] decoded representation
#         """
#         # PMA pooling: [B, S, D] -> [B, num_outputs, D]
#         # Pass attention_mask so PMA can ignore padded positions
#         # x = self.pma(x, attention_mask=attention_mask)
        
#         # Optional SAB processing
#         if self.sab_layers is not None:
#             for sab in self.sab_layers:
#                 x = sab(x, x)  # Self-attention on pooled outputs (no masking needed)
        
#         # Project to output dimension
#         x = self.output_proj(x)  # [B, num_outputs, dim_output]
        
#         return x