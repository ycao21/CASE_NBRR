import torch
import torch.nn as nn
from typing import Dict, Optional


class TemporalCNN(nn.Module):
    """
    Multi-scale CNN for detecting temporal patterns across different time horizons.
    
    This module extracts temporal features from purchase history by applying
    convolutional filters at different time scales to capture various shopping
    behaviors:
    
    Time Scales:
        - Weekly (7 days): Regular shopping routines, weekend patterns
        - Bi-weekly (14 days): Paycheck cycles, bi-weekly restocking
        - Monthly (28 days): Monthly replenishment, bill cycles
        - Seasonal (90 days): Quarterly behavior, seasonal needs
        - Trend (180 days): Long-term habits, semi-annual patterns
    
    Architecture:
        For each time scale:
        1. Conv1d with stride=kernel_size (non-overlapping windows)
        2. Flatten the output
        3. Concatenate all scales
        4. Dense layers to project to output dimension
    
    Design Rationale:
        - Smaller kernels (3-7 days) need MORE filters → higher pattern diversity
        - Larger kernels (90-180 days) need FEWER filters → more stable patterns
        - Strided convolutions reduce dimensionality while capturing patterns
    
    Args:
        output_dim: Final embedding dimension (default: 128)
        num_days: Length of temporal history in days (default: 365)
        kernel_sizes: Dict mapping scale names to kernel sizes
                     If None, uses default scales (weekly, biweekly, monthly, seasonal, trend)
        dropout: Dropout rate for regularization (default: 0.0)
        name: Layer name (default: 'repurchase_cnn')
    
    Input Shape:
        (batch_size, num_items, num_days)
        - batch_size: Number of users
        - num_items: Variable number of items per user
        - num_days: Purchase count history (e.g., 365 days)
    
    Output Shape:
        (batch_size, num_items, output_dim)
        - Temporal features for each item
    
    Example:
        >>> # Create CNN with default configuration
        >>> cnn = TemporalCNN(output_dim=128, num_days=365)
        >>> 
        >>> # Process purchase counts
        >>> counts = torch.randn(32, 50, 365)  # batch=32, items=50, days=365
        >>> features = cnn(counts)
        >>> print(features.shape)  # torch.Size([32, 50, 128])
        >>> 
        >>> # Create CNN with custom kernel sizes
        >>> custom_kernels = {'weekly': 7, 'monthly': 30, 'quarterly': 90}
        >>> cnn_custom = TemporalCNN(output_dim=64, kernel_sizes=custom_kernels)
    
    Reference:
        Inspired by multi-scale temporal convolutions for time series analysis.
        Adapted for repurchase behavior modeling in e-commerce.
    """
    
    def __init__(
        self,
        output_dim: int = 128,
        num_days: int = 364,
        kernel_sizes: Optional[Dict[str, int]] = None,
        dropout: float = 0.0,
        name: str = 'repurchase_cnn'
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_days = num_days
        self.dropout_rate = dropout
        self.name = name
        
        # Default kernel sizes if not provided
        if kernel_sizes is None:
            kernel_sizes = {
                'weekly': 7,      # Weekly shopping patterns
                'biweekly': 14,   # Bi-weekly paycheck cycles
                'monthly': 28,    # Monthly replenishment
                'seasonal': 91,   # Seasonal/quarterly behavior
                'trend': 182      # Long-term trends
            }
        self.kernel_sizes = kernel_sizes
        
        # Build the CNN architecture
        self._build_cnn()
        
    def _build_cnn(self):
        """
        Build multi-scale convolutional layers and dense projection.
        
        Creates separate Conv1d layers for each time scale, then combines
        them with dense layers for final projection.
        """
        # Create convolution layers for each time scale
        self.convolutions = nn.ModuleDict()
        
        for scale_name, kernel_size in self.kernel_sizes.items():
            # Strided convolution to downsample
            # PyTorch Conv1d: expects (batch, channels, length)
            self.convolutions[scale_name] = nn.Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=kernel_size,
                stride=kernel_size,  # Non-overlapping windows
                padding=0 #kernel_size // 2  # 'same' padding approximation
            )
        
        # ReLU activation
        self.relu = nn.ReLU()
        
        # Dense layers for final projection
        self.dense1 = nn.Linear(
            in_features=self._calculate_total_features(),
            out_features=self.output_dim
        )
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)
        self.dense2 = nn.Linear(
            in_features=self.output_dim,
            out_features=self.output_dim
        )
        
    def _calculate_total_features(self) -> int:
        """
        Calculate the total number of features after all convolutions.
        
        Returns:
            Total feature dimension after concatenating all scales
        """
        total_features = 0
        for kernel_size in self.kernel_sizes.values():
            # Calculate output length after convolution with 'same' padding
            # Output length ≈ ceil(input_length / stride)
            output_length = self.num_days // kernel_size
            total_features += output_length
        return total_features
    
    def forward(self, counts: torch.Tensor) -> torch.Tensor:
        """
        Process temporal purchase patterns through multi-scale CNN.
        
        Args:
            counts: Purchase count history tensor [batch_size, num_items, num_days]
                   Values represent purchase counts/activity per day
        
        Returns:
            features: Temporal feature embeddings [batch_size, num_items, output_dim]
        
        Processing Steps:
            1. Reshape to process all items together: [batch*items, 1, days]
            2. Apply convolutions at each time scale
            3. Flatten and concatenate all scales
            4. Project through dense layers
            5. Reshape back to [batch, items, output_dim]
        """
        batch_size = counts.shape[0]
        num_items = counts.shape[1]
        actual_days = counts.shape[2]  # Get actual number of days from input
        
        # Validate dimension - fail fast if mismatch
        if actual_days != self.num_days:
            raise ValueError(
                f"Membership dimension mismatch! "
                f"Model expects {self.num_days} days but received {actual_days} days. "
                f"Please update config['model']['membership_dim'] to {actual_days} "
                f"or ensure your data has {self.num_days} days per item."
            )
        
        # Reshape for Conv1d: [batch*items, 1, days]
        # PyTorch Conv1d expects (batch, channels, length)
        counts_reshaped = counts.reshape(-1, 1, self.num_days)
        
        # Apply convolutions at each scale and collect outputs
        scale_features = []
        for scale_name in sorted(self.kernel_sizes.keys()):  # Sort for consistency
            # Apply convolution: [batch*items, 1, downsampled_days]
            conv_out = self.convolutions[scale_name](counts_reshaped)
            conv_out = self.relu(conv_out)
            
            # Flatten: [batch*items, downsampled_days]
            flattened = conv_out.squeeze(1)  # Remove channel dimension
            scale_features.append(flattened)
        
        # Concatenate all temporal scales: [batch*items, total_features]
        all_features = torch.cat(scale_features, dim=-1)
        
        # Dense projection to output dimension
        dense1_out = self.dense1(all_features)
        dense1_out = self.relu(dense1_out)
        dense1_out = self.dropout_layer(dense1_out)
        output = self.dense2(dense1_out)
        output = self.relu(output)
        
        # Reshape back to [batch, num_items, output_dim]
        output = output.reshape(batch_size, num_items, self.output_dim)
        
        return output


def build_temporal_cnn_model(
    output_dim: int = 128,
    num_days: int = 365,
    kernel_sizes: Optional[Dict[str, int]] = None
) -> nn.Module:
    """
    Factory function to build TemporalCNN as a standalone model.
    
    This is a convenience function that returns the TemporalCNN module
    directly, matching the TensorFlow functional API pattern.
    
    Args:
        output_dim: Output dimension
        num_days: Number of days in history
        kernel_sizes: Dict of kernel sizes per scale
    
    Returns:
        TemporalCNN module
    
    Example:
        >>> model = build_temporal_cnn_model(output_dim=128, num_days=365)
        >>> counts = torch.randn(32, 50, 365)
        >>> features = model(counts)
        >>> print(features.shape)  # torch.Size([32, 50, 128])
    """
    model = TemporalCNN(
        output_dim=output_dim,
        num_days=num_days,
        kernel_sizes=kernel_sizes
    )
    return model
