import torch
import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from .UNetEx import UNetEx
from ..MPS_Utilities import to_device

class PositionalEncoding2D(nn.Module):
    """
    2D positional encoding for transformer feature maps.
    Adds positional information to each position in a 2D feature map.
    """
    def __init__(self, d_model, max_h=64, max_w=64):
        super(PositionalEncoding2D, self).__init__()
        
        pe = torch.zeros(max_h, max_w, d_model)
        
        # Create position indices for height and width
        h_position = torch.arange(0, max_h, dtype=torch.float).unsqueeze(1).unsqueeze(2)
        w_position = torch.arange(0, max_w, dtype=torch.float).unsqueeze(0).unsqueeze(2)
        
        # Use sine and cosine functions of different frequencies
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sinusoidal positional encoding
        pe[:, :, 0::2] = torch.sin(h_position * div_term)
        pe[:, :, 1::2] = torch.cos(h_position * div_term)
        pe[:, :, 0::2] += torch.sin(w_position * div_term)
        pe[:, :, 1::2] += torch.cos(w_position * div_term)
        
        # Reshape to (H*W, 1, d_model) for adding to flattened feature maps
        pe = pe.view(-1, 1, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch, d_model) where seq_len = H*W
        Returns:
            Tensor of same shape with positional encoding added
        """
        return x + self.pe[:x.size(0), :]

class LayerNormConv2d(nn.Module):
    """
    Layer that applies Layer Normalization before convolution.
    Helps with training stability in transformer-based models.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super(LayerNormConv2d, self).__init__()
        # Layer norm first, normalizing over the channel dimension
        self.norm = nn.LayerNorm(in_channels)
        # Then convolution
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                            padding=padding, stride=stride)
        
    def forward(self, x):
        # [B, C, H, W] -> [B, H, W, C]
        x_transposed = x.permute(0, 2, 3, 1)
        x_normalized = self.norm(x_transposed)
        # Back to [B, C, H, W]
        x = x_normalized.permute(0, 3, 1, 2)
        # Then apply convolution
        return self.conv(x)
    
class TransformerUNetEx(UNetEx):
    """
    Extended UNet architecture with a Transformer module in the bottleneck.
    Combines the spatial hierarchical features of UNet with the global attention
    mechanism of Transformers for improved feature representation.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, filters=[16, 32, 64], layers=3,
                 weight_norm=True, batch_norm=True, activation=nn.ReLU, final_activation=None,
                 transformer_dim=128, nhead=4, num_layers=2): 
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolutional kernels
            filters: List defining the number of filters in each level
            layers: Number of conv layers per encoder/decoder block
            weight_norm: Whether to use weight normalization
            batch_norm: Whether to use batch normalization
            activation: Activation function to use
            final_activation: Final activation function (can be None)
            transformer_dim: Dimension of the transformer feature space
            nhead: Number of attention heads in transformer
            num_layers: Number of transformer encoder layers
        """
        super().__init__(in_channels, out_channels, kernel_size, filters, layers, 
                         weight_norm, batch_norm, activation, final_activation)
        
        # Ensure transformer_dim is divisible by nhead
        assert transformer_dim % nhead == 0, "Transformer dimension must be divisible by the number of heads"
        
        self.transformer_dim = transformer_dim
        
        # Projection to transformer dimension
        self.to_transformer_dim = LayerNormConv2d(filters[-1], transformer_dim, kernel_size=1, padding=0)
        
        # Positional encoding (will be initialized in forward pass with actual feature map size)
        self.pos_encoder = None
        
        # Transformer encoder configuration
        encoder_layer = TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=nhead,
            dim_feedforward=transformer_dim * 4,
            dropout=0.1,
            activation='relu',
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Projection back from transformer dimension
        self.from_transformer_dim = LayerNormConv2d(transformer_dim, filters[-1], kernel_size=1, padding=0)
        
        # Learnable parameter for residual connection
        self.residual_alpha = nn.Parameter(torch.tensor(0.2))
        
    def _initialize_pos_encoder(self, x, height, width):
        """Helper method to initialize or update positional encoding"""
        if self.pos_encoder is None or self.pos_encoder.pe.size(0) != height * width:
            self.pos_encoder = PositionalEncoding2D(self.transformer_dim, height, width)
            
            # Move to the same device as x
            if x.device.type == 'mps':
                self.pos_encoder = to_device(self.pos_encoder, x.device)
            else:
                self.pos_encoder = self.pos_encoder.to(x.device)
        
    def forward(self, x):
        """
        Forward pass through the TransformerUNetEx model.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """
        # Encode the input using the standard UNetEx encoder
        x, tensors, indices, sizes = self.encode(x)
        
        # Store the encoder output for residual connection
        transformer_input = x.clone()
        
        # Process the encoded feature through transformer
        batch_size, channels, height, width = x.shape
        
        # Project to transformer dimension
        x_proj = self.to_transformer_dim(x)
        
        # Initialize or update positional encoding
        self._initialize_pos_encoder(x, height, width)
        
        # Reshape to sequence format for transformer: (seq_len, batch, features)
        x_seq = x_proj.flatten(2).permute(2, 0, 1)
        
        # Add positional encoding
        x_seq = self.pos_encoder(x_seq)
        
        # Apply the transformer encoder
        x_transformed = self.transformer_encoder(x_seq)
        
        # Reshape back to feature map: (batch, features, height, width)
        x_transformed = x_transformed.permute(1, 2, 0).reshape(batch_size, self.transformer_dim, height, width)
        
        # Project back to original channel dimension
        x_from_transformer = self.from_transformer_dim(x_transformed)
        
        # Add residual connection from encoder
        x_bottleneck = x_from_transformer + self.residual_alpha * transformer_input
        
        # Decode using the standard UNetEx decoder
        x = self.decode(x_bottleneck, tensors, indices, sizes)
        
        # Apply final activation if specified
        if self.final_activation is not None:
            x = self.final_activation(x)
            
        return x