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
        # Make sure we don't index outside the buffer size
        seq_len = x.size(0)
        if seq_len > self.pe.size(0):
            # If the input sequence is longer than our max buffer, we need to regenerate
            # This might happen with very large feature maps
            raise ValueError(f"Input sequence length {seq_len} exceeds maximum positional encoding size {self.pe.size(0)}")
        
        # Ensure positional encoding is on the same device as the input tensor
        if x.device != self.pe.device:
            pe = self.pe.to(x.device)
        else:
            pe = self.pe
            
        return x + pe[:seq_len]

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
        
        # Use safe layer normalization (handles MPS edge cases)
        from ..MPS_Utilities import safe_layer_norm
        if x.device.type == 'mps':
            x_normalized = safe_layer_norm(x_transposed, [x_transposed.size(-1)], 
                                          self.norm.weight, self.norm.bias, self.norm.eps)
        else:
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
                 transformer_dim=128, nhead=4, num_layers=2, use_checkpointing=True): 
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
            use_checkpointing: Whether to use gradient checkpointing to save memory
        """
        super().__init__(in_channels, out_channels, kernel_size, filters, layers, 
                         weight_norm, batch_norm, activation, final_activation)
        
        # Ensure transformer_dim is divisible by nhead
        assert transformer_dim % nhead == 0, "Transformer dimension must be divisible by the number of heads"
        
        self.transformer_dim = transformer_dim
        self.use_checkpointing = use_checkpointing
        
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
            # Calculate max dimensions dynamically based on current input size
            # Use slightly larger size to accommodate minor size variations
            max_h = max(64, height * 2)
            max_w = max(64, width * 2)
            
            self.pos_encoder = PositionalEncoding2D(self.transformer_dim, max_h, max_w)
            
            # Move to the same device as x
            if x.device.type == 'mps':
                self.pos_encoder = to_device(self.pos_encoder, x.device)
            else:
                self.pos_encoder = self.pos_encoder.to(x.device)
        
    def _transformer_forward_with_fallback(self, x_seq):
        """
        Apply transformer encoder with fallback mechanisms
        
        Args:
            x_seq: Input sequence tensor of shape (seq_len, batch, d_model)
            
        Returns:
            Transformed sequence
        """
        device = x_seq.device
        
        try:
            # Try standard forward pass
            if self.use_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                return checkpoint(lambda s: self.transformer_encoder(s), x_seq)
            else:
                return self.transformer_encoder(x_seq)
        except Exception as e:
            # If on MPS, try CPU fallback
            if device.type == 'mps':
                print(f"MPS transformer operation failed with error: {e}. Falling back to CPU.")
                # Move to CPU, compute, then back to MPS
                cpu_result = self.transformer_encoder(x_seq.cpu())
                return cpu_result.to(device)
            else:
                # For other devices, re-raise the exception
                raise

    def _process_with_transformer(self, x_proj, batch_size, height, width):
        """
        Process feature maps through transformer with memory-efficient chunking
        
        Args:
            x_proj: Projected feature map of shape (batch_size, transformer_dim, height, width)
            batch_size: Batch size
            height: Feature map height
            width: Feature map width
            
        Returns:
            Transformer processed feature map
        """
        # Initialize or update positional encoding
        self._initialize_pos_encoder(x_proj, height, width)
        
        # Determine if chunking is needed based on feature map size
        use_chunking = (height * width) > 1024  # Threshold where chunking becomes beneficial
        max_chunk_size = 512  # Maximum pixels to process in one chunk
        
        if not use_chunking:
            # Standard processing without chunking
            # Reshape to sequence format for transformer: (seq_len, batch, features)
            x_seq = x_proj.flatten(2).permute(2, 0, 1)
            
            # Add positional encoding
            x_seq = self.pos_encoder(x_seq)
            
            # Apply the transformer encoder with fallback
            x_transformed = self._transformer_forward_with_fallback(x_seq)
            
            # Reshape back to feature map: (batch, features, height, width)
            return x_transformed.permute(1, 2, 0).reshape(batch_size, self.transformer_dim, height, width)
        else:
            # Process in chunks to save memory
            num_chunks = math.ceil((height * width) / max_chunk_size)
            chunks = torch.chunk(x_proj.flatten(2), num_chunks, dim=2)
            transformed_chunks = []
            
            # Ensure positional encoding is on the same device as input
            pe_device = x_proj.device
            pe_buffer = self.pos_encoder.pe.to(pe_device)
            
            for chunk_idx, chunk in enumerate(chunks):
                # Get current chunk size
                curr_chunk_size = chunk.size(2)
                
                # Reshape to sequence format: (seq_len, batch, features)
                chunk_seq = chunk.permute(2, 0, 1)
                
                # Add positional encoding - calculate offset for positional encoding
                pos_offset = chunk_idx * max_chunk_size
                pos_end = pos_offset + curr_chunk_size
                chunk_seq = chunk_seq + pe_buffer[pos_offset:pos_end]
                
                # Apply the transformer encoder with fallback
                chunk_transformed = self._transformer_forward_with_fallback(chunk_seq)
                
                # Save transformed chunk
                transformed_chunks.append(chunk_transformed)
            
            # Concatenate all chunks
            x_transformed = torch.cat(transformed_chunks, dim=0)
            
            # Reshape back to feature map: (batch, features, height, width)
            return x_transformed.permute(1, 2, 0).reshape(batch_size, self.transformer_dim, height, width)
            
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
        
        # Process with transformer (with memory-efficient handling)
        x_transformed = self._process_with_transformer(x_proj, batch_size, height, width)
        
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

def get_optimal_transformer_config(device_type='cuda', device_memory_gb=None):
    """
    Returns optimal configuration parameters for TransformerUNetEx based on device type and available memory.
    
    Args:
        device_type: The device type ('cuda', 'mps', or 'cpu')
        device_memory_gb: Approximate memory available in GB (if None, will estimate based on device_type)
        
    Returns:
        dict: Configuration parameters for TransformerUNetEx
    """
    # Default configuration (conservative)
    config = {
        'transformer_dim': 128,
        'nhead': 4,
        'num_layers': 2,
        'use_checkpointing': True,
    }
    
    # Estimate memory if not provided
    if device_memory_gb is None:
        if device_type == 'cuda':
            if torch.cuda.is_available():
                # Get device properties
                device_props = torch.cuda.get_device_properties(0)
                device_memory_gb = device_props.total_memory / (1024**3)
            else:
                device_memory_gb = 4  # Conservative default
        elif device_type == 'mps':
            device_memory_gb = 8  # Conservative estimate for Apple Silicon
        else:  # CPU
            device_memory_gb = 8  # Conservative default
    
    # Configure based on available memory and device type
    if device_type == 'cuda':
        if device_memory_gb >= 16:
            # High-end GPU
            config.update({
                'transformer_dim': 256,
                'nhead': 8,
                'num_layers': 4,
                'use_checkpointing': False  # Enough memory to avoid checkpointing
            })
        elif device_memory_gb >= 8:
            # Mid-range GPU
            config.update({
                'transformer_dim': 192,
                'nhead': 6,
                'num_layers': 3,
                'use_checkpointing': True
            })
    elif device_type == 'mps':
        # Apple Silicon (M1/M2/M3)
        if device_memory_gb >= 16:
            # High-end Apple Silicon 
            config.update({
                'transformer_dim': 192,
                'nhead': 6,
                'num_layers': 3,
                'use_checkpointing': True
            })
        else:
            # Standard Apple Silicon
            config.update({
                'transformer_dim': 128,
                'nhead': 4,
                'num_layers': 2,
                'use_checkpointing': True
            })
    else:  # CPU
        # More conservative for CPU
        config.update({
            'transformer_dim': 96,
            'nhead': 3,
            'num_layers': 1,
            'use_checkpointing': True
        })
        
    return config