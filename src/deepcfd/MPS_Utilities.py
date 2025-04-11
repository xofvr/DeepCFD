"""
Utilities for working with Apple's Metal Performance Shaders (MPS) backend in PyTorch.
These utilities help ensure smooth transitions between CPU, CUDA, and MPS devices,
and provide workarounds for operations not fully supported on MPS.
"""
import torch
import torch.nn.functional as F
import warnings

def to_device(tensor, device):
    """
    Safely moves tensor to device, handling MPS limitations.
    
    Args:
        tensor: A tensor, list, or tuple of tensors to move
        device: The target device
        
    Returns:
        The tensor(s) moved to the target device
    """
    if isinstance(tensor, (list, tuple)):
        return [to_device(t, device) for t in tensor]
    
    if not isinstance(tensor, torch.Tensor):
        return tensor
        
    if tensor.device == device:
        return tensor
    
    # Handle device transitions gracefully
    if device is None:
        return tensor
        
    if tensor.device.type == 'mps' and device.type != 'mps':
        # MPS to other device (via CPU)
        return tensor.detach().cpu().to(device)
    elif tensor.device.type != 'mps' and device.type == 'mps':
        # Other device to MPS (via CPU)
        return tensor.detach().cpu().to(device)
    else:
        # Direct transfer for CPU<->CUDA or same device type
        return tensor.to(device)

def is_mps_available():
    """
    Check if MPS (Metal Performance Shaders) is available.
    
    Returns:
        bool: True if MPS is available, False otherwise
    """
    return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

def get_device_info(device):
    """
    Get information about the specified device.
    
    Args:
        device: The device to get information about
        
    Returns:
        dict: A dictionary with device information
    """
    info = {
        "type": device.type,
        "index": device.index,
    }
    
    if device.type == 'cuda':
        if torch.cuda.is_available():
            info["name"] = torch.cuda.get_device_name(device)
            info["memory_allocated"] = torch.cuda.memory_allocated(device) / 1024**2  # MB
            info["memory_reserved"] = torch.cuda.memory_reserved(device) / 1024**2    # MB
            info["max_memory"] = torch.cuda.get_device_properties(device).total_memory / 1024**2  # MB
    elif device.type == 'mps':
        info["name"] = "Apple Silicon (MPS)"
        # MPS doesn't have memory query functions yet
    
    return info

def custom_max_unpool2d(x, indices, kernel_size, stride=None, padding=0, output_size=None):
    """
    Custom implementation of max_unpool2d that works on MPS and other devices.
    
    This function provides a workaround for max_unpool2d which might not be fully
    supported on MPS or have performance issues.
    
    Args:
        x: Input tensor
        indices: Max indices returned by max_pool2d
        kernel_size: Size of the max pooling window
        stride: Stride of the max pooling window
        padding: Padding added to input before pooling
        output_size: Size of the output
        
    Returns:
        Unpooled tensor
    """
    if stride is None:
        stride = kernel_size
        
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
        
    batch_size, channels, height, width = x.shape
    if output_size is None:
        output_height = (height - 1) * stride[0] - 2 * padding[0] + kernel_size[0]
        output_width = (width - 1) * stride[1] - 2 * padding[1] + kernel_size[1]
        output_size = (output_height, output_width)
    else:
        output_height, output_width = output_size
        
    # Create output tensor filled with zeros
    output = torch.zeros((batch_size, channels, output_height, output_width), 
                        dtype=x.dtype, device=x.device)
    
    # Handle based on device type
    if x.device.type == 'mps':
        # For MPS, use a more memory-efficient approach by processing in chunks
        try:
            # Process in chunks to avoid memory issues on MPS
            chunk_size = min(batch_size, 16)  # Process 16 samples at a time
            for b in range(0, batch_size, chunk_size):
                end_idx = min(b + chunk_size, batch_size)
                chunk_indices = indices[b:end_idx]
                chunk_values = x[b:end_idx]
                
                # Convert indices to positions
                h_idx = ((chunk_indices // width) * stride[0]).long()
                w_idx = ((chunk_indices % width) * stride[1]).long()
                
                # Use scatter to place values in the output tensor
                for i in range(end_idx - b):
                    for c in range(channels):
                        output[b+i, c].view(-1)[
                            h_idx[i, c].view(-1) * output_width + w_idx[i, c].view(-1)
                        ] = chunk_values[i, c].view(-1)
                        
        except Exception as e:
            # Fallback to CPU implementation if MPS fails
            warnings.warn(f"MPS max_unpool2d implementation failed with error: {str(e)}. Falling back to CPU.")
            cpu_x = x.cpu()
            cpu_indices = indices.cpu()
            cpu_output = F.max_unpool2d(cpu_x, cpu_indices, kernel_size, stride, padding, output_size)
            return cpu_output.to(x.device)
    else:
        # For CUDA and CPU use the built-in implementation when possible
        try:
            return F.max_unpool2d(x, indices, kernel_size, stride, padding, output_size)
        except Exception as e:
            # Fallback implementation if built-in fails
            flat_indices = indices.view(-1)
            flat_values = x.view(-1)
            batch_steps = channels * height * width
            channel_steps = height * width
            
            for b in range(batch_size):
                for c in range(channels):
                    idx = flat_indices[b * batch_steps + c * channel_steps:
                                     b * batch_steps + (c + 1) * channel_steps]
                    val = flat_values[b * batch_steps + c * channel_steps:
                                    b * batch_steps + (c + 1) * channel_steps]
                    
                    h_idx = (idx // width) * stride[0]
                    w_idx = (idx % width) * stride[1]
                    
                    for i in range(len(idx)):
                        h, w = h_idx[i], w_idx[i]
                        if 0 <= h < output_height and 0 <= w < output_width:
                            output[b, c, h, w] = val[i]
                
    return output

def safe_max_pool2d_with_indices(x, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
    """
    Safe version of max_pool2d that works on both MPS and other devices.
    
    Args:
        x: Input tensor
        kernel_size: Size of max pooling window
        stride: Stride of max pooling window
        padding: Padding added to input
        dilation: Dilation applied to pooling window
        ceil_mode: Use ceil instead of floor for output size
        
    Returns:
        Tuple of (output, indices) tensors
    """
    if x.device.type == 'mps':
        try:
            # Try native MPS implementation first
            return F.max_pool2d(x, kernel_size, stride, padding, dilation, ceil_mode, return_indices=True)
        except Exception:
            # Fall back to chunked CPU approach if native implementation fails
            chunk_size = min(x.shape[0], 16)
            output_chunks = []
            indices_chunks = []
            
            for i in range(0, x.shape[0], chunk_size):
                chunk = x[i:i + chunk_size]
                # Move to CPU, perform operation, and move back
                chunk_cpu = chunk.cpu()
                output_chunk, indices_chunk = F.max_pool2d(
                    chunk_cpu, kernel_size, stride, padding, 
                    dilation, ceil_mode, return_indices=True
                )
                output_chunks.append(output_chunk.to('mps'))
                indices_chunks.append(indices_chunk.to('mps'))
            
            return torch.cat(output_chunks, dim=0), torch.cat(indices_chunks, dim=0)
    else:
        return F.max_pool2d(x, kernel_size, stride, padding, dilation, ceil_mode, return_indices=True)

def mps_adaptive_avg_pool2d(x, output_size):
    """
    Safe adaptive average pooling for MPS.
    
    Args:
        x: Input tensor
        output_size: Desired output size
        
    Returns:
        Pooled tensor
    """
    if x.device.type == 'mps':
        try:
            # Try native MPS implementation
            return F.adaptive_avg_pool2d(x, output_size)
        except Exception:
            # Fall back to CPU if native implementation fails
            x_cpu = x.cpu()
            out_cpu = F.adaptive_avg_pool2d(x_cpu, output_size)
            return out_cpu.to('mps')
    else:
        return F.adaptive_avg_pool2d(x, output_size)

def safe_layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    """
    Safe layer normalization that works on MPS.
    
    Args:
        x: Input tensor
        normalized_shape: Shape of the normalization
        weight: Optional weight parameter
        bias: Optional bias parameter
        eps: Small constant for numerical stability
        
    Returns:
        Normalized tensor
    """
    if x.device.type == 'mps':
        try:
            # Try native MPS implementation
            return F.layer_norm(x, normalized_shape, weight, bias, eps)
        except Exception:
            # Fall back to manual implementation
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, unbiased=False, keepdim=True)
            normalized = (x - mean) / torch.sqrt(var + eps)
            
            if weight is not None and bias is not None:
                return normalized * weight + bias
            elif weight is not None:
                return normalized * weight
            elif bias is not None:
                return normalized + bias
            else:
                return normalized
    else:
        return F.layer_norm(x, normalized_shape, weight, bias, eps)