import torch
import torch.nn.functional as F

def to_device(tensor, device):
    """Safely moves tensor to device, handling MPS limitations"""
    if isinstance(tensor, (list, tuple)):
        return [to_device(t, device) for t in tensor]
    
    if not isinstance(tensor, torch.Tensor):
        return tensor
        
    if tensor.device == device:
        return tensor
    
    return tensor.to(device)

def custom_max_unpool2d(x, indices, kernel_size, stride=None, padding=0, output_size=None):
    """
    Custom implementation of max_unpool2d that works on MPS
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
    
    # Flatten indices for easier processing
    flat_indices = indices.view(-1)
    flat_values = x.view(-1)
    
    # Calculate base indices for each element in the batch
    batch_steps = channels * height * width
    channel_steps = height * width
    base_indices = torch.arange(0, batch_size * channels * height * width, 
                              channel_steps, device=x.device).view(-1, 1)
    
    # Convert pooled indices back to unpooled positions
    h_stride, w_stride = stride
    for b in range(batch_size):
        for c in range(channels):
            idx = flat_indices[b * batch_steps + c * channel_steps:
                             b * batch_steps + (c + 1) * channel_steps]
            val = flat_values[b * batch_steps + c * channel_steps:
                            b * batch_steps + (c + 1) * channel_steps]
            
            # Convert flattened indices to 2D positions
            h_idx = (idx // width) * h_stride
            w_idx = (idx % width) * w_stride
            
            # Place values in output tensor
            output[b, c, h_idx, w_idx] = val
            
    return output

def safe_max_pool2d_with_indices(x, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
    """
    Safe version of max_pool2d that works on both MPS and other devices
    """
    if x.device.type == 'mps':
        # Move to CPU, perform operation, and move back
        x_cpu = x.cpu()
        output, indices = F.max_pool2d(x_cpu, kernel_size, stride, padding, dilation, ceil_mode, return_indices=True)
        return output.to('mps'), indices.to('mps')
    else:
        return F.max_pool2d(x, kernel_size, stride, padding, dilation, ceil_mode, return_indices=True)