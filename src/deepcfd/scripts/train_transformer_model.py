#!/usr/bin/env python3
"""
Helper script for training the TransformerUNetEx model with optimal configuration.
This script provides convenience functions to set up and train the model
with optimal parameters for different hardware configurations.
"""

import os
import sys
import argparse
import torch
import pickle
import numpy as np
from torch.utils.data import TensorDataset
from ..models.TransformerUNetEx import TransformerUNetEx, get_optimal_transformer_config
from ..train_functions import train_model
from ..functions import visualize
from ..MPS_Utilities import to_device, is_mps_available

def get_device_from_args(args):
    """Get the appropriate device based on command line arguments"""
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'mps' and is_mps_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device

def load_data(x_path, y_path, transform_data=True):
    """Load input and output data from pickle files with optional transformations
    
    Args:
        x_path: Path to input data pickle file
        y_path: Path to output data pickle file
        transform_data: Whether to apply transformations (default: True)
    
    Returns:
        x_data: Input data
        y_data: Output data
        transform_params: Transformation parameters (for inverse transform during inference)
    """
    with open(x_path, 'rb') as f:
        x_data = pickle.load(f)
    with open(y_path, 'rb') as f:
        y_data = pickle.load(f)
        
    # Print data statistics to help diagnose any issues
    print(f"Input data shape: {x_data.shape}, Output data shape: {y_data.shape}")
    print(f"Input data range: [{np.min(x_data):.4f}, {np.max(x_data):.4f}], mean: {np.mean(x_data):.4f}, std: {np.std(x_data):.4f}")
    print(f"Output data range: [{np.min(y_data):.4f}, {np.max(y_data):.4f}], mean: {np.mean(y_data):.4f}, std: {np.std(y_data):.4f}")
    
    # Default transformation parameters
    transform_params = {
        'x_params': [],
        'y_params': []
    }
    
    # Apply transformations only if explicitly requested
    if transform_data:
        # Create copies to avoid modifying the original data
        x_transformed = x_data.copy()
        y_transformed = y_data.copy()
        
        # Transform input data - channel-wise approach
        x_channel_params = []
        for c in range(x_data.shape[1]):
            channel_data = x_data[:, c, :, :]
            channel_min = np.min(channel_data)
            channel_max = np.max(channel_data)
            channel_mean = np.mean(channel_data)
            channel_std = np.std(channel_data)
            
            # Different transformation strategy for different types of channels
            # For signed distance function channels (channel 0, 2)
            if c == 0 or c == 2:
                # Apply symlog transformation (signed log) to compress extreme values
                # while preserving sign: y = sign(x) * log(1 + |x|)
                sign = np.sign(channel_data)
                log_data = np.log1p(np.abs(channel_data))
                x_transformed[:, c, :, :] = sign * log_data
                x_channel_params.append({
                    'transform': 'symlog',
                    'min': channel_min,
                    'max': channel_max,
                    'mean': channel_mean,
                    'std': channel_std
                })
            # For flow region channel (channel 1) - leave as is
            else:
                x_channel_params.append({
                    'transform': 'identity',
                    'min': channel_min,
                    'max': channel_max,
                    'mean': channel_mean,
                    'std': channel_std
                })
        
        # Transform output data - physics-aware scaling
        y_channel_params = []
        for c in range(y_data.shape[1]):
            channel_data = y_data[:, c, :, :]
            channel_min = np.min(channel_data)
            channel_max = np.max(channel_data)
            channel_mean = np.mean(channel_data)
            channel_std = np.std(channel_data)
            
            # For velocity channels (0, 1) - keep as is, models can handle this range well
            if c == 0 or c == 1:
                y_channel_params.append({
                    'transform': 'identity',
                    'min': channel_min,
                    'max': channel_max,
                    'mean': channel_mean,
                    'std': channel_std
                })
            # For pressure channel (2) - ensure better numerical stability
            else:
                # Shift mean to zero to improve numerical stability
                mean_adjusted = channel_data - channel_mean
                # Divide by a modest factor to maintain the relationships
                scaled_data = mean_adjusted / (3 * channel_std)
                y_transformed[:, c, :, :] = scaled_data
                
                y_channel_params.append({
                    'transform': 'mean_std_scale',
                    'min': channel_min,
                    'max': channel_max,
                    'mean': channel_mean,
                    'std': channel_std,
                    'scale_factor': 3.0
                })
        
        # Update transformation parameters
        transform_params = {
            'x_params': x_channel_params,
            'y_params': y_channel_params
        }
        
        # Print transformed data statistics
        print(f"Transformed input data range: [{np.min(x_transformed):.4f}, {np.max(x_transformed):.4f}], mean: {np.mean(x_transformed):.4f}, std: {np.std(x_transformed):.4f}")
        print(f"Transformed output data range: [{np.min(y_transformed):.4f}, {np.max(y_transformed):.4f}], mean: {np.mean(y_transformed):.4f}, std: {np.std(y_transformed):.4f}")
        
        return x_transformed, y_transformed, transform_params
    
    # Return original data without transformations
    return x_data, y_data, transform_params

class TrainVisualizer:
    """Helper class to visualize training results"""
    def __init__(self, visualize=False, device=None):
        self.visualize = visualize
        self.device = device
        self.epoch_count = 0
    
    def on_val_epoch(self, scope):
        if not self.visualize:
            return
        
        # Get a sample from validation data for visualization
        model = scope["model"]
        val_dataset = scope["dataset"]
        
        # Increment epoch counter
        self.epoch_count += 1
        
        # Get a random sample
        sample_idx = np.random.randint(0, len(val_dataset))
        x, y = val_dataset[sample_idx]
        
        # Add batch dimension
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        
        # Move to device
        if self.device is not None:
            x = x.to(self.device)
            y = y.to(self.device)
        
        # Get prediction
        with torch.no_grad():
            pred = model(x)
        
        # Move to CPU for visualization
        y_np = y.cpu().numpy()
        pred_np = pred.cpu().numpy()
        
        # Calculate error
        error = np.abs(y_np - pred_np)
        
        # Save visualization to file instead of showing interactively
        saved_path = visualize(y_np, pred_np, error, 0, save_to_file=True)
        print(f"Saved visualization for epoch {self.epoch_count} to {saved_path}")

def loss_function(model, tensors):
    """Physics-informed loss function for training CFD models
    
    This loss function combines:
    1. Standard MSE component-wise losses
    2. Gradient penalties to enforce physical constraints
    3. Adaptive weighting for improved convergence
    """
    x, y = tensors
    
    # Forward pass with the model
    y_pred = model(x)
    
    # Check for NaN values in predictions and replace with zeros
    if torch.isnan(y_pred).any():
        print("Warning: NaN values detected in model output. Stabilizing...")
        y_pred = torch.where(torch.isnan(y_pred), torch.zeros_like(y_pred), y_pred)
    
    # Extract individual components
    # y dimensions: [batch_size, channels, height, width]
    # Channel 0: Ux (x-velocity)
    # Channel 1: Uy (y-velocity)
    # Channel 2: P (pressure)
    ux_true = y[:, 0:1]  # Keep dim for later operations
    uy_true = y[:, 1:2]
    p_true = y[:, 2:3]
    
    ux_pred = y_pred[:, 0:1]
    uy_pred = y_pred[:, 1:2]
    p_pred = y_pred[:, 2:3]
    
    # 1. MSE Loss components
    mse_ux = torch.mean((ux_pred - ux_true) ** 2)
    mse_uy = torch.mean((uy_pred - uy_true) ** 2)
    mse_p = torch.mean((p_pred - p_true) ** 2)
    
    # Basic MSE loss
    base_loss = mse_ux + mse_uy + mse_p
    
    # 2. Gradient-based physics-informed constraints
    # We penalize large gradients in adjacent cells to ensure smoothness
    # This is important for fluid dynamics where sharp transitions are physically unrealistic
    def gradient_penalty(field):
        # Compute gradients in x and y directions using finite differences
        # We use slicing for simplicity but more sophisticated methods could be used
        dx = field[:, :, 1:, :] - field[:, :, :-1, :]  # x-direction gradient
        dy = field[:, :, :, 1:] - field[:, :, :, :-1]  # y-direction gradient
        
        # Penalize extreme gradients that may cause instability
        penalty_x = torch.mean(dx**2)
        penalty_y = torch.mean(dy**2)
        
        return penalty_x + penalty_y
    
    # Apply gradient penalties with appropriate weighting
    grad_penalty_ux = gradient_penalty(ux_pred)
    grad_penalty_uy = gradient_penalty(uy_pred)
    grad_penalty_p = gradient_penalty(p_pred)
    
    # Set gradient penalty weight as a small fraction of the MSE
    grad_weight = 0.01
    grad_loss = grad_weight * (grad_penalty_ux + grad_penalty_uy + grad_penalty_p)
    
    # 3. Weighted component loss with physical meaning
    # Velocity components (Ux, Uy) are typically more important than pressure
    velocity_weight = 1.0
    pressure_weight = 0.5  # Reduced weight for pressure
    
    # Final weighted loss
    weighted_loss = (velocity_weight * (mse_ux + mse_uy) + pressure_weight * mse_p) / (2 * velocity_weight + pressure_weight)
    
    # Combine all loss components
    loss = weighted_loss + grad_loss
    
    # Return the combined loss and prediction
    return loss, y_pred

def get_training_args():
    """Parse command line arguments for model training"""
    parser = argparse.ArgumentParser(description='Train TransformerUNetEx model with optimal configuration')
    
    # Data arguments
    parser.add_argument('--model-input', type=str, required=True, help='Path to input data file')
    parser.add_argument('--model-output', type=str, required=True, help='Path to output data file')
    parser.add_argument('--output', type=str, required=True, help='Path to save the trained model')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'],
                        help='Device to train on (cpu, cuda, mps)')
    
    # Model architecture arguments
    parser.add_argument('--kernel-size', type=int, default=3, help='Kernel size for convolutional layers')
    parser.add_argument('--filters', type=str, default='16,32,64,64', 
                        help='Number of filters in each level (comma-separated)')
    parser.add_argument('--transformer-dim', type=int, default=None, 
                        help='Transformer embedding dimension (default: auto-configured)')
    parser.add_argument('--nhead', type=int, default=None,
                        help='Number of transformer attention heads (default: auto-configured)')
    parser.add_argument('--num-layers', type=int, default=None,
                        help='Number of transformer encoder layers (default: auto-configured)')
    
    # Training arguments
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--patience', type=int, default=300, help='Patience for early stopping')
    parser.add_argument('--no-checkpointing', action='store_true', 
                       help='Disable gradient checkpointing (uses more memory)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # Visualization arguments
    parser.add_argument('--visualize', action='store_true', help='Visualize training results')
    
    return parser.parse_args()

def main():
    """Main function to set up and train the model"""
    # Parse arguments
    args = get_training_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Get device
    device = get_device_from_args(args)
    print(f"Using device: {device}")
    
    # Load data with physics-informed transformations
    x_data, y_data, transform_params = load_data(args.model_input, args.model_output, transform_data=True)
    
    # Convert numpy arrays to torch tensors
    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32)
    
    # Create dataset and split into train/val sets
    dataset = TensorDataset(x_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    # Use a fixed random seed for reproducible train/val splits
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    # Parse filters
    filters = list(map(int, args.filters.split(',')))
    print(f"Using filter configuration: {filters}")
    
    # Get optimal transformer configuration based on device
    config = get_optimal_transformer_config(device.type)
    
    # Override with user-specified parameters
    if args.transformer_dim is not None:
        config['transformer_dim'] = args.transformer_dim
    if args.nhead is not None:
        config['nhead'] = args.nhead
    if args.num_layers is not None:
        config['num_layers'] = args.num_layers
    if args.no_checkpointing:
        config['use_checkpointing'] = False
        
    print(f"Transformer configuration: {config}")
    
    # Create model with Kaiming initialization for better convergence
    model = TransformerUNetEx(
        in_channels=x_data.shape[1], 
        out_channels=y_data.shape[1],
        kernel_size=args.kernel_size,
        filters=filters,
        transformer_dim=config['transformer_dim'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        use_checkpointing=config['use_checkpointing'],
        final_activation=None  # No final activation since we're using our own transformations
    )
    
    # Apply proper weight initialization for different layer types
    def init_weights(m):
        if isinstance(m, torch.nn.Conv2d):
            # Kaiming initialization for convolutional layers
            # Better for ReLU activations and helps with vanishing/exploding gradients
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                # Initialize biases to a small constant for better stability
                torch.nn.init.constant_(m.bias, 0.01)
        elif isinstance(m, torch.nn.BatchNorm2d):
            # Standard initialization for batch norm
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            # Xavier initialization for linear layers
            # Good for layers with various activation functions
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
    
    # Apply initialization to all layers
    model.apply(init_weights)
    
    # Create optimizer with improved hyperparameters
    optimizer = torch.optim.AdamW(  # AdamW instead of Adam for better weight decay handling
        model.parameters(), 
        lr=args.learning_rate * 0.1,  # Reduced learning rate for stability
        betas=(0.9, 0.999),
        eps=1e-5,  # Higher epsilon for numerical stability
        weight_decay=1e-4  # L2 regularization
    )
    
    # Use OneCycleLR scheduler for better convergence
    # This implements the 1cycle policy from the paper "Super-Convergence: Very Fast Training of Neural Networks"
    # It helps reach better minima faster and reduces overfitting
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        steps_per_epoch=train_size // args.batch_size + (1 if train_size % args.batch_size != 0 else 0),
        epochs=args.epochs,
        pct_start=0.3,  # Spend 30% of training in warmup phase
        div_factor=25.0,  # initial_lr = max_lr/div_factor
        final_div_factor=10000.0,  # final_lr = initial_lr/final_div_factor
    )
    
    # Save transformation parameters for later use in inference
    transform_config_path = os.path.splitext(args.output)[0] + "_transform_config.json"
    # Make sure the output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(transform_config_path, 'w') as f:
        import json
        # Convert NumPy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            # Updated for NumPy 2.0 compatibility
            if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            return obj
        
        # Convert all NumPy values in the transform_params dict
        clean_params = {}
        for key, value in transform_params.items():
            if isinstance(value, list):
                clean_params[key] = [
                    {k: convert_numpy_types(v) for k, v in item.items()}
                    for item in value
                ]
            else:
                clean_params[key] = convert_numpy_types(value)
        
        json.dump(clean_params, f, indent=4)
    print(f"Transformation parameters saved to {transform_config_path}")
    
    # Set up training visualizer
    visualizer = TrainVisualizer(visualize=args.visualize, device=device)
    
    # Define additional physics-aware metrics
    def mse_on_batch(scope):
        return scope["loss"].item()
    
    def mse_on_epoch(scope):
        return sum(scope["list"]) / len(scope["list"])
    
    # Improved R2 score calculation with safeguards for numerical stability
    def r2_on_batch(scope):
        # Extract prediction and target
        y_pred = scope["output"].detach()
        y_true = scope["batch"][1].detach()
        
        # Calculate total sum of squares with epsilon for numerical stability
        eps = 1e-8
        y_mean = torch.mean(y_true, dim=[0, 2, 3], keepdim=True)
        ss_tot = torch.sum((y_true - y_mean) ** 2) + eps
        
        # Calculate residual sum of squares
        ss_res = torch.sum((y_true - y_pred) ** 2) + eps
        
        # R2 = 1 - (residual sum of squares / total sum of squares)
        r2 = 1 - (ss_res / ss_tot)
        
        # Clamp R2 to reasonable range to avoid extreme values
        r2 = torch.clamp(r2, min=-1.0, max=1.0)
        
        return r2.item()
    
    def r2_on_epoch(scope):
        # Filter out potential NaN values
        valid_values = [x for x in scope["list"] if not (np.isnan(x) or np.isinf(x))]
        if not valid_values:
            return -1.0
        return sum(valid_values) / len(valid_values)
    
    # Component-wise metrics (Ux, Uy, P)
    def component_mse_on_batch(scope, component_idx):
        y_pred = scope["output"].detach()
        y_true = scope["batch"][1].detach()
        
        mse = torch.mean((y_pred[:, component_idx] - y_true[:, component_idx]) ** 2)
        return mse.item()
    
    def ux_mse_on_batch(scope):
        return component_mse_on_batch(scope, 0)
    
    def uy_mse_on_batch(scope):
        return component_mse_on_batch(scope, 1)
    
    def p_mse_on_batch(scope):
        return component_mse_on_batch(scope, 2)
    
    def component_mse_on_epoch(scope):
        return sum(scope["list"]) / len(scope["list"])
    
    # RMSE metric
    def rmse_on_batch(scope):
        return np.sqrt(scope["loss"].item())
    
    def rmse_on_epoch(scope):
        return np.sqrt(sum(scope["list"]) / len(scope["list"]))
    
    # Add divergence metric to measure physical consistency
    def divergence_on_batch(scope):
        # Extract velocity predictions
        y_pred = scope["output"].detach()
        ux_pred = y_pred[:, 0]
        uy_pred = y_pred[:, 1]
        
        # Compute velocity divergence using finite differences
        # For incompressible flow, this should be close to zero
        ux_dx = ux_pred[:, 1:, :] - ux_pred[:, :-1, :]
        uy_dy = uy_pred[:, :, 1:] - uy_pred[:, :, :-1]
        
        # Pad to maintain dimensions
        ux_dx_padded = torch.nn.functional.pad(ux_dx, (0, 0, 0, 1), "constant", 0)
        uy_dy_padded = torch.nn.functional.pad(uy_dy, (0, 1, 0, 0), "constant", 0)
        
        # Calculate divergence magnitude
        div = torch.abs(ux_dx_padded + uy_dy_padded)
        
        # Return mean divergence
        return torch.mean(div).item()
    
    def divergence_on_epoch(scope):
        return sum(scope["list"]) / len(scope["list"])
    
    # Print training configuration
    print(f"Starting training for {args.epochs} epochs with batch size {args.batch_size}")
    print(f"Using physics-informed transformations and gradient penalties")
    print(f"Early stopping patience: {args.patience}")
    
    # Train the model with our physics-informed approach
    best_model, train_metrics, train_loss, val_metrics, val_loss = train_model(
        model=model,
        loss_func=loss_function,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        on_val_epoch=visualizer.on_val_epoch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        device=device,
        max_grad_norm=1.0,  # Gradient clipping to avoid exploding gradients
        # Register all our physics-informed metrics
        m_mse_name="MSE",
        m_mse_on_batch=mse_on_batch,
        m_mse_on_epoch=mse_on_epoch,
        m_r2_name="R²",
        m_r2_on_batch=r2_on_batch,
        m_r2_on_epoch=r2_on_epoch,
        m_rmse_name="RMSE",
        m_rmse_on_batch=rmse_on_batch,
        m_rmse_on_epoch=rmse_on_epoch,
        m_ux_name="Ux MSE",
        m_ux_on_batch=ux_mse_on_batch,
        m_ux_on_epoch=component_mse_on_epoch,
        m_uy_name="Uy MSE",
        m_uy_on_batch=uy_mse_on_batch,
        m_uy_on_epoch=component_mse_on_epoch,
        m_p_name="P MSE",
        m_p_on_batch=p_mse_on_batch,
        m_p_on_epoch=component_mse_on_epoch,
        m_div_name="Divergence",
        m_div_on_batch=divergence_on_batch,
        m_div_on_epoch=divergence_on_epoch
    )
    
    # Save the best model
    torch.save(best_model.state_dict(), args.output)
    print(f"Training complete. Best model saved to {args.output}")
    
    # Save model configuration for future use
    config_path = os.path.splitext(args.output)[0] + "_config.json"
    import json
    with open(config_path, 'w') as f:
        # Add filter configuration and other parameters to config
        full_config = {
            **config,
            'filters': filters,
            'kernel_size': args.kernel_size,
            'in_channels': x_data.shape[1],
            'out_channels': y_data.shape[1],
        }
        json.dump(full_config, f, indent=4)
    print(f"Model configuration saved to {config_path}")
    
    # Calculate and report final metrics on best model
    model.load_state_dict(best_model.state_dict())
    model.eval()
    
    # Create a summary of best validation performance
    print("\nBest Model Performance Summary:")
    print(f"Best validation loss: {min(val_loss):.6f}")
    
    # Get metrics at best epoch
    best_epoch = val_loss.index(min(val_loss))
    for metric_name, metric_values in val_metrics.items():
        if best_epoch < len(metric_values):
            print(f"Best {metric_name}: {metric_values[best_epoch]:.6f}")
    
    print("\nTraining complete. Use model for inference with appropriate inverse transformations.")

if __name__ == "__main__":
    main()