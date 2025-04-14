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

def load_data(x_path, y_path):
    """Load input and output data from pickle files"""
    with open(x_path, 'rb') as f:
        x_data = pickle.load(f)
    with open(y_path, 'rb') as f:
        y_data = pickle.load(f)
    return x_data, y_data

class TrainVisualizer:
    """Helper class to visualize training results"""
    def __init__(self, visualize=False, device=None):
        self.visualize = visualize
        self.device = device
    
    def on_val_epoch(self, scope):
        if not self.visualize:
            return
        
        # Get a sample from validation data for visualization
        model = scope["model"]
        val_dataset = scope["dataset"]
        
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
        
        # Visualize
        visualize(y_np, pred_np, error, 0)

def loss_function(model, tensors):
    """Loss function for training the model"""
    x, y = tensors
    y_pred = model(x)
    loss = torch.nn.functional.mse_loss(y_pred, y)
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
    
    # Load data
    x_data, y_data = load_data(args.model_input, args.model_output)
    
    # Convert numpy arrays to torch tensors
    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32)
    
    # Create dataset and split into train/val sets
    dataset = TensorDataset(x_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
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
    
    # Create model
    model = TransformerUNetEx(
        in_channels=x_data.shape[1], 
        out_channels=y_data.shape[1],
        kernel_size=args.kernel_size,
        filters=filters,
        transformer_dim=config['transformer_dim'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        use_checkpointing=config['use_checkpointing'],
        final_activation=torch.nn.Tanh()
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Set up training visualizer
    visualizer = TrainVisualizer(visualize=args.visualize, device=device)
    
    # Setup MSE metric
    def mse_on_batch(scope):
        return scope["loss"].item()
    
    def mse_on_epoch(scope):
        return sum(scope["list"]) / len(scope["list"])
    
    # Train the model
    print(f"Starting training for {args.epochs} epochs with batch size {args.batch_size}")
    best_model, train_metrics, train_loss, val_metrics, val_loss = train_model(
        model=model,
        loss_func=loss_function,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        optimizer=optimizer,
        on_val_epoch=visualizer.on_val_epoch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        device=device,
        m_mse_name="MSE",
        m_mse_on_batch=mse_on_batch,
        m_mse_on_epoch=mse_on_epoch
    )
    
    # Save the best model
    torch.save(best_model.state_dict(), args.output)
    print(f"Training complete. Model saved to {args.output}")
    
    # Save transformer config as well for future reference
    config_path = os.path.splitext(args.output)[0] + "_config.json"
    import json
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Model configuration saved to {config_path}")

if __name__ == "__main__":
    main()