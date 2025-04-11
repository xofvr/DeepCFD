import matplotlib.pyplot as plt
import argparse
import os

def parse_log_file(log_path):
    """Parse a training log file and extract metrics."""
    train_loss = []
    train_total_mse = []
    train_ux_mse = []
    train_uy_mse = []
    train_p_mse = []
    val_loss = []
    val_total_mse = []
    val_ux_mse = []
    val_uy_mse = []
    val_p_mse = []

    with open(log_path, 'r') as f:
        for line in f:
            if line.strip().startswith('Train Loss'):
                train_loss.append(float(line.split()[-1]))
            elif line.strip().startswith('Train Total MSE'):
                train_total_mse.append(float(line.split()[-1]))
            elif line.strip().startswith('Train Ux MSE'):
                train_ux_mse.append(float(line.split()[-1]))
            elif line.strip().startswith('Train Uy MSE'):
                train_uy_mse.append(float(line.split()[-1]))
            elif line.strip().startswith('Train p MSE'):
                train_p_mse.append(float(line.split()[-1]))
            elif line.strip().startswith('Validation Loss'):
                val_loss.append(float(line.split()[-1]))
            elif line.strip().startswith('Validation Total MSE'):
                val_total_mse.append(float(line.split()[-1]))
            elif line.strip().startswith('Validation Ux MSE'):
                val_ux_mse.append(float(line.split()[-1]))
            elif line.strip().startswith('Validation Uy MSE'):
                val_uy_mse.append(float(line.split()[-1]))
            elif line.strip().startswith('Validation p MSE'):
                val_p_mse.append(float(line.split()[-1]))
    
    return {
        'train_loss': train_loss,
        'train_total_mse': train_total_mse,
        'train_ux_mse': train_ux_mse,
        'train_uy_mse': train_uy_mse,
        'train_p_mse': train_p_mse,
        'val_loss': val_loss,
        'val_total_mse': val_total_mse,
        'val_ux_mse': val_ux_mse,
        'val_uy_mse': val_uy_mse,
        'val_p_mse': val_p_mse
    }

def visualize_training_metrics(metrics, initial_epoch=0, save_path=None):
    """Visualize training metrics from parsed log data."""
    train_loss = metrics['train_loss']
    train_total_mse = metrics['train_total_mse']
    train_ux_mse = metrics['train_ux_mse']
    train_uy_mse = metrics['train_uy_mse']
    train_p_mse = metrics['train_p_mse']
    val_loss = metrics['val_loss']
    val_total_mse = metrics['val_total_mse']
    val_ux_mse = metrics['val_ux_mse']
    val_uy_mse = metrics['val_uy_mse']
    val_p_mse = metrics['val_p_mse']
    
    epochs = list(range(1, len(train_loss) + 1))

    fig, axs = plt.subplots(ncols=2, sharex=True, figsize=(14, 8))

    axs[0].plot(epochs[initial_epoch:], train_loss[initial_epoch:], label='Train Loss')
    axs[0].plot(epochs[initial_epoch:], val_loss[initial_epoch:], label='Validation Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Training and Validation Loss')
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.7)

    axs[1].plot(epochs[initial_epoch:], train_total_mse[initial_epoch:], label='Train Total MSE')
    axs[1].plot(epochs[initial_epoch:], train_ux_mse[initial_epoch:], label='Train Ux MSE')
    axs[1].plot(epochs[initial_epoch:], train_uy_mse[initial_epoch:], label='Train Uy MSE')
    axs[1].plot(epochs[initial_epoch:], train_p_mse[initial_epoch:], label='Train p MSE')
    axs[1].plot(epochs[initial_epoch:], val_total_mse[initial_epoch:], label='Validation Total MSE')
    axs[1].plot(epochs[initial_epoch:], val_ux_mse[initial_epoch:], label='Validation Ux MSE')
    axs[1].plot(epochs[initial_epoch:], val_uy_mse[initial_epoch:], label='Validation Uy MSE')
    axs[1].plot(epochs[initial_epoch:], val_p_mse[initial_epoch:], label='Validation p MSE')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('MSE')
    axs[1].set_title('Component-wise MSE Metrics')
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

def compare_training_logs(log_paths, labels=None, initial_epoch=0, save_path=None):
    """Compare metrics from multiple training logs."""
    if labels is None:
        labels = [os.path.basename(path) for path in log_paths]
    
    metrics_list = [parse_log_file(path) for path in log_paths]
    
    fig, axs = plt.subplots(ncols=2, sharex=True, figsize=(15, 8))
    
    # Plot training and validation loss
    for i, metrics in enumerate(metrics_list):
        train_loss = metrics['train_loss']
        val_loss = metrics['val_loss']
        epochs = list(range(1, len(train_loss) + 1))
        
        axs[0].plot(epochs[initial_epoch:], train_loss[initial_epoch:], 
                   label=f'{labels[i]} - Train Loss', linestyle='-')
        axs[0].plot(epochs[initial_epoch:], val_loss[initial_epoch:], 
                   label=f'{labels[i]} - Val Loss', linestyle='--')
    
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Comparison of Training and Validation Loss')
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot total MSE
    for i, metrics in enumerate(metrics_list):
        total_mse = metrics['train_total_mse']
        val_total_mse = metrics['val_total_mse']
        epochs = list(range(1, len(total_mse) + 1))
        
        axs[1].plot(epochs[initial_epoch:], total_mse[initial_epoch:], 
                   label=f'{labels[i]} - Train MSE', linestyle='-')
        axs[1].plot(epochs[initial_epoch:], val_total_mse[initial_epoch:], 
                   label=f'{labels[i]} - Val MSE', linestyle='--')
    
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Total MSE')
    axs[1].set_title('Comparison of Total MSE')
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison figure saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize DeepCFD training logs')
    
    parser.add_argument('log_files', type=str, nargs='+', 
                        help='Path to log file(s) to visualize')
    parser.add_argument('--initial-epoch', type=int, default=0,
                        help='Starting epoch for visualization (skip early training noise)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare multiple log files')
    parser.add_argument('--labels', type=str, nargs='+',
                        help='Labels for log files when comparing')
    parser.add_argument('--save', type=str, default=None,
                        help='Save figure to specified path instead of displaying')
    
    args = parser.parse_args()
    
    if args.compare and len(args.log_files) > 1:
        compare_training_logs(
            args.log_files, 
            args.labels, 
            args.initial_epoch, 
            args.save
        )
    elif len(args.log_files) == 1:
        metrics = parse_log_file(args.log_files[0])
        visualize_training_metrics(
            metrics, 
            args.initial_epoch, 
            args.save
        )
    else:
        print("Please provide at least one log file")

