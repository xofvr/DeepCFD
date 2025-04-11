"""
Configuration module for DeepCFD.
Handles command-line argument parsing and configuration management.
"""
import os
import sys
import getopt
import torch
import json
from typing import Dict, Any, List, Optional, Union

class DeepCFDConfig:
    """
    Configuration class for DeepCFD.
    Handles command-line arguments and provides configuration options.
    """
    def __init__(self):
        # Default configuration
        self.device = self._get_default_device()
        self.net = "UNetEx"
        self.model_input = "dataX.pkl"
        self.model_output = "dataY.pkl"
        self.output = "mymodel.pt"
        self.kernel_size = 5
        self.filters = [8, 16, 32, 32]
        self.learning_rate = 0.001
        self.epochs = 2000
        self.batch_size = 32
        self.patience = 500
        self.visualize = False
        self.transformer_dim = 128
        self.nhead = 4
        self.num_layers = 2
        self.use_augmentation = False
        self.aug_flip_prob = 0.3
        self.aug_rotate_prob = 0.3
        self.aug_noise_prob = 0.2
        self.weight_decay = 0.005
        self.use_batch_norm = False
        self.use_weight_norm = False
        self.num_workers = 4
        
        # Will be set after parsing arguments
        self.net_class = None
        
    def _get_default_device(self) -> torch.device:
        """Determine the best available device"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def parse_args(self, argv: List[str]) -> Dict[str, Any]:
        """
        Parse command-line arguments and update configuration.
        
        Args:
            argv: Command-line arguments
            
        Returns:
            Dictionary with configuration options
        """
        try:
            opts, args = getopt.getopt(
                argv, "hd:n:mi:mo:o:k:f:l:e:b:p:va:",
                [
                    "device=",
                    "net=",
                    "model-input=",
                    "model-output=",
                    "output=",
                    "kernel-size=",
                    "filters=",
                    "learning-rate=",
                    "epochs=",
                    "batch-size=",
                    "patience=",
                    "visualize",
                    "augmentation",
                    "transformer-dim=",
                    "nhead=",
                    "num-layers=",
                    "weight-decay=",
                    "batch-norm",
                    "weight-norm",
                    "num-workers="
                ]
            )
        except getopt.GetoptError as e:
            print(e)
            print("python -m deepcfd --help")
            sys.exit(2)

        for opt, arg in opts:
            if opt in ('-h', '--help'):
                self._print_help()
                sys.exit()
            elif opt in ("-d", "--device"):
                self._set_device(arg)
            elif opt in ("-n", "--net"):
                self.net = arg
            elif opt in ("-mi", "--model-input"):
                self.model_input = arg
            elif opt in ("-mo", "--model-output"):
                self.model_output = arg
            elif opt in ("-k", "--kernel-size"):
                self.kernel_size = int(arg)
            elif opt in ("-f", "--filters"):
                self.filters = [int(x) for x in arg.split(',')]
            elif opt in ("-l", "--learning-rate"):
                self.learning_rate = float(arg)
            elif opt in ("-e", "--epochs"):
                self.epochs = int(arg)
            elif opt in ("-b", "--batch-size"):
                self.batch_size = int(arg)
            elif opt in ("-o", "--output"):
                self.output = arg
            elif opt in ("-p", "--patience"):
                self.patience = int(arg)
            elif opt in ("-v", "--visualize"):
                self.visualize = True
            elif opt in ("-a", "--augmentation"):
                self.use_augmentation = True
            elif opt == "--transformer-dim":
                self.transformer_dim = int(arg)
            elif opt == "--nhead":
                self.nhead = int(arg)
            elif opt == "--num-layers":
                self.num_layers = int(arg)
            elif opt == "--weight-decay":
                self.weight_decay = float(arg)
            elif opt == "--batch-norm":
                self.use_batch_norm = True
            elif opt == "--weight-norm":
                self.use_weight_norm = True
            elif opt == "--num-workers":
                self.num_workers = int(arg)

        # Load the appropriate network class
        self._load_network_class()
        
        return self.to_dict()
    
    def _set_device(self, device_str: str) -> None:
        """Set the device based on the provided string"""
        if device_str == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif device_str == "cpu":
            self.device = torch.device("cpu")
        elif device_str.startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device(device_str)
        else:
            print(f"Unknown device {device_str}, using {self.device} instead")
    
    def _load_network_class(self) -> None:
        """Load the appropriate network class based on the network name"""
        if self.net == "UNetEx":
            from .models.UNetEx import UNetEx
            self.net_class = UNetEx
        elif self.net == "TransformerUNetEx":
            from .models.TransformerUNetEx import TransformerUNetEx
            self.net_class = TransformerUNetEx
        elif self.net == "AutoEncoder":
            from .models.AutoEncoder import AutoEncoder
            self.net_class = AutoEncoder
        else:
            print(f"Unknown network {self.net}, falling back to UNetEx")
            from .models.UNetEx import UNetEx
            self.net_class = UNetEx
            self.net = "UNetEx"
    
    def _print_help(self) -> None:
        """Print help information"""
        print("""DeepCFD - Neural network-based CFD solver

Usage: python -m deepcfd [OPTIONS]

Basic options:
  -h, --help                Show this help message and exit
  -d, --device              Device: 'cpu', 'cuda', 'cuda:0', 'cuda:0,cuda:n' (default: best available)
  -n, --net                 Network architecture: UNetEx, TransformerUNetEx, or AutoEncoder (default: UNetEx)
  -mi, --model-input        Input dataset path (default: dataX.pkl)
  -mo, --model-output       Output dataset path (default: dataY.pkl)
  -o, --output              Model output path (default: mymodel.pt)
  -k, --kernel-size         Kernel size (default: 5)
  -f, --filters             Filter sizes (comma-separated, default: 8,16,32,32)
  -l, --learning-rate       Learning rate (default: 0.001)
  -e, --epochs              Number of epochs (default: 2000)
  -b, --batch-size          Training batch size (default: 32)
  -p, --patience            Epochs for early stopping (default: 500)
  -v, --visualize           Visualize ground-truth vs prediction plots (default: False)
  -a, --augmentation        Use data augmentation (default: False)

Advanced options:
  --transformer-dim         Transformer feature dimension (default: 128)
  --nhead                   Number of attention heads (default: 4)
  --num-layers              Number of transformer layers (default: 2)
  --weight-decay            Weight decay for optimizer (default: 0.005)
  --batch-norm              Use batch normalization (default: False)
  --weight-norm             Use weight normalization (default: False)
  --num-workers             Number of data loading workers (default: 4)

Examples:
  python -m deepcfd --net UNetEx --device cuda
  python -m deepcfd --net TransformerUNetEx --model-input data/dataX.pkl --model-output data/dataY.pkl
""")
    
    def save_to_json(self, filepath: str) -> None:
        """
        Save configuration to a JSON file
        
        Args:
            filepath: Path to save the configuration
        """
        config_dict = self.to_dict()
        # Convert torch.device to string for JSON serialization
        config_dict['device'] = str(config_dict['device'])
        # Remove net_class as it's not serializable
        if 'net_class' in config_dict:
            del config_dict['net_class']
            
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
            
    @classmethod
    def load_from_json(cls, filepath: str) -> 'DeepCFDConfig':
        """
        Load configuration from a JSON file
        
        Args:
            filepath: Path to the configuration file
            
        Returns:
            DeepCFDConfig instance with loaded settings
        """
        config = cls()
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Set attributes from the loaded dictionary
        for key, value in config_dict.items():
            if hasattr(config, key):
                if key == 'device':
                    config._set_device(value)
                else:
                    setattr(config, key, value)
        
        # Load the network class
        config._load_network_class()
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary
        
        Returns:
            Dictionary with configuration options
        """
        return {
            'device': self.device,
            'net': self.net,
            'net_class': self.net_class,
            'model_input': self.model_input,
            'model_output': self.model_output,
            'output': self.output,
            'kernel_size': self.kernel_size,
            'filters': self.filters,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'patience': self.patience,
            'visualize': self.visualize,
            'use_augmentation': self.use_augmentation,
            'aug_flip_prob': self.aug_flip_prob,
            'aug_rotate_prob': self.aug_rotate_prob,
            'aug_noise_prob': self.aug_noise_prob,
            'transformer_dim': self.transformer_dim,
            'nhead': self.nhead,
            'num_layers': self.num_layers,
            'weight_decay': self.weight_decay,
            'use_batch_norm': self.use_batch_norm,
            'use_weight_norm': self.use_weight_norm,
            'num_workers': self.num_workers
        }