import os
import json
import torch
import pickle
import random
import getopt
import sys
from .train_functions import *
from .functions import *
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.autograd import Variable
from .lr_scheduler import TransformerLRScheduler
from .data_augmentation import FluidDataAugmentation, create_augmented_dataloader
from .config import DeepCFDConfig

# changed to mps from cuda 
def parseOpts(argv):
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    net= "UNetEx"
    kernel_size = 5
    filters = [8, 16, 32, 32]
    model_input = "dataX.pkl"
    model_output = "dataY.pkl"
    output = "mymodel.pt"
    learning_rate = 0.001
    epochs = 2000
    batch_size = 32
    patience = 500
    visualize = False

    try:
        opts, args = getopt.getopt(
            argv,"hd:n:mi:mo:o:k:f:l:e:b:p:v",
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
                "visualize"
            ]
        )
    except getopt.GetoptError as e:
       print(e)
       print("python -m deepcfd --help")
       sys.exit(2)

    for opt, arg in opts:
        if opt == '-h' or opt == '--help':
            print("deepcfd "
                "\n    -d  <device> device: 'cpu', 'cuda', 'cuda:0', 'cuda:0,cuda:n', (default: cuda if available)"
                "\n    -n  <net> network architecture: UNet, UNetEx, "
                    "TransformerUNetEx or AutoEncoder (default: UNetEx)"
                "\n    -mi <model-input>  input dataset with sdf1,"
                    "flow-region and sdf2 fields (default: dataX.pkl)"
                "\n    -mo <model-output>  output dataset with Ux,"
                    "Uy and p (default: dataY.pkl)"
                "\n    -o <output>  model output (default: mymodel.pt)"
                "\n    -k <kernel-size>  kernel size (default: 5)"
                "\n    -f <filters>  filter sizes (default: 8,16,32,32)"
                "\n    -l <learning-rate>  learning rate (default: 0.001)"
                "\n    -e <epochs>  number of epochs (default: 1000)"
                "\n    -b <batch-size>  training batch size (default: 32)"
                "\n    -p <patience>  number of epochs for early stopping (default: 300)"
                "\n    -v <visualize> flag for visualizing ground-truth vs prediction plots (default: False)\n"
                "\n    -n <net> network architecture: UNet, UNetEx, "
                    "TransformerUNetEx or AutoEncoder (default: UNetEx)"
            )
            sys.exit()
        elif opt in ("-d", "--device"):
            if arg == "mps" and torch.backends.mps.is_available():
                device = torch.device("mps")
            elif arg == "cpu":
                device = torch.device("cpu")
            elif arg.startswith("cuda") and torch.cuda.is_available():
                device = torch.device(arg)
            else:
                print("Unkown device " + str(arg) + ", only 'cpu', 'cuda'"
                    "'cuda:index', or comma-separated list of 'cuda:index'"
                    "are supported")
                exit(0)
        elif opt in ("-n", "--net"):

            if (arg == "UNetEx"):
                from .models.UNetEx import UNetEx
                net = UNetEx
            elif (arg == "TransformerUNetEx"):
                from .models.TransformerUNetEx import TransformerUNetEx
                net = TransformerUNetEx

            elif (arg == "AutoEncoder"):
                from .models.AutoEncoder import AutoEncoder
                net = AutoEncoder
            else:
                print("Unkown network " + str(arg) + ", only UNet, UNetEx"
                    "and AutoEncoder are supported")
                exit(0)
        elif opt in ("-mi", "--model-input"):
            model_input = arg
        elif opt in ("-mo", "--model-output"):
            model_output = arg
        elif opt in ("-k", "--kernel-size"):
            kernel_size = int(arg)
        elif opt in ("-f", "--filters"):
            filters = [int(x) for x in arg.split(',')]
        elif opt in ("-l", "--learning-rate"):
            learning_rate = float(arg)
        elif opt in ("-e", "--epochs"):
            epochs = int(arg)
        elif opt in ("-b", "--batch-size"):
            batch_size = int(arg)
        elif opt in ("-o", "--output"):
            output = arg
        elif opt in ("-p", "--patience"):
            patience = int(arg)
        elif opt in ("-v", "--visualize"):
            visualize = True

    if '--net' not in sys.argv or '-n' not in sys.argv:
        from .models.UNetEx import UNetEx
        net = UNetEx

    options = {
        'device': device,
        'net': net,
        'model_input': model_input,
        'model_output': model_output,
        'output': output,
        'kernel_size': kernel_size,
        'filters': filters,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'batch_size': batch_size,
        'patience': patience,
        'visualize': visualize,
    }

    return options

def main():
    # Parse command-line arguments using the new config system
    config = DeepCFDConfig()
    options = config.parse_args(sys.argv[1:])
    
    # Save the configuration for reproducibility
    config_filename = os.path.splitext(options["output"])[0] + "_config.json"
    config.save_to_json(config_filename)
    print(f"Configuration saved to {config_filename}")

    # Load data
    x = pickle.load(open(options["model_input"], "rb"))
    y = pickle.load(open(options["model_output"], "rb"))

    # Shuffle the data
    indices = list(range(len(x)))
    random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)

    batch = x.shape[0]
    nx = x.shape[2]
    ny = x.shape[3]

    # Calculate the weights for each channel based on their magnitude
    channels_weights = torch.sqrt(torch.mean(y.permute(0, 2, 3, 1)
        .reshape((batch*nx*ny,3)) ** 2, dim=0)).view(1, -1, 1, 1)
    channels_weights = channels_weights.to(options["device"])

    # Ensure output directory exists
    dirname = os.path.dirname(os.path.abspath(options["output"]))
    if dirname and not os.path.exists(dirname):
       os.makedirs(dirname, exist_ok=True)

    # Split dataset into 70% train and 30% test
    train_data, test_data = split_tensors(x, y, ratio=0.7)
    
    # Create datasets
    train_dataset, test_dataset = TensorDataset(*train_data), TensorDataset(*test_data)
    
    # Set up data augmentation if requested
    if config.use_augmentation:
        print("Using data augmentation")
        augmentation = FluidDataAugmentation(
            flip_prob=config.aug_flip_prob,
            rotate_prob=config.aug_rotate_prob,
            noise_prob=config.aug_noise_prob
        )
        train_loader = create_augmented_dataloader(
            train_dataset, 
            batch_size=options["batch_size"], 
            shuffle=True,
            augmentation=augmentation,
            num_workers=config.num_workers,
            pin_memory=True
        )
    else:
        train_loader = create_augmented_dataloader(
            train_dataset, 
            batch_size=options["batch_size"], 
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        )
    
    # Get a sample for testing
    test_x, test_y = test_dataset[:]
    
    # Set reproducible seed
    torch.manual_seed(0)

    # Initialize model based on the selected architecture
    model_args = {
        'in_channels': 3,
        'out_channels': 3,
        'filters': options["filters"],
        'kernel_size': options["kernel_size"],
        'batch_norm': config.use_batch_norm,
        'weight_norm': config.use_weight_norm
    }
    
    # Add transformer-specific parameters if using TransformerUNetEx
    if options["net"] == "TransformerUNetEx":
        model_args.update({
            'transformer_dim': config.transformer_dim,
            'nhead': config.nhead,
            'num_layers': config.num_layers
        })
        
    model = options["net_class"](**model_args)

    # Define optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=options["learning_rate"],
        weight_decay=config.weight_decay
    )
    
    # Configure learning rate scheduler
    warmup_steps = int(0.02 * options["epochs"])  # 2% of total epochs for warmup
    scheduler = TransformerLRScheduler(
        optimizer, 
        warmup_steps=warmup_steps,
        max_steps=options["epochs"],
        min_lr=1e-6,
        warmup_init_lr=1e-4
    )

    # Initialize metrics tracking
    metrics = {
        'train_loss_curve': [],
        'test_loss_curve': [],
        'train_mse_curve': [],
        'test_mse_curve': [],
        'train_ux_curve': [],
        'test_ux_curve': [],
        'train_uy_curve': [],
        'test_uy_curve': [],
        'train_p_curve': [],
        'test_p_curve': []
    }
    
    def after_epoch(scope):
        """Callback function after each epoch to track metrics"""
        metrics['train_loss_curve'].append(scope["train_loss"])
        metrics['test_loss_curve'].append(scope["val_loss"])
        metrics['train_mse_curve'].append(scope["train_metrics"]["mse"])
        metrics['test_mse_curve'].append(scope["val_metrics"]["mse"])
        metrics['train_ux_curve'].append(scope["train_metrics"]["ux"])
        metrics['test_ux_curve'].append(scope["val_metrics"]["ux"])
        metrics['train_uy_curve'].append(scope["train_metrics"]["uy"])
        metrics['test_uy_curve'].append(scope["val_metrics"]["uy"])
        metrics['train_p_curve'].append(scope["train_metrics"]["p"])
        metrics['test_p_curve'].append(scope["val_metrics"]["p"])

    def loss_func(model, batch):
        """Custom loss function with component weighting"""
        x, y = batch
        x = x.to(options["device"])
        y = y.to(options["device"])
        output = model(x)
        
        # Calculate component-wise loss
        lossu = ((output[:,0,:,:] - y[:,0,:,:]) ** 2).reshape(
            (output.shape[0],1,output.shape[2],output.shape[3]))
        lossv = ((output[:,1,:,:] - y[:,1,:,:]) ** 2).reshape(
            (output.shape[0],1,output.shape[2],output.shape[3]))
        lossp = torch.abs((output[:,2,:,:] - y[:,2,:,:])).reshape(
            (output.shape[0],1,output.shape[2],output.shape[3]))
        
        # Apply component importance weighting
        weighted_loss = (lossu * 0.4 + lossv * 0.4 + lossp * 0.2) / channels_weights.to(output.device)
        
        return torch.sum(weighted_loss), output
    
    # Train the model
    DeepCFD, train_metrics, train_loss, test_metrics, test_loss = train_model(
        model,
        loss_func,
        train_dataset,
        test_dataset,
        optimizer,
        scheduler=scheduler,
        epochs=options["epochs"],
        batch_size=options["batch_size"],
        device=options["device"],
        m_mse_name="Total MSE",
        m_mse_on_batch=lambda scope:
            float(torch.sum((scope["output"] - scope["batch"][1]) ** 2)),
        m_mse_on_epoch=lambda scope:
            sum(scope["list"]) / len(scope["dataset"]),
        m_ux_name="Ux MSE",
        m_ux_on_batch=lambda scope:
            float(torch.sum((scope["output"][:,0,:,:] -
            scope["batch"][1][:,0,:,:]) ** 2)),
        m_ux_on_epoch=lambda scope:
            sum(scope["list"]) / len(scope["dataset"]),
            m_uy_name="Uy MSE",
        m_uy_on_batch=lambda scope:
            float(torch.sum((scope["output"][:,1,:,:] -
            scope["batch"][1][:,1,:,:]) ** 2)),
        m_uy_on_epoch=lambda scope:
            sum(scope["list"]) / len(scope["dataset"]),
            m_p_name="p MSE",
        m_p_on_batch=lambda scope:
            float(torch.sum((scope["output"][:,2,:,:] -
            scope["batch"][1][:,2,:,:]) ** 2)),
        m_p_on_epoch=lambda scope:
            sum(scope["list"]) /
            len(scope["dataset"]), patience=options["patience"], after_epoch=after_epoch
    )

    # Save model with metadata
    state_dict = DeepCFD.state_dict()
    state_dict["input_shape"] = (1, 3, nx, ny)
    state_dict["filters"] = options["filters"]
    state_dict["kernel_size"] = options["kernel_size"]
    state_dict["architecture"] = options["net"]
    
    # For transformer models, save additional parameters
    if options["net"] == "TransformerUNetEx":
        state_dict["transformer_dim"] = config.transformer_dim
        state_dict["nhead"] = config.nhead
        state_dict["num_layers"] = config.num_layers
    
    torch.save(state_dict, options["output"])
    print(f"Model saved to {options['output']}")

    # Visualize results if requested
    if options["visualize"]:
        print("Generating visualizations...")
        sample_size = min(10, len(test_x))  # Limit to 10 samples for visualization
        out = DeepCFD(test_x[:sample_size].to(options["device"]))
        error = torch.abs(out.cpu() - test_y[:sample_size].cpu())
        s = 0
        visualize(
            test_y[:sample_size].cpu().detach().numpy(),
            out[:sample_size].cpu().detach().numpy(),
            error[:sample_size].cpu().detach().numpy(),
            s
       )

    # Save training metrics to file
    metrics_filename = os.path.splitext(options["output"])[0] + "_metrics.json"
    with open(metrics_filename, 'w') as f:
        json.dump({
            'train_loss': metrics['train_loss_curve'],
            'val_loss': metrics['test_loss_curve'],
            'train_mse': metrics['train_mse_curve'],
            'val_mse': metrics['test_mse_curve'],
            'train_ux': metrics['train_ux_curve'],
            'val_ux': metrics['test_ux_curve'],
            'train_uy': metrics['train_uy_curve'],
            'val_uy': metrics['test_uy_curve'],
            'train_p': metrics['train_p_curve'],
            'val_p': metrics['test_p_curve']
        }, f)
    print(f"Training metrics saved to {metrics_filename}")

if __name__ == "__main__":
    main()
