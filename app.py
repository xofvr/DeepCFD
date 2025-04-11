import streamlit as st
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import io
import sys
import os
from PIL import Image

# Add path for deepcfd imports
sys.path.append(".")
sys.path.append("./src")

# Setup page config
st.set_page_config(
    page_title="TransformerDeepCFD Dashboard",
    layout="wide"
)

# Page title
st.title("TransformerDeepCFD Dashboard")
st.write("Upload your trained model and dataset to visualize flow field predictions")

# Sidebar for model loading
with st.sidebar:
    st.header("Model & Data Upload")
    
    # Model file uploader
    model_file = st.file_uploader("Upload trained model (.pt file)", type=["pt"])
    
    # Data file uploader
    data_file = st.file_uploader("Upload test data (.pkl file)", type=["pkl"])
    
    # Add info about model architecture
    st.subheader("Model Architecture")
    architecture = st.selectbox(
        "Select model architecture",
        ["UNetEx", "TransformerUNetEx", "AutoEncoder"]
    )
    
    # Or choose from existing models
    st.markdown("---")
    st.subheader("Or select from available models")
    
    # Detect available models
    model_paths = []
    model_names = []
    
    # Check main project directory
    for file in os.listdir('.'):
        if file.endswith('.pt'):
            model_paths.append(os.path.join('.', file))
            model_names.append(file)
    
    # Check models directory if it exists
    if os.path.exists('./models'):
        for file in os.listdir('./models'):
            if file.endswith('.pt'):
                model_paths.append(os.path.join('./models', file))
                model_names.append(f"models/{file}")
                
    # Check in checkpoint directory if it exists
    if os.path.exists('./checkpoint'):
        for file in os.listdir('./checkpoint'):
            if file.endswith('.pt'):
                model_paths.append(os.path.join('./checkpoint', file))
                model_names.append(f"checkpoint/{file}")
    
    # Add option for no selection
    model_names.insert(0, "None (Use uploaded file)")
    model_paths.insert(0, None)
    
    selected_model_name = st.selectbox("Select model", model_names)
    selected_model_path = model_paths[model_names.index(selected_model_name)]
    
    # Data selection from filesystem
    st.markdown("---")
    st.subheader("Or select data from filesystem")
    
    # Default data paths
    default_input_path = "./data/dataX.pkl"
    default_output_path = "./data/dataY.pkl"
    
    use_default_data = st.checkbox("Use default data paths", value=True)
    
    if use_default_data:
        input_data_path = default_input_path
        output_data_path = default_output_path
    else:
        input_data_path = st.text_input("Input data path (dataX.pkl)", default_input_path)
        output_data_path = st.text_input("Output data path (dataY.pkl)", default_output_path)

# Function to load model
def load_model(model_bytes, architecture):
    try:
        # Load state dict
        state_dict = torch.load(io.BytesIO(model_bytes), map_location=torch.device('cpu'))
        
        # Get model parameters from state dict or use defaults
        filters = state_dict.get("filters", [8, 16, 32, 32])
        kernel_size = state_dict.get("kernel_size", 5)
        
        # Additional parameters for transformer models
        transformer_dim = state_dict.get("transformer_dim", 128)
        nhead = state_dict.get("nhead", 4)
        num_layers = state_dict.get("num_layers", 2)
        
        # Import correct model architecture
        if architecture == "UNetEx":
            from deepcfd.models.UNetEx import UNetEx
            model = UNetEx(3, 3, filters=filters, kernel_size=kernel_size)
        elif architecture == "TransformerUNetEx":
            from deepcfd.models.TransformerUNetEx import TransformerUNetEx
            model = TransformerUNetEx(3, 3, 
                                     filters=filters, 
                                     kernel_size=kernel_size,
                                     transformer_dim=transformer_dim,
                                     nhead=nhead,
                                     num_layers=num_layers)
        else:  # AutoEncoder
            from deepcfd.models.AutoEncoder import AutoEncoder
            model = AutoEncoder(3, 3, filters=filters, kernel_size=kernel_size)
        
        # Clean state dict for loading - remove metadata keys
        clean_state_dict = {k: v for k, v in state_dict.items() 
                           if not k.startswith('_') and k not in 
                           ["architecture", "input_shape", "filters", "kernel_size", 
                            "transformer_dim", "nhead", "num_layers"]}
        
        # Load the cleaned state dict into model
        model.load_state_dict(clean_state_dict, strict=False)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


# Function to load data
def load_data(data_bytes=None, file_path=None):
    try:
        if data_bytes:
            data = pickle.load(io.BytesIO(data_bytes))
        elif file_path and os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        else:
            st.error("No valid data source provided")
            return None
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


# Function to visualize results
def visualize_cfd_results(truth, prediction, sample_idx=0):
    """
    Visualizes CFD results for a specific sample
    
    Args:
        truth: Ground truth data array [batch, channels, height, width]
        prediction: Predicted data array [batch, channels, height, width]
        sample_idx: Index of the sample to visualize
    
    Returns:
        fig: Matplotlib figure
    """
    # Extract the sample
    sample_truth = truth[sample_idx]
    sample_pred = prediction[sample_idx]
    
    # Calculate error
    error = np.abs(sample_pred - sample_truth)
    
    # Create figure with subplots
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    # Channel names
    channel_names = ["Ux [m/s]", "Uy [m/s]", "p [m²/s²]"]
    
    # Set up color ranges for consistent visualization
    vmin_max = [
        (0, 0.15),     # Ux
        (-0.045, 0.045),  # Uy
        (0, 0.015)     # p
    ]
    
    error_max = [0.018, 0.008, 0.007]  # Max error values for each channel
    
    # Plot for each channel
    for i in range(3):
        # Ground truth
        ax = axs[i, 0]
        im = ax.imshow(sample_truth[i], cmap='jet', origin='lower', 
                   vmin=vmin_max[i][0], vmax=vmin_max[i][1])
        ax.set_title(f"Ground Truth - {channel_names[i]}")
        plt.colorbar(im, ax=ax)
        
        # Prediction
        ax = axs[i, 1]
        im = ax.imshow(sample_pred[i], cmap='jet', origin='lower',
                   vmin=vmin_max[i][0], vmax=vmin_max[i][1])
        ax.set_title(f"Prediction - {channel_names[i]}")
        plt.colorbar(im, ax=ax)
        
        # Error
        ax = axs[i, 2]
        im = ax.imshow(error[i], cmap='jet', origin='lower', 
                   vmin=0, vmax=error_max[i])
        ax.set_title(f"Absolute Error - {channel_names[i]}")
        plt.colorbar(im, ax=ax)
    
    return fig


def load_and_plot_training_log(log_file):
    """
    Load and plot training log data showing loss over epochs
    
    Args:
        log_file: Path to training log file
    
    Returns:
        fig: Matplotlib figure
    """
    try:
        # Read log file
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Parse data
        epochs = []
        train_loss = []
        val_loss = []
        
        for line in lines:
            if "Epoch" in line and "Train Loss" in line and "Val Loss" in line:
                parts = line.strip().split()
                
                # Extract epoch number
                epoch_idx = parts.index("Epoch")
                if epoch_idx + 1 < len(parts):
                    try:
                        # Remove colon if present
                        epoch_str = parts[epoch_idx + 1].rstrip(':')
                        epochs.append(int(epoch_str))
                    except ValueError:
                        continue
                
                # Extract train loss
                train_idx = parts.index("Loss:")
                if train_idx + 1 < len(parts):
                    try:
                        train_loss.append(float(parts[train_idx + 1]))
                    except ValueError:
                        continue
                
                # Extract validation loss
                val_idx = parts.index("Loss:", train_idx + 1) if "Val" in line else -1
                if val_idx != -1 and val_idx + 1 < len(parts):
                    try:
                        val_loss.append(float(parts[val_idx + 1]))
                    except ValueError:
                        continue
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if epochs and train_loss:
            ax.plot(epochs, train_loss, 'b-', label='Training Loss')
        if epochs and val_loss and len(val_loss) == len(epochs):
            ax.plot(epochs, val_loss, 'r-', label='Validation Loss')
        
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True)
        
        return fig
    except Exception as e:
        st.error(f"Error loading training log: {str(e)}")
        return None


# Main dashboard logic
def run_inference(model, x_data, y_data=None):
    # Show data information
    st.subheader("Data Information")
    st.write(f"Input shape: {x_data.shape}")
    has_ground_truth = y_data is not None
    
    if has_ground_truth:
        st.write(f"Ground truth shape: {y_data.shape}")
        st.write(f"Number of samples: {x_data.shape[0]}")
    
    # Sample selection
    if x_data.shape[0] > 1:
        sample_idx = st.slider("Select sample index", 0, x_data.shape[0]-1, 0)
    else:
        sample_idx = 0
        
    # Run inference
    with st.spinner("Running inference..."):
        with torch.no_grad():
            # Process only the selected sample
            input_tensor = x_data[sample_idx:sample_idx+1]
            prediction = model(input_tensor)
            
            # Move back to CPU for visualization
            prediction = prediction.cpu().numpy()
            
    # Visualize results
    st.subheader("Visualization Results")
    
    if has_ground_truth:
        # Use ground truth for comparison
        truth = y_data[sample_idx:sample_idx+1].numpy()
        fig = visualize_cfd_results(truth, prediction)
        st.pyplot(fig)
    else:
        # Only show prediction
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        # Channel names
        channel_names = ["Ux [m/s]", "Uy [m/s]", "p [m²/s²]"]
        
        for i in range(3):
            im = axs[i].imshow(prediction[0, i], cmap='jet', origin='lower')
            axs[i].set_title(f"Prediction - {channel_names[i]}")
            plt.colorbar(im, ax=axs[i])
        
        st.pyplot(fig)
    
    # Display input geometry
    st.subheader("Input Geometry")
    
    # Assuming first channel is SDF and second is flow region
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    # Show SDF
    im = axs[0].imshow(x_data[sample_idx, 0].numpy(), cmap='RdBu', origin='lower')
    axs[0].set_title("Signed Distance Function")
    plt.colorbar(im, ax=axs[0])
    
    # Show flow region
    im = axs[1].imshow(x_data[sample_idx, 1].numpy(), cmap='viridis', origin='lower')
    axs[1].set_title("Flow Region")
    plt.colorbar(im, ax=axs[1])
    
    st.pyplot(fig)
    
    return sample_idx, prediction


# Main execution
if __name__ == "__main__":
    # Main logic - prioritize uploaded files, then selected files
    model = None
    x_data = None
    y_data = None
    
    # Try to load model from uploaded file or selected path
    if model_file:
        model_bytes = model_file.getvalue()
        model = load_model(model_bytes, architecture)
    elif selected_model_path:
        try:
            model = load_model(None, architecture)
            # Load state dict directly from file
            state_dict = torch.load(selected_model_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            model.eval()
            st.success(f"Model loaded from {selected_model_name}")
        except Exception as e:
            st.error(f"Error loading model from {selected_model_path}: {str(e)}")
            model = None
    
    # Try to load data from uploaded file or selected paths
    if data_file:
        data_bytes = data_file.getvalue()
        data = load_data(data_bytes=data_bytes)
        if isinstance(data, (tuple, list)) and len(data) == 2:
            x_data, y_data = data
        else:
            x_data = data
    elif use_default_data or (input_data_path and os.path.exists(input_data_path)):
        # Try to load X data
        x_data = load_data(file_path=input_data_path)
        
        # Try to load Y data if it exists
        if output_data_path and os.path.exists(output_data_path):
            y_data = load_data(file_path=output_data_path)
            st.success(f"Input and ground truth data loaded successfully")
        else:
            st.warning(f"Ground truth data not found at {output_data_path}")
    
    # Run inference if both model and data are available
    if model is not None and x_data is not None:
        # Convert to tensor if not already
        if not isinstance(x_data, torch.Tensor):
            x_data = torch.FloatTensor(x_data)
        
        if y_data is not None and not isinstance(y_data, torch.Tensor):
            y_data = torch.FloatTensor(y_data)
        
        # Run inference and visualization
        run_inference(model, x_data, y_data)
        
        # Check if there's a training log file associated with the selected model
        if model is not None:
            st.markdown("---")
            st.subheader("Training Progress")
            
            # Look for training log files
            available_logs = []
            
            # Try to find log file with same name as model
            if selected_model_path:
                model_name = os.path.basename(selected_model_path).replace('.pt', '')
                for log_name in [f"training_log_{model_name}.txt", f"training_log{model_name}.txt", "training_log.txt"]:
                    if os.path.exists(log_name):
                        available_logs.append((log_name, log_name))
            
            # Check in root directory
            for log_file in ["training_log.txt", "training_log1.txt", "training_log2.txt"]:
                if os.path.exists(log_file) and (log_file, log_file) not in available_logs:
                    available_logs.append((log_file, log_file))
                    
            # Check in deepcfd directory
            for log_file in ["training_log.txt", "training_log1.txt", "training_log2.txt"]:
                path = os.path.join("DeepCFD", log_file)
                if os.path.exists(path) and (path, log_file) not in available_logs:
                    available_logs.append((path, log_file))
            
            if available_logs:
                selected_log = st.selectbox("Select training log", 
                                            options=[name for _, name in available_logs],
                                            index=0)
                
                # Find the file path for the selected name
                selected_log_path = next((path for path, name in available_logs if name == selected_log), None)
                
                if selected_log_path:
                    log_fig = load_and_plot_training_log(selected_log_path)
                    if log_fig:
                        st.pyplot(log_fig)
                    else:
                        st.info("Could not parse training log file format.")
            else:
                st.info("No training log files found for this model.")
                
            # Add model stats
            if model:
                st.subheader("Model Statistics")
                # Calculate number of parameters
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                # Display in columns
                cols = st.columns(4)
                cols[0].metric("Total Parameters", f"{total_params:,}")
                cols[1].metric("Trainable Parameters", f"{trainable_params:,}")
                
                # Try to extract epochs and learning rate from state dict if available
                if hasattr(model, '_trained_epochs'):
                    cols[2].metric("Epochs Trained", model._trained_epochs)
                if hasattr(model, '_learning_rate'):
                    cols[3].metric("Learning Rate", model._learning_rate)
    else:
        if model is None:
            st.info("Please upload or select a model file to continue.")
        if x_data is None:
            st.info("Please upload or select input data to continue.")
        
        # Display example images
        cols = st.columns(2)
        with cols[0]:
            st.subheader("Sample Velocity Field (Ux)")
            placeholder_img = Image.new('RGB', (300, 300), color='white')
            st.image(placeholder_img, caption="Load data and model to see prediction")
        
        with cols[1]:
            st.subheader("Sample Pressure Field (p)")
            placeholder_img = Image.new('RGB', (300, 300), color='white')
            st.image(placeholder_img, caption="Load data and model to see prediction")
        
        # Add readme section showing architecture diagram
        if os.path.exists("./ReadmeFiles/arch.png"):
            st.markdown("---")
            st.subheader("TransformerDeepCFD Architecture")
            st.image("./ReadmeFiles/arch.png", caption="Model Architecture")
            
            # Add example flows if available
            example_files = [f for f in os.listdir("./ReadmeFiles") if f.endswith(".png") and f != "arch.png" and f != "DataStruct.png"]
            if len(example_files) > 0:
                st.subheader("Example Flow Predictions")
                cols = st.columns(2)
                for i, file in enumerate(example_files[:4]):  # Show up to 4 examples
                    with cols[i % 2]:
                        st.image(f"./ReadmeFiles/{file}", caption=file.replace(".png", ""))