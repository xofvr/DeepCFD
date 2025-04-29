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

# Simplified interface with two columns for model and data selection
col1, col2 = st.columns(2)

with col1:
    # Model selection
    st.subheader("Model Selection")
    
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
                
    # Check trained_models directory if it exists
    if os.path.exists('./trained_models'):
        for file in os.listdir('./trained_models'):
            if file.endswith('.pt'):
                model_paths.append(os.path.join('./trained_models', file))
                model_names.append(f"trained_models/{file}")
    
    # Check in checkpoint directory if it exists
    if os.path.exists('./checkpoint'):
        for file in os.listdir('./checkpoint'):
            if file.endswith('.pt'):
                model_paths.append(os.path.join('./checkpoint', file))
                model_names.append(f"checkpoint/{file}")
    
    # Add option for file upload
    model_names.insert(0, "Upload model file")
    model_paths.insert(0, None)
    
    selected_model_name = st.selectbox("Select model", model_names)
    
    # Only show file uploader if upload option is selected
    model_file = None
    if selected_model_name == "Upload model file":
        model_file = st.file_uploader("Upload trained model (.pt file)", type=["pt"])
        selected_model_path = None
    else:
        selected_model_path = model_paths[model_names.index(selected_model_name)]

with col2:
    # Data selection
    st.subheader("Data Selection")
    
    # Data selection options
    data_option = st.radio(
        "Data source",
        ["Default data", "Upload data", "Custom path"]
    )
    
    # Default data paths
    default_input_path = "./data/dataX.pkl"
    default_output_path = "./data/dataY.pkl"
    
    data_file = None
    if data_option == "Upload data":
        data_file = st.file_uploader("Upload test data (.pkl file)", type=["pkl"])
        input_data_path = None
        output_data_path = None
    elif data_option == "Custom path":
        input_data_path = st.text_input("Input data path (dataX.pkl)", default_input_path)
        output_data_path = st.text_input("Output data path (dataY.pkl)", default_output_path)
    else:  # Default data
        input_data_path = default_input_path
        output_data_path = default_output_path


# Function to load model
def load_model(model_bytes):
    try:
        # Check if model_bytes is provided
        if model_bytes:
            # Load state dict from bytes
            state_dict = torch.load(io.BytesIO(model_bytes), map_location=torch.device('cpu'))
        else:
            # We're loading from a path, this will be handled differently
            return None
            
        # Try to detect if this is the enhanced transformer model
        is_enhanced_model = False
        
        # If this is the enhanced model, load its config file
        if "transformer_model_enhanced" in str(model_bytes):
            config_path = "./trained_models/transformer_model_enhanced_config.json"
            if os.path.exists(config_path):
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                filters = config.get("filters", [16, 32, 64, 64])
                kernel_size = config.get("kernel_size", 3)
                transformer_dim = config.get("transformer_dim", 192)
                nhead = config.get("nhead", 6)
                num_layers = config.get("num_layers", 3)
                is_enhanced_model = True
        
        # If not enhanced model or config not found, get from state dict or use defaults
        if not is_enhanced_model:
            filters = state_dict.get("filters", [8, 16, 32, 32])
            kernel_size = state_dict.get("kernel_size", 5)
            transformer_dim = state_dict.get("transformer_dim", 128)
            nhead = state_dict.get("nhead", 4)
            num_layers = state_dict.get("num_layers", 2)
        
        # Import TransformerUNetEx model
        from deepcfd.models.TransformerUNetEx import TransformerUNetEx
        model = TransformerUNetEx(3, 3, 
                                 filters=filters, 
                                 kernel_size=kernel_size,
                                 transformer_dim=transformer_dim,
                                 nhead=nhead,
                                 num_layers=num_layers)
        
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
        model = load_model(model_bytes)
    elif selected_model_path:
        try:
            # Check if this is the enhanced model
            if "transformer_model_enhanced" in selected_model_path:
                # Load the configuration
                config_path = "./trained_models/transformer_model_enhanced_config.json"
                if os.path.exists(config_path):
                    import json
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    # Get model parameters
                    filters = config.get("filters", [16, 32, 64, 64]) 
                    kernel_size = config.get("kernel_size", 3)
                    transformer_dim = config.get("transformer_dim", 192)
                    nhead = config.get("nhead", 6)
                    num_layers = config.get("num_layers", 3)
                    
                    # Create model with correct parameters
                    from deepcfd.models.TransformerUNetEx import TransformerUNetEx
                    model = TransformerUNetEx(3, 3, 
                                         filters=filters, 
                                         kernel_size=kernel_size,
                                         transformer_dim=transformer_dim,
                                         nhead=nhead,
                                         num_layers=num_layers)
                else:
                    # Fallback to default instantiation
                    state_dict = torch.load(selected_model_path, map_location=torch.device('cpu'))
                    
                    # Use defaults
                    filters = [8, 16, 32, 32]
                    kernel_size = 5
                    transformer_dim = 128
                    nhead = 4
                    num_layers = 2
                    
                    # Create model with default parameters
                    from deepcfd.models.TransformerUNetEx import TransformerUNetEx
                    model = TransformerUNetEx(3, 3, 
                                         filters=filters, 
                                         kernel_size=kernel_size,
                                         transformer_dim=transformer_dim,
                                         nhead=nhead,
                                         num_layers=num_layers)
            else:
                # Regular model - use defaults or try to extract from state_dict
                state_dict = torch.load(selected_model_path, map_location=torch.device('cpu'))
                
                # Get parameters from state dict or use defaults
                filters = state_dict.get("filters", [8, 16, 32, 32])
                kernel_size = state_dict.get("kernel_size", 5)
                transformer_dim = state_dict.get("transformer_dim", 128)
                nhead = state_dict.get("nhead", 4)
                num_layers = state_dict.get("num_layers", 2)
                
                # Create TransformerUNetEx model
                from deepcfd.models.TransformerUNetEx import TransformerUNetEx
                model = TransformerUNetEx(3, 3, 
                                     filters=filters, 
                                     kernel_size=kernel_size,
                                     transformer_dim=transformer_dim,
                                     nhead=nhead,
                                     num_layers=num_layers)
            
            # Load state dict
            state_dict = torch.load(selected_model_path, map_location=torch.device('cpu'))
            
            # Clean state dict for loading - remove metadata keys
            clean_state_dict = {k: v for k, v in state_dict.items() 
                              if not k.startswith('_') and k not in 
                              ["architecture", "input_shape", "filters", "kernel_size", 
                               "transformer_dim", "nhead", "num_layers"]}
            
            model.load_state_dict(clean_state_dict, strict=False)
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
    elif data_option in ["Default data", "Custom path"] and input_data_path:
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
        
    else:
        if model is None:
            st.info("Please select or upload a model file to continue.")
        if x_data is None:
            st.info("Please select or upload input data to continue.")
        
        # Display example of what to expect
        st.subheader("Expected visualization output")
        cols = st.columns(3)
        for i, channel in enumerate(["Ux", "Uy", "Pressure"]):
            with cols[i]:
                st.text(f"{channel} field")