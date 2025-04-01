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

# Device selection
device_options = ["cpu"]
if torch.cuda.is_available():
    device_options.append("cuda")
if hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device_options.append("mps")

device = st.sidebar.selectbox("Device", device_options)


# Function to load model
def load_model(model_bytes, architecture):
    try:
        # Load state dict
        state_dict = torch.load(io.BytesIO(model_bytes), map_location=torch.device('cpu'))
        
        # Get model parameters from state dict
        filters = state_dict.get("filters", [8, 16, 32, 32])
        kernel_size = state_dict.get("kernel_size", 5)
        
        # Import correct model architecture
        if architecture == "UNetEx":
            from deepcfd.models.UNetEx import UNetEx
            model = UNetEx(3, 3, filters=filters, kernel_size=kernel_size)
        elif architecture == "TransformerUNetEx":
            from deepcfd.models.TransformerUNetEx import TransformerUNetEx
            model = TransformerUNetEx(3, 3, filters=filters, kernel_size=kernel_size)
        else:  # AutoEncoder
            from deepcfd.models.AutoEncoder import AutoEncoder
            model = AutoEncoder(3, 3, filters=filters, kernel_size=kernel_size)
        
        # Remove architecture and dimension keys for loading
        keys_to_remove = ["architecture", "input_shape", "filters", "kernel_size"]
        for key in keys_to_remove:
            if key in state_dict:
                del state_dict[key]
        
        # Load the state dict into model
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


# Function to load data
def load_data(data_bytes):
    try:
        data = pickle.load(io.BytesIO(data_bytes))
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
if model_file and data_file:
    # Load model
    model_bytes = model_file.getvalue()
    model = load_model(model_bytes, architecture)
    
    # Load data
    data_bytes = data_file.getvalue()
    data = load_data(data_bytes)
    
    if model and data is not None:
        # Check if data is a tuple/list of (x, y) or just x
        if isinstance(data, (tuple, list)) and len(data) == 2:
            x_data, y_data = data
            has_ground_truth = True
        else:
            x_data = data
            y_data = None
            has_ground_truth = False
        
        # Convert to tensor if not already
        if not isinstance(x_data, torch.Tensor):
            x_data = torch.FloatTensor(x_data)
        
        if has_ground_truth and not isinstance(y_data, torch.Tensor):
            y_data = torch.FloatTensor(y_data)
        
        # Show data information
        st.subheader("Data Information")
        st.write(f"Input shape: {x_data.shape}")
        if has_ground_truth:
            st.write(f"Ground truth shape: {y_data.shape}")
            st.write(f"Number of samples: {x_data.shape[0]}")
        
        # Sample selection
        if x_data.shape[0] > 1:
            sample_idx = st.slider("Select sample index", 0, x_data.shape[0]-1, 0)
        else:
            sample_idx = 0
            
        # Move model to selected device
        model = model.to(device)
        
        # Run inference
        with st.spinner("Running inference..."):
            with torch.no_grad():
                # Process only the selected sample
                input_tensor = x_data[sample_idx:sample_idx+1].to(device)
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
        
else:
    # Display instructions if no files are uploaded
    st.info("Please upload a model file (.pt) and a data file (.pkl) to continue.")
    
    # Display example images
    cols = st.columns(2)
    with cols[0]:
        st.subheader("Sample Velocity Field (Ux)")
        placeholder_img = Image.new('RGB', (300, 300), color='white')
        st.image(placeholder_img, caption="Upload data to see prediction")
    
    with cols[1]:
        st.subheader("Sample Pressure Field (p)")
        placeholder_img = Image.new('RGB', (300, 300), color='white')
        st.image(placeholder_img, caption="Upload data to see prediction")