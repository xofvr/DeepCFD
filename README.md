# TransformerDeepCFD

This project extends the original DeepCFD architecture by adding a transformer module to improve flow field prediction capabilities. The transformer's self-attention mechanism helps capture global relationships in the flow data, complementing the CNN's local feature extraction abilities.

## Key Features

* Combines CNN spatial hierarchy with transformer global attention
* Maintains compatibility with existing DeepCFD datasets
* Achieves improved accuracy while preserving computational efficiency
* Supports visualization of predictions vs ground truth

## Installation

The module can be installed with:

```
pip install git+https://github.com/yourusername/TransformerDeepCFD.git@master
```

Or clone and install locally:

```
git clone https://github.com/yourusername/TransformerDeepCFD.git
cd TransformerDeepCFD
pip install -e .
```

## Requirements

* torch
* torchvision
* matplotlib
* CUDA or MPS acceleration recommended but not required

## Usage

To train a TransformerDeepCFD model:

```bash
python -m deepcfd \
    --net TransformerUNetEx \
    --device mps \
    --model-input data/dataX.pkl \
    --model-output data/dataY.pkl \
    --output model_transformer.pt \
    --kernel-size 5 \
    --filters 8,16,32,32 \
    --learning-rate 0.001 \
    --epochs 1000 \
    --batch-size 32 \
    --patience 300 \
    --visualize > training_log.txt
```

### Parameters:

* `--net`: Network architecture (TransformerUNetEx for the transformer version)
* `--device`: Computing device (cpu, cuda, mps for Mac M1/M2)
* `--model-input`: Path to input dataset (SDF and flow region channels)
* `--model-output`: Path to output dataset (velocity and pressure fields)
* `--output`: Path to save the trained model
* `--kernel-size`: Kernel size for convolutional layers
* `--filters`: Number of filters in each level (comma-separated)
* `--learning-rate`: Initial learning rate
* `--epochs`: Number of training epochs
* `--batch-size`: Batch size for training
* `--patience`: Patience for early stopping
* `--visualize`: Flag to enable visualization of results

## Dataset Format

The dataset follows the same format as the original DeepCFD:

* Input data (dataX.pkl): Contains SDF and flow region information
* Output data (dataY.pkl): Contains velocity (Ux, Uy) and pressure (p) fields
