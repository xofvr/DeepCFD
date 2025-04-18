import torch
import random
import numpy as np

class FluidDataAugmentation:
    """
    Data augmentation for fluid dynamics data. Applies transformations
    that preserve physical meaning of the flow fields.
    """
    def __init__(self, flip_prob=0.3, rotate_prob=0.3, noise_prob=0.2, noise_level=0.02):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.noise_prob = noise_prob
        self.noise_level = noise_level
        
    def __call__(self, x, y):
        """
        Apply augmentations to input (geometry) and target (flow field) tensors.
        
        Args:
            x: Input tensor with shape [batch, channels, height, width]
            y: Target tensor with shape [batch, channels, height, width]
            
        Returns:
            Augmented versions of x and y
        """
        # Make copies to avoid modifying originals
        x_aug = x.clone()
        y_aug = y.clone()
        
        # Horizontal flip (preserves physical meaning for channel flows)
        if random.random() < self.flip_prob:
            x_aug = torch.flip(x_aug, dims=[-1])
            y_aug = torch.flip(y_aug, dims=[-1])
            # Flip velocity x-component direction
            y_aug[:, 0, :, :] = -y_aug[:, 0, :, :]
        
        # 90-degree rotations for square domains (if applicable)
        if random.random() < self.rotate_prob:
            k = random.choice([1, 2, 3])  # Number of 90-degree rotations
            x_aug = torch.rot90(x_aug, k=k, dims=[-2, -1])
            y_aug = torch.rot90(y_aug, k=k, dims=[-2, -1])
            
            # For each rotation, we need to adjust velocity components
            if k == 1:  # 90 degrees
                # Swap x and y velocity components and flip one
                y_aug_temp = y_aug.clone()
                y_aug[:, 0, :, :] = -y_aug_temp[:, 1, :, :]  # vx = -vy
                y_aug[:, 1, :, :] = y_aug_temp[:, 0, :, :]   # vy = vx
            elif k == 2:  # 180 degrees
                # Flip both velocity components
                y_aug[:, 0, :, :] = -y_aug[:, 0, :, :]  # vx = -vx
                y_aug[:, 1, :, :] = -y_aug[:, 1, :, :]  # vy = -vy
            elif k == 3:  # 270 degrees
                # Swap x and y velocity components and flip one
                y_aug_temp = y_aug.clone()
                y_aug[:, 0, :, :] = y_aug_temp[:, 1, :, :]   # vx = vy
                y_aug[:, 1, :, :] = -y_aug_temp[:, 0, :, :]  # vy = -vx
        
        # Add small random noise to input geometry (SDF values)
        if random.random() < self.noise_prob:
            # Only add noise to the SDF channel, not the flow region channel
            noise = torch.randn_like(x_aug[:, 0:1, :, :]) * self.noise_level
            x_aug[:, 0:1, :, :] = x_aug[:, 0:1, :, :] + noise
            
        return x_aug, y_aug

# Function to create a data loader with augmentation
def create_augmented_dataloader(dataset, batch_size, shuffle=True, augmentation=None):
    """
    Creates a DataLoader with on-the-fly data augmentation.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size for the dataloader
        shuffle: Whether to shuffle the data
        augmentation: Augmentation object to apply
        
    Returns:
        DataLoader with augmentation
    """
    if augmentation is None:
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    class AugmentedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform
            
        def __len__(self):
            return len(self.dataset)
            
        def __getitem__(self, idx):
            x, y = self.dataset[idx]
            # Add batch dimension for augmentation
            x_aug, y_aug = self.transform(x.unsqueeze(0), y.unsqueeze(0))
            # Remove batch dimension
            return x_aug.squeeze(0), y_aug.squeeze(0)
    
    augmented_dataset = AugmentedDataset(dataset, augmentation)
    return torch.utils.data.DataLoader(augmented_dataset, batch_size=batch_size, shuffle=shuffle)