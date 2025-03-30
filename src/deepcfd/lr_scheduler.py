import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

class TransformerLRScheduler(_LRScheduler):
    """
    Learning rate scheduler with warmup phase specifically designed for transformer models.
    Implements a warmup period followed by cosine decay.
    """
    def __init__(self, optimizer, warmup_steps, max_steps, min_lr=1e-6, warmup_init_lr=1e-4, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.warmup_init_lr = warmup_init_lr
        super(TransformerLRScheduler, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        step = self.last_epoch + 1
        
        # Warmup phase
        if step <= self.warmup_steps:
            lr_factor = float(step) / float(max(1, self.warmup_steps))
            return [self.warmup_init_lr + lr_factor * (base_lr - self.warmup_init_lr) 
                    for base_lr in self.base_lrs]
        
        # Cosine decay phase
        else:
            progress = float(step - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            
            return [self.min_lr + cosine_decay * (base_lr - self.min_lr) 
                    for base_lr in self.base_lrs]