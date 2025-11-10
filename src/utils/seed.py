"""
Reproducibility utilities for deterministic training.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Ensure deterministic behavior on CPU
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Set number of threads for reproducibility
    torch.set_num_threads(1)


def worker_init_fn(worker_id: int):
    """
    DataLoader worker initialization for reproducibility.
    
    Args:
        worker_id: Worker process ID
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)