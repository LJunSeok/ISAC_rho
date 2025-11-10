"""
I/O utilities for saving and loading models, metrics, and configurations.
"""

import torch
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import dataclasses


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: Dict[str, list],
    save_path: Path
):
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        history: Training history
        save_path: Path to save checkpoint
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'history': history,
    }
    
    torch.save(checkpoint, save_path)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    load_path: Path
) -> tuple:
    """
    Load training checkpoint.
    
    Args:
        model: Model to load into
        optimizer: Optimizer to load into
        load_path: Path to checkpoint
    
    Returns:
        (epoch, history)
    """
    # Use weights_only=False since we're loading optimizer state and history
    # This is safe since we're loading our own checkpoints
    checkpoint = torch.load(load_path, map_location='cpu', weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint.get('history', {})


def save_metrics_csv(metrics: pd.DataFrame, save_path: Path):
    """Save metrics DataFrame to CSV."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(save_path, index=False)


def load_metrics_csv(load_path: Path) -> pd.DataFrame:
    """Load metrics DataFrame from CSV."""
    return pd.read_csv(load_path)


def save_config(config: Any, save_path: Path):
    """
    Save dataclass configuration to JSON.
    
    Args:
        config: Dataclass instance
        save_path: Path to save JSON
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = dataclasses.asdict(config)
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)


def load_config(config_class: type, load_path: Path):
    """
    Load dataclass configuration from JSON.
    
    Args:
        config_class: Dataclass type
        load_path: Path to JSON
    
    Returns:
        Dataclass instance
    """
    with open(load_path, 'r') as f:
        config_dict = json.load(f)
    
    # Filter out fields with init=False
    init_fields = set()
    for field in dataclasses.fields(config_class):
        if field.init:
            init_fields.add(field.name)
    
    # Keep only fields that should be passed to __init__
    filtered_dict = {k: v for k, v in config_dict.items() if k in init_fields}
    
    return config_class(**filtered_dict)


def setup_results_dir(base_path: Path = Path("results")) -> Dict[str, Path]:
    """
    Create results directory structure.
    
    Returns:
        Dictionary of paths
    """
    paths = {
        'base': base_path,
        'checkpoints': base_path / 'checkpoints',
        'plots': base_path / 'plots',
        'metrics': base_path,
    }
    
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    
    return paths

def get_checkpoint_name(
    alpha_chatter: float = None,
    lambda_trade: float = None,
    seed: int = None,
    use_distill: bool = False,
    frozen_wc: bool = False,
    pretrain_rho: float = None,
    default: bool = False
) -> str:
    """
    Generate unique checkpoint name based on training configuration.
    
    Args:
        alpha_chatter: Anti-chatter weight
        lambda_trade: Trade-off parameter
        seed: Random seed
        use_distill: Whether distillation was used
        frozen_wc: Whether W_C was frozen
        pretrain_rho: Pretraining rho value
        default: If True, return default name for backward compatibility
    
    Returns:
        Checkpoint filename (e.g., 'model_a0.05_l0.5_s42.pt')
    """
    if default:
        return 'dynamic_rho_model.pt'
    
    # Build name components
    parts = ['model']
    
    if alpha_chatter is not None:
        parts.append(f'a{alpha_chatter:.3f}'.replace('.', 'p'))
    
    if lambda_trade is not None:
        parts.append(f'l{lambda_trade:.2f}'.replace('.', 'p'))
    
    if seed is not None:
        parts.append(f's{seed}')
    
    if use_distill:
        parts.append('distill')
    
    if frozen_wc:
        parts.append('frozenwc')
    
    if pretrain_rho is not None:
        parts.append(f'pretrain{pretrain_rho:.2f}'.replace('.', 'p'))
    
    return '_'.join(parts) + '.pt'


def get_metrics_name(checkpoint_name: str, metric_type: str = 'test_metrics') -> str:
    """
    Generate metrics filename matching checkpoint name.
    
    Args:
        checkpoint_name: Checkpoint filename (e.g., 'model_a0.05_l0.5_s42.pt')
        metric_type: Type of metrics ('test_metrics' or 'test_rho_analysis')
    
    Returns:
        Metrics filename (e.g., 'test_metrics_a0.05_l0.5_s42.csv')
    """
    # Extract config suffix from checkpoint name
    # model_a0.05_l0.5_s42.pt -> a0.05_l0.5_s42
    if checkpoint_name == 'dynamic_rho_model.pt':
        return f'{metric_type}.csv'
    
    base = checkpoint_name.replace('model_', '').replace('.pt', '')
    return f'{metric_type}_{base}.csv'