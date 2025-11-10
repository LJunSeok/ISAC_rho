"""
System configuration for ISAC dynamic-rho learning.
Default parameters optimized for CPU execution with small memory footprint.
"""

import dataclasses
from typing import Tuple


@dataclasses.dataclass
class SystemConfig:
    """Physical system parameters."""
    
    # Antenna configuration
    Nt: int = 64              # Number of transmit antennas
    Nr: int = 16              # Number of receive antennas
    K: int = 4                # Number of communication users
    M: int = 16               # Number of sensing targets
    
    # Temporal parameters
    tau: int = 4              # History window size
    T: float = 0.01           # Slot duration in seconds
    carrier_freq: float = 28e9  # 28 GHz mmWave
    
    # Rho bounds (time-split constraints)
    rho_min: float = 0.01      # Minimum sensing time fraction
    rho_max: float = 0.99      # Maximum sensing time fraction
    
    # Channel parameters
    wavelength: float = dataclasses.field(init=False)
    antenna_spacing: float = 0.5  # In wavelengths
    
    # Noise and power
    noise_power: float = 1e-10    # Noise power
    tx_power: float = 1.0         # Transmit power
    
    # Geometry bounds
    comm_range: Tuple[float, float] = (50.0, 150.0)  # meters
    target_range: Tuple[float, float] = (100.0, 500.0)  # meters
    
    def __post_init__(self):
        self.wavelength = 3e8 / self.carrier_freq


@dataclasses.dataclass
class ModelConfig:
    """Neural network model parameters."""
    
    # CNN stem parameters
    cnn_channels: int = 32     # CNN output channels
    cnn_kernel: int = 3        # Kernel size
    
    # LSTM parameters
    lstm_hidden: int = 128     # LSTM hidden size
    lstm_layers: int = 1       # Number of LSTM layers
    
    # Attention parameters
    attn_heads: int = 4        # Number of attention heads
    attn_dim: int = 128        # Attention dimension
    
    # Regularization
    dropout: float = 0.1       # Dropout probability


@dataclasses.dataclass
class TrainingConfig:
    """Training hyperparameters."""
    
    # Optimization
    lr: float = 1e-3           # Learning rate
    weight_decay: float = 1e-5  # L2 regularization
    grad_clip: float = 1.0     # Gradient clipping norm
    
    # Loss weights
    lambda_trade: float = 0.5   # Trade-off: 0=rate only, 1=sensing only
    alpha_chatter: float = 0.05  # Anti-chatter regularization
    l2_logit: float = 1e-4      # L2 on head logits
    
    # CRB scaling
    crb_scale: float = 1.0          # CRB scaling factor (auto-calibrated if auto_calibrate=True)
    auto_calibrate: bool = True     # Auto-calibrate CRB scale on first batch
    target_balance: float = 0.5     # Target: scaled_CRB ≈ target_balance * Rate

    # Training schedule
    epochs: int = 20             # Number of epochs
    batch_size: int = 16        # Batch size (memory-efficient)
    batches_per_epoch: int = 100  # Batches per epoch
    
    # Pretrain/distillation
    pretrain_rho: float = None   # Fixed rho for pretraining (None=skip)
    pretrain_epochs: int = 2     # Pretrain epochs
    use_distill: bool = False    # Use oracle-rho distillation
    distill_epochs: int = 3      # Distillation epochs
    distill_weight: float = 0.1  # Distillation loss weight
    
    # Evaluation
    test_batches: int = 50      # Test set size
    
    # Reproducibility
    seed: int = 42              # Random seed


@dataclasses.dataclass
class AblationConfig:
    """Ablation study configuration."""
    
    no_dual_stream: bool = False   # Collapse to single echo stream
    frozen_wc: bool = False         # Freeze W_C (RX idealization)
    no_anti_chatter: bool = False   # Set alpha=0
    fixed_rho: bool = False         # Turn off rho head
    fixed_rho_value: float = 0.35    # Fixed rho value when fixed_rho=True
    
    # Oracle-rho analysis
    oracle_rho_grid: Tuple[float, ...] = (0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)


def get_default_configs():
    """Get default configuration tuple."""
    return SystemConfig(), ModelConfig(), TrainingConfig(), AblationConfig()