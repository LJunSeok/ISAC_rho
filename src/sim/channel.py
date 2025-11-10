"""
MIMO channel generation for ISAC.
"""

import torch
import numpy as np
from typing import Tuple, Dict
from .geometry import steering_vector, generate_ula_angles


def generate_comm_channels(
    ranges: torch.Tensor,
    angles: torch.Tensor,
    Nt: int,
    Nr: int,
    wavelength: float,
    num_paths: int = 3
) -> torch.Tensor:
    """
    Generate communication channels with multi-path.
    
    Args:
        ranges: User ranges [B, K]
        angles: User angles [B, K]
        Nt: Number of TX antennas
        Nr: Number of RX antennas
        wavelength: Wavelength in meters
        num_paths: Number of paths (LoS + scatterers)
    
    Returns:
        Channels [B, K, Nr, Nt, 2] (complex)
    """
    B, K = ranges.shape
    antenna_pos_tx = generate_ula_angles(Nt)
    antenna_pos_rx = generate_ula_angles(Nr)
    
    # LoS path
    a_tx_los = steering_vector(angles, antenna_pos_tx)  # [B, K, Nt, 2]
    a_rx_los = steering_vector(angles, antenna_pos_rx)  # [B, K, Nr, 2]
    
    # Path loss (free space)
    path_loss = wavelength / (4 * np.pi * ranges)  # [B, K]
    
    # LoS channel: a_rx @ a_tx^H
    from ..utils.complex_ops import complex_conj
    H_los = torch.einsum('bknc,bkmc->bknmc', a_rx_los, complex_conj(a_tx_los))  # [B, K, Nr, Nt, 2]
    H_los = H_los * path_loss[:, :, None, None, None]
    
    # Add scattered paths (simplified)
    H_scatter = torch.randn_like(H_los) * 0.1 * path_loss[:, :, None, None, None]
    
    H = H_los + H_scatter
    
    # Normalize - FIX: add unsqueeze for proper broadcasting
    H_power = torch.sqrt(torch.sum(H[..., 0]**2 + H[..., 1]**2, dim=(-3, -2), keepdim=True))  # [B, K, 1, 1]
    H = H / (H_power.unsqueeze(-1) + 1e-12)  # Now [B, K, 1, 1, 1] broadcasts correctly
    
    return H


def generate_sensing_channels(
    ranges: torch.Tensor,
    angles: torch.Tensor,
    velocities: torch.Tensor,
    Nt: int,
    Nr: int,
    wavelength: float,
    carrier_freq: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate sensing channels (radar-style).
    
    Args:
        ranges: Target ranges [B, M]
        angles: Target angles [B, M]
        velocities: Target velocities [B, M]
        Nt: Number of TX antennas
        Nr: Number of RX antennas
        wavelength: Wavelength
        carrier_freq: Carrier frequency
    
    Returns:
        (Channels [B, M, Nr, Nt, 2], Doppler [B, M])
    """
    from .geometry import compute_doppler
    from ..utils.complex_ops import complex_conj
    
    B, M = ranges.shape
    antenna_pos_tx = generate_ula_angles(Nt)
    antenna_pos_rx = generate_ula_angles(Nr)
    
    # Steering vectors
    a_tx = steering_vector(angles, antenna_pos_tx)  # [B, M, Nt, 2]
    a_rx = steering_vector(angles, antenna_pos_rx)  # [B, M, Nr, 2]
    
    # Path loss (two-way for radar)
    path_loss = (wavelength / (4 * np.pi * ranges))**2  # [B, M]
    
    # Doppler
    doppler = compute_doppler(velocities, carrier_freq)  # [B, M]
    
    # Radar cross-section (RCS) - simplified constant
    rcs = torch.ones_like(ranges) * 1.0
    
    # Channel: sqrt(RCS) * a_rx @ a_tx^H
    H = torch.einsum('bmnc,bmkc->bmnkc', a_rx, complex_conj(a_tx))  # [B, M, Nr, Nt, 2]
    H = H * torch.sqrt(rcs * path_loss)[:, :, None, None, None]
    
    return H, doppler


def add_temporal_dynamics(
    angles: torch.Tensor,
    velocities: torch.Tensor,
    T: float,
    carrier_freq: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Add temporal dynamics to angles (AoD change over slot T).
    
    Args:
        angles: Current angles [B, K/M]
        velocities: Velocities [B, K/M]
        T: Slot duration
        carrier_freq: Carrier frequency
    
    Returns:
        (new_angles [B, K/M], aod_change [B, K/M])
    """
    # Simplified: small random walk + velocity-dependent drift
    drift = velocities * T * 1e-4  # Small drift proportional to velocity
    noise = torch.randn_like(angles) * 0.01  # Small random walk
    
    new_angles = angles + drift + noise
    new_angles = torch.clamp(new_angles, -np.pi/2, np.pi/2)
    
    aod_change = torch.abs(new_angles - angles)
    
    return new_angles, aod_change