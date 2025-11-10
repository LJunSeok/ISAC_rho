"""
Geometric configuration for ISAC system.
"""

import torch
import numpy as np
from typing import Tuple


def generate_ula_angles(Nt: int, spacing: float = 0.5) -> torch.Tensor:
    """
    Generate ULA antenna positions.
    
    Args:
        Nt: Number of antennas
        spacing: Spacing in wavelengths
    
    Returns:
        Antenna positions [Nt] in wavelengths
    """
    return torch.arange(Nt, dtype=torch.float32) * spacing


def generate_user_positions(
    K: int,
    range_bounds: Tuple[float, float],
    batch_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate random user positions.
    
    Args:
        K: Number of users
        range_bounds: (min_range, max_range) in meters
        batch_size: Batch size
    
    Returns:
        (ranges [B, K], angles [B, K])
    """
    min_r, max_r = range_bounds
    
    ranges = torch.rand(batch_size, K) * (max_r - min_r) + min_r
    angles = (torch.rand(batch_size, K) - 0.5) * np.pi  # [-π/2, π/2]
    
    return ranges, angles


def generate_target_positions(
    M: int,
    range_bounds: Tuple[float, float],
    batch_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate random target positions.
    
    Args:
        M: Number of targets
        range_bounds: (min_range, max_range) in meters
        batch_size: Batch size
    
    Returns:
        (ranges [B, M], angles [B, M])
    """
    min_r, max_r = range_bounds
    
    ranges = torch.rand(batch_size, M) * (max_r - min_r) + min_r
    angles = (torch.rand(batch_size, M) - 0.5) * np.pi
    
    return ranges, angles


def generate_target_velocities(
    M: int,
    batch_size: int,
    max_velocity: float = 30.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate target velocities (radial).
    
    Args:
        M: Number of targets
        batch_size: Batch size
        max_velocity: Maximum velocity in m/s
    
    Returns:
        (velocities [B, M], Doppler shifts [B, M])
    """
    velocities = (torch.rand(batch_size, M) - 0.5) * 2 * max_velocity
    return velocities


def compute_doppler(velocity: torch.Tensor, carrier_freq: float) -> torch.Tensor:
    """
    Compute Doppler shift from radial velocity.
    
    Args:
        velocity: Radial velocity [B, M] in m/s
        carrier_freq: Carrier frequency in Hz
    
    Returns:
        Doppler shift [B, M] in Hz
    """
    c = 3e8  # Speed of light
    return 2 * velocity * carrier_freq / c


def steering_vector(angles: torch.Tensor, antenna_pos: torch.Tensor) -> torch.Tensor:
    """
    Compute ULA steering vectors.
    
    Args:
        angles: Angles [B, K] in radians
        antenna_pos: Antenna positions [Nt] in wavelengths
    
    Returns:
        Steering vectors [B, K, Nt, 2] (complex)
    """
    # Phase: 2π * d * sin(θ)
    phase = 2 * np.pi * antenna_pos[None, None, :] * torch.sin(angles[:, :, None])
    
    # Complex exponential
    real = torch.cos(phase)
    imag = torch.sin(phase)
    
    return torch.stack([real, imag], dim=-1)  # [B, K, Nt, 2]