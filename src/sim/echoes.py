"""
Generate dual-stream echo histories for sensing and communication.
"""

import torch
from typing import Tuple, Dict
from .channel import generate_comm_channels, generate_sensing_channels, add_temporal_dynamics


def generate_echo_history(
    sys_config,
    batch_size: int,
    tau: int,
    rho_history: torch.Tensor = None
) -> Dict[str, torch.Tensor]:
    """
    Generate dual-stream echo history (sensing + comm) for τ slots.
    
    Args:
        sys_config: System configuration
        batch_size: Batch size
        tau: History window size
        rho_history: Previous rho values [B, tau] (None = initialize)
    
    Returns:
        Dictionary with keys:
            - sensing_echoes: [B, tau, Nr*Nt*2] (flattened complex)
            - comm_echoes: [B, tau, Nr*Nt*K*2] (flattened complex)
            - rho_prev: [B, tau] previous rho values
            - doppler: [B, M] Doppler shifts
            - aod_change: [B, M] AoD changes
            - H_comm: [B, K, Nr, Nt, 2] current comm channels
            - H_sense: [B, M, Nr, Nt, 2] current sensing channels
            - target_params: Dict with target parameters
    """
    from .geometry import generate_user_positions, generate_target_positions, generate_target_velocities
    
    # Initialize rho_history if not provided
    if rho_history is None:
        rho_history = torch.ones(batch_size, tau) * 0.2  # Default initialization
    
    # Generate geometry
    comm_ranges, comm_angles = generate_user_positions(
        sys_config.K, sys_config.comm_range, batch_size
    )
    target_ranges, target_angles = generate_target_positions(
        sys_config.M, sys_config.target_range, batch_size
    )
    target_velocities = generate_target_velocities(sys_config.M, batch_size)
    
    # Temporal dynamics
    target_angles_new, aod_change = add_temporal_dynamics(
        target_angles, target_velocities, sys_config.T, sys_config.carrier_freq
    )
    
    # Generate channels
    H_comm = generate_comm_channels(
        comm_ranges, comm_angles,
        sys_config.Nt, sys_config.Nr, sys_config.wavelength
    )
    
    H_sense, doppler = generate_sensing_channels(
        target_ranges, target_angles_new, target_velocities,
        sys_config.Nt, sys_config.Nr, sys_config.wavelength, sys_config.carrier_freq
    )
    
    # Generate echo histories (simplified: use noisy channel observations)
    # In practice, these would be actual received signals
    sensing_echoes = []
    comm_echoes = []
    
    for t in range(tau):
        # Sensing echo: flatten H_sense + noise
        sense_echo = H_sense.reshape(batch_size, -1, 2)  # [B, M*Nr*Nt, 2]
        sense_echo = sense_echo + torch.randn_like(sense_echo) * 0.1
        sense_echo_flat = sense_echo.reshape(batch_size, -1)  # [B, M*Nr*Nt*2]
        sensing_echoes.append(sense_echo_flat)
        
        # Comm echo: flatten H_comm + noise
        comm_echo = H_comm.reshape(batch_size, -1, 2)  # [B, K*Nr*Nt, 2]
        comm_echo = comm_echo + torch.randn_like(comm_echo) * 0.1
        comm_echo_flat = comm_echo.reshape(batch_size, -1)  # [B, K*Nr*Nt*2]
        comm_echoes.append(comm_echo_flat)
    
    sensing_echoes = torch.stack(sensing_echoes, dim=1)  # [B, tau, M*Nr*Nt*2]
    comm_echoes = torch.stack(comm_echoes, dim=1)  # [B, tau, K*Nr*Nt*2]
    
    target_params = {
        'ranges': target_ranges,
        'angles': target_angles_new,
        'velocities': target_velocities,
        'doppler': doppler,
    }
    
    return {
        'sensing_echoes': sensing_echoes,
        'comm_echoes': comm_echoes,
        'rho_prev': rho_history,
        'doppler': doppler,
        'aod_change': aod_change,
        'H_comm': H_comm,
        'H_sense': H_sense,
        'target_params': target_params,
    }


def apply_duration_normalization(
    sensing_echoes: torch.Tensor,
    comm_echoes: torch.Tensor,
    rho_prev: torch.Tensor,
    eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply duration-aware normalization to echo inputs.
    
    Args:
        sensing_echoes: [B, tau, D_sense]
        comm_echoes: [B, tau, D_comm]
        rho_prev: [B, tau] previous rho values
        eps: Small constant for stability
    
    Returns:
        (normalized_sensing, normalized_comm)
    """
    # Scale sensing by 1/sqrt(rho + eps)
    sense_scale = 1.0 / torch.sqrt(rho_prev + eps)  # [B, tau]
    sensing_normalized = sensing_echoes * sense_scale.unsqueeze(-1)
    
    # Scale comm by 1/sqrt(1 - rho + eps)
    comm_scale = 1.0 / torch.sqrt(1.0 - rho_prev + eps)  # [B, tau]
    comm_normalized = comm_echoes * comm_scale.unsqueeze(-1)
    
    return sensing_normalized, comm_normalized