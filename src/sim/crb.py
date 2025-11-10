"""
Cramér-Rao Bound computation for sensing performance.
"""

import torch
import numpy as np
from typing import Dict


def compute_fisher_information(
    H: torch.Tensor,
    F: torch.Tensor,
    rho: torch.Tensor,
    T: float,
    noise_power: float,
    wavelength: float
) -> torch.Tensor:
    """
    Compute Fisher Information Matrix for target parameters.
    
    Simplified FIM for angle-of-arrival estimation.
    
    Args:
        H: Sensing channels [B, M, Nr, Nt, 2]
        F: Sensing beamformer [B, Nt, 1, 2]
        rho: Time fraction [B]
        T: Slot duration
        noise_power: Noise power
        wavelength: Wavelength
    
    Returns:
        FIM [B, M] (simplified scalar per target)
    """
    from ..utils.complex_ops import complex_matmul, complex_abs
    
    B, M, Nr, Nt, _ = H.shape
    
    # Effective sensing time
    T_sense = rho * T  # [B]
    
    # Compute signal power for all targets at once
    # Expand F: [B, Nt, 1, 2] -> [B, 1, Nt, 1, 2] -> [B, M, Nt, 1, 2]
    F_expanded = F.unsqueeze(1)  # [B, 1, Nt, 1, 2]
    F_expanded = F_expanded.expand(B, M, Nt, 1, 2)  # [B, M, Nt, 1, 2]
    
    # Reshape for batched matmul: combine B and M dimensions
    H_flat = H.reshape(B * M, Nr, Nt, 2)  # [B*M, Nr, Nt, 2]
    F_flat = F_expanded.reshape(B * M, Nt, 1, 2)  # [B*M, Nt, 1, 2]
    
    # Compute HF
    HF = complex_matmul(H_flat, F_flat)  # [B*M, Nr, 1, 2]
    
    # Compute power: complex_abs gives [B*M, Nr, 1], squeeze to [B*M, Nr]
    power = complex_abs(HF).squeeze(-1)**2  # [B*M, Nr] - squeeze the dimension with size 1
    power = torch.sum(power, dim=-1)  # [B*M] - sum over Nr antennas
    
    # Reshape back to [B, M]
    signal_power = power.reshape(B, M)  # [B, M]
    
    # SNR
    snr = signal_power / (noise_power + 1e-12)  # [B, M]
    
    # Simplified FIM (diagonal, angle-only)
    # FIM ≈ (2π/λ)^2 * SNR * T_sense * effective_aperture
    effective_aperture = Nt  # Simplified
    
    # Expand T_sense from [B] to [B, M]
    T_sense_expanded = T_sense.view(B, 1).expand(B, M)  # [B, M]
    
    fim_diag = (2 * np.pi / wavelength)**2 * snr * T_sense_expanded * effective_aperture
    
    return fim_diag


def compute_crb_trace(
    H: torch.Tensor,
    F: torch.Tensor,
    rho: torch.Tensor,
    target_params: Dict[str, torch.Tensor],
    T: float,
    noise_power: float,
    wavelength: float
) -> torch.Tensor:
    """
    Compute trace of CRB matrix.
    
    Args:
        H: Sensing channels [B, M, Nr, Nt, 2]
        F: Sensing beamformer [B, Nt, 1, 2]
        rho: Time fraction [B]
        target_params: Target parameters dict
        T: Slot duration
        noise_power: Noise power
        wavelength: Wavelength
    
    Returns:
        CRB trace [B] (lower is better)
    """
    # Compute FIM
    fim = compute_fisher_information(H, F, rho, T, noise_power, wavelength)  # [B, M]
    
    # CRB = inv(FIM), so trace(CRB) = sum(1/FIM_diag)
    crb_trace = torch.sum(1.0 / (fim + 1e-12), dim=-1)  # [B]
    
    return crb_trace