"""
Metrics for ISAC performance evaluation.
"""

import torch
from typing import Dict


def compute_sinr(
    H: torch.Tensor,
    F: torch.Tensor,
    W: torch.Tensor,
    noise_power: float,
    tx_power: float
) -> torch.Tensor:
    """
    Compute SINR for each user.
    
    Args:
        H: Channel [..., K, Nr, Nt, 2]
        F: Precoders [..., Nt, K, 2]
        W: Combiners [..., Nr, K, 2]
        noise_power: Noise power
        tx_power: Transmit power
    
    Returns:
        SINR [..., K]
    """
    from .complex_ops import complex_matmul, complex_conj, complex_abs
    
    K = H.shape[-4]
    batch_shape = H.shape[:-4]
    
    # Normalize transmit power
    # F: [B, Nt, K, 2] -> power per user: [B, 1, K]
    F_power = torch.sum(F[..., 0]**2 + F[..., 1]**2, dim=-3, keepdim=True)  # [B, 1, K]
    # Need [B, 1, K, 1] to broadcast with [B, Nt, K, 2]
    F_norm = F * torch.sqrt(tx_power / (F_power.unsqueeze(-1) + 1e-12))
    
    sinrs = []
    for k in range(K):
        # Signal: W_k^H @ H_k @ F_k
        H_k = H[..., k, :, :, :]  # [..., Nr, Nt, 2]
        F_k = F_norm[..., :, k:k+1, :]  # [..., Nt, 1, 2]
        W_k = W[..., :, k:k+1, :]  # [..., Nr, 1, 2]
        
        signal = complex_matmul(H_k, F_k)  # [..., Nr, 1, 2]
        signal = complex_matmul(complex_conj(W_k).transpose(-2, -3), signal)  # [..., 1, 1, 2]
        signal_power = complex_abs(signal.squeeze(-2)).squeeze(-1)**2  # [...]
        
        # Interference from other users
        interference = torch.zeros_like(signal_power)  # Initialize as tensor with correct shape
        for j in range(K):
            if j != k:
                F_j = F_norm[..., :, j:j+1, :]
                intrf = complex_matmul(H_k, F_j)
                intrf = complex_matmul(complex_conj(W_k).transpose(-2, -3), intrf)
                interference = interference + complex_abs(intrf.squeeze(-2)).squeeze(-1)**2
        
        # Noise power scaled by combiner norm
        # W_k shape: [B, Nr, 1, 2]
        # W_k[..., 0]**2 + W_k[..., 1]**2 gives [B, Nr, 1]
        # Sum over Nr dimension (dim=1 or dim=-2), not batch dimension
        W_k_power = torch.sum(W_k[..., 0]**2 + W_k[..., 1]**2, dim=1)  # [B, 1]
        noise = noise_power * W_k_power.squeeze(-1)  # [B]
        
        # SINR
        sinr_k = signal_power / (interference + noise + 1e-12)
        sinrs.append(sinr_k)
    
    return torch.stack(sinrs, dim=-1)


def compute_sum_rate(sinr: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """
    Compute time-scaled sum rate.
    
    Args:
        sinr: SINR values [..., K]
        rho: Sensing time fraction [...]
    
    Returns:
        Sum rate [...] in bits/s/Hz scaled by (1-rho)
    """
    rate_per_user = torch.log2(1 + sinr)  # [..., K]
    sum_rate = torch.sum(rate_per_user, dim=-1)  # [...]
    return (1 - rho) * sum_rate


def compute_composite_metric(
    sum_rate: torch.Tensor,
    crb_trace: torch.Tensor,
    lambda_trade: float
) -> torch.Tensor:
    """
    Compute composite metric: λ·(-R) + (1-λ)·CRB
    
    Args:
        sum_rate: Sum rate [...]
        crb_trace: CRB trace [...]
        lambda_trade: Trade-off parameter
    
    Returns:
        Composite metric [...]
    """
    return lambda_trade * (-sum_rate) + (1 - lambda_trade) * crb_trace

def compute_efficiency_score(
    sum_rate: torch.Tensor,
    crb_trace: torch.Tensor,
    epsilon: float = 1e-6
) -> torch.Tensor:
    """
    Compute efficiency score: Rate × (1/CRB)
    
    Higher is better. Captures joint improvement in both metrics.
    
    Args:
        sum_rate: Sum rate [...]
        crb_trace: CRB trace [...]
        epsilon: Small constant to avoid division by zero
    
    Returns:
        Efficiency score [...]
    """
    return sum_rate / (crb_trace + epsilon)

def batch_metrics(
    H_comm: torch.Tensor,
    H_sense: torch.Tensor,
    F_S: torch.Tensor,
    F_C: torch.Tensor,
    W_C: torch.Tensor,
    rho: torch.Tensor,
    target_params: Dict[str, torch.Tensor],
    sys_config,
    lambda_trade: float
) -> Dict[str, float]:
    """
    Compute all metrics for a batch.
    
    Args:
        H_comm: Communication channels [B, K, Nr, Nt, 2]
        H_sense: Sensing channels [B, M, Nr, Nt, 2]
        F_S: Sensing beamformers [B, Nt, 1, 2]
        F_C: Communication beamformers [B, Nt, K, 2]
        W_C: Communication combiners [B, Nr, K, 2]
        rho: Time split [B]
        target_params: Target parameters (ranges, angles, Doppler)
        sys_config: System configuration
        lambda_trade: Trade-off parameter
    
    Returns:
        Dictionary of metrics
    """
    from ..sim.crb import compute_crb_trace
    
    # SINR and sum rate
    sinr = compute_sinr(H_comm, F_C, W_C, sys_config.noise_power, sys_config.tx_power)
    sum_rate = compute_sum_rate(sinr, rho)
    
    # CRB trace
    crb_trace = compute_crb_trace(
        H_sense, F_S, rho, target_params,
        sys_config.T, sys_config.noise_power, sys_config.wavelength
    )

    # Composite (legacy)
    composite = compute_composite_metric(sum_rate, crb_trace, lambda_trade)
    
    # Efficiency score
    efficiency = compute_efficiency_score(sum_rate, crb_trace)
    
    return {
        'sum_rate': sum_rate.mean().item(),
        'crb_trace': crb_trace.mean().item(),
        'composite': composite.mean().item(),
        'efficiency': efficiency.mean().item(),
        'rho_mean': rho.mean().item(),
        'rho_std': rho.std().item(),
        'sinr_min': sinr.min().item(),
        'sinr_mean': sinr.mean().item(),
    }