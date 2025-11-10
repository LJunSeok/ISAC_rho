"""
EKF-based baseline for ISAC beamforming.

Uses fixed pilot/tracking overhead instead of learned dynamic-ρ.
"""

import torch
import numpy as np
from typing import Dict


class EKFBaseline:
    """
    EKF-based beamforming baseline.
    
    Assumptions:
    - ρ_pilot = 0.05 (5% pilot overhead)
    - ρ_tracking = 0.03 (3% EKF tracking overhead)
    - ρ_sensing = 0 (no dedicated sensing)
    - Communication time = 1 - 0.05 - 0.03 = 0.92
    """
    
    def __init__(
        self,
        sys_config,
        rho_pilot: float = 0.05,
        rho_tracking: float = 0.03
    ):
        self.sys_config = sys_config
        self.rho_pilot = rho_pilot
        self.rho_tracking = rho_tracking
        self.rho_sensing = 0.0  # No dedicated sensing
        self.rho_comm = 1.0 - rho_pilot - rho_tracking
        
    def compute_beamformers(
        self,
        H_comm: torch.Tensor,
        H_sense: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute beamformers using simplified MRT/MRC.
        
        Args:
            H_comm: [B, K, Nr, Nt, 2]
            H_sense: [B, M, Nr, Nt, 2]
        
        Returns:
            Dict with F_S, F_C, W_C, rho
        """
        from ..utils.complex_ops import complex_conj, complex_normalize
        
        B, K = H_comm.shape[:2]
        M = H_sense.shape[1]
        Nt = self.sys_config.Nt
        Nr = self.sys_config.Nr
        
        # Communication beamformers (MRT): F_C[k] = H_comm[k]^H (conjugate transpose)
        # H_comm: [B, K, Nr, Nt, 2]
        # We want: [B, Nt, K, 2]
        F_C_list = []
        for k in range(K):
            H_k = H_comm[:, k, :, :, :]  # [B, Nr, Nt, 2]
            # MRT: sum over Nr dimension and conjugate
            F_k = torch.sum(complex_conj(H_k), dim=1)  # [B, Nt, 2]
            F_k = complex_normalize(F_k, dim=-2)  # Normalize
            F_C_list.append(F_k.unsqueeze(2))  # [B, Nt, 1, 2]
        
        F_C = torch.cat(F_C_list, dim=2)  # [B, Nt, K, 2]
        
        # Communication combiners (MRC): W_C[k] = H_comm[k] @ F_C[k]
        W_C_list = []
        for k in range(K):
            H_k = H_comm[:, k, :, :, :]  # [B, Nr, Nt, 2]
            F_k = F_C[:, :, k:k+1, :]  # [B, Nt, 1, 2]
            
            # W_k = H_k @ F_k
            from ..utils.complex_ops import complex_matmul
            W_k = complex_matmul(H_k, F_k)  # [B, Nr, 1, 2]
            W_k = complex_normalize(W_k, dim=-3)  # Normalize
            W_C_list.append(W_k)
        
        W_C = torch.cat(W_C_list, dim=2)  # [B, Nr, K, 2]
        
        # Sensing beamformer (use first target, MRT-style)
        H_sense_0 = H_sense[:, 0, :, :, :]  # [B, Nr, Nt, 2]
        F_S = torch.sum(complex_conj(H_sense_0), dim=1)  # [B, Nt, 2]
        F_S = complex_normalize(F_S, dim=-2)
        F_S = F_S.unsqueeze(2)  # [B, Nt, 1, 2]
        
        # Fixed rho (no sensing time, only pilot/tracking overhead)
        rho = torch.full((B,), self.rho_sensing)
        
        return {
            'F_S': F_S,
            'F_C': F_C,
            'W_C': W_C,
            'rho': rho,
            'rho_effective': torch.full((B,), self.rho_comm)  # Effective comm time
        }
    
    def evaluate(
        self,
        H_comm: torch.Tensor,
        H_sense: torch.Tensor,
        target_params: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Evaluate EKF baseline performance.
        
        Returns:
            Dict with sum_rate, crb_trace, rho_mean
        """
        from ..utils.metrics import compute_sinr, batch_metrics
        
        outputs = self.compute_beamformers(H_comm, H_sense)
        
        # Use effective communication time for rate calculation
        rho_eff = outputs['rho_effective']
        
        # Compute metrics
        metrics = batch_metrics(
            H_comm, H_sense,
            outputs['F_S'], outputs['F_C'], outputs['W_C'],
            rho_eff,  # Use effective comm time
            target_params,
            self.sys_config,
            lambda_trade=0.5
        )
        
        # Override rho_mean to show actual sensing time (0)
        metrics['rho_mean'] = self.rho_sensing
        metrics['rho_overhead'] = self.rho_pilot + self.rho_tracking
        
        return metrics