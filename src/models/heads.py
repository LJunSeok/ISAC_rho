"""
Output heads for F_S, F_C, W_C, and ρ.
"""

import torch
import torch.nn as nn
from ..utils.complex_ops import complex_normalize, real_to_complex


class SensingBeamHead(nn.Module):
    """Head for sensing beamformer F_S."""
    
    def __init__(self, latent_dim: int, Nt: int):
        super().__init__()
        self.Nt = Nt
        
        self.fc_real = nn.Linear(latent_dim, Nt)
        self.fc_imag = nn.Linear(latent_dim, Nt)
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: [B, latent_dim]
        
        Returns:
            F_S: [B, Nt, 1, 2] (complex, unit-norm)
        """
        f_real = self.fc_real(latent)  # [B, Nt]
        f_imag = self.fc_imag(latent)  # [B, Nt]
        
        f = real_to_complex(f_real, f_imag)  # [B, Nt, 2]
        f = complex_normalize(f, dim=-2)  # [B, Nt, 2]
        
        return f.unsqueeze(-2)  # [B, Nt, 1, 2]


class CommBeamHead(nn.Module):
    """Head for communication beamformers F_C."""
    
    def __init__(self, latent_dim: int, Nt: int, K: int):
        super().__init__()
        self.Nt = Nt
        self.K = K
        
        self.fc_real = nn.Linear(latent_dim, Nt * K)
        self.fc_imag = nn.Linear(latent_dim, Nt * K)
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: [B, latent_dim]
        
        Returns:
            F_C: [B, Nt, K, 2] (complex, per-user unit-norm)
        """
        f_real = self.fc_real(latent).reshape(-1, self.Nt, self.K)  # [B, Nt, K]
        f_imag = self.fc_imag(latent).reshape(-1, self.Nt, self.K)  # [B, Nt, K]
        
        f = real_to_complex(f_real, f_imag)  # [B, Nt, K, 2]
        
        # Normalize per user - avoid in-place operations
        f_list = []
        for k in range(self.K):
            f_k = f[:, :, k:k+1, :]  # [B, Nt, 1, 2]
            f_k_norm = complex_normalize(f_k, dim=-3)  # [B, Nt, 1, 2]
            f_list.append(f_k_norm)
        
        f_normalized = torch.cat(f_list, dim=2)  # [B, Nt, K, 2]
        
        return f_normalized


class CommCombinerHead(nn.Module):
    """Head for communication combiners W_C."""
    
    def __init__(self, latent_dim: int, Nr: int, K: int):
        super().__init__()
        self.Nr = Nr
        self.K = K
        
        self.fc_real = nn.Linear(latent_dim, Nr * K)
        self.fc_imag = nn.Linear(latent_dim, Nr * K)
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: [B, latent_dim]
        
        Returns:
            W_C: [B, Nr, K, 2] (complex, per-user unit-norm)
        """
        w_real = self.fc_real(latent).reshape(-1, self.Nr, self.K)  # [B, Nr, K]
        w_imag = self.fc_imag(latent).reshape(-1, self.Nr, self.K)  # [B, Nr, K]
        
        w = real_to_complex(w_real, w_imag)  # [B, Nr, K, 2]
        
        # Normalize per user - avoid in-place operations
        w_list = []
        for k in range(self.K):
            w_k = w[:, :, k:k+1, :]  # [B, Nr, 1, 2]
            w_k_norm = complex_normalize(w_k, dim=-3)  # [B, Nr, 1, 2]
            w_list.append(w_k_norm)
        
        w_normalized = torch.cat(w_list, dim=2)  # [B, Nr, K, 2]
        
        return w_normalized


class RhoHead(nn.Module):
    """Head for time-split ρ with bounded sigmoid."""
    
    def __init__(self, latent_dim: int, rho_min: float = 0.0, rho_max: float = 0.4):
        super().__init__()
        self.rho_min = rho_min
        self.rho_max = rho_max
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: [B, latent_dim]
        
        Returns:
            rho: [B] in [rho_min, rho_max]
        """
        logit = self.fc(latent).squeeze(-1)  # [B]
        rho = self.rho_min + (self.rho_max - self.rho_min) * torch.sigmoid(logit)
        return rho


class ISACModel(nn.Module):
    """Complete ISAC model with all heads."""
    
    def __init__(self, backbone: nn.Module, sys_config, frozen_wc: bool = False):
        super().__init__()
        self.backbone = backbone
        self.frozen_wc = frozen_wc
        
        latent_dim = backbone.latent_dim
        
        self.head_fs = SensingBeamHead(latent_dim, sys_config.Nt)
        self.head_fc = CommBeamHead(latent_dim, sys_config.Nt, sys_config.K)
        self.head_wc = CommCombinerHead(latent_dim, sys_config.Nr, sys_config.K)
        self.head_rho = RhoHead(latent_dim, sys_config.rho_min, sys_config.rho_max)
        
        # Freeze W_C if needed
        if self.frozen_wc:
            for param in self.head_wc.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        sensing_echoes: torch.Tensor,
        comm_echoes: torch.Tensor,
        rho_prev: torch.Tensor
    ) -> dict:
        """
        Args:
            sensing_echoes: [B, tau, D_sense]
            comm_echoes: [B, tau, D_comm]
            rho_prev: [B, tau]
        
        Returns:
            Dict with F_S, F_C, W_C, rho
        """
        latent = self.backbone(sensing_echoes, comm_echoes, rho_prev)
        
        F_S = self.head_fs(latent)
        F_C = self.head_fc(latent)
        W_C = self.head_wc(latent)
        rho = self.head_rho(latent)
        
        return {
            'F_S': F_S,
            'F_C': F_C,
            'W_C': W_C,
            'rho': rho,
        }