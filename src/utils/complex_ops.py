"""
Complex-valued operations for beamforming.
"""

import torch
import torch.nn as nn


def complex_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Complex matrix multiplication: C = A @ B
    
    Args:
        A: Complex tensor [..., M, K] as [..., M, K, 2] (real, imag)
        B: Complex tensor [..., K, N] as [..., K, N, 2] (real, imag)
    
    Returns:
        C: Complex tensor [..., M, N, 2]
    """
    # A @ B = (Ar + jAi) @ (Br + jBi) = (Ar@Br - Ai@Bi) + j(Ar@Bi + Ai@Br)
    Ar, Ai = A[..., 0], A[..., 1]
    Br, Bi = B[..., 0], B[..., 1]
    
    Cr = torch.matmul(Ar, Br) - torch.matmul(Ai, Bi)
    Ci = torch.matmul(Ar, Bi) + torch.matmul(Ai, Br)
    
    return torch.stack([Cr, Ci], dim=-1)


def complex_conj(x: torch.Tensor) -> torch.Tensor:
    """
    Complex conjugate.
    
    Args:
        x: Complex tensor [..., 2] (real, imag)
    
    Returns:
        Conjugate [..., 2]
    """
    return torch.stack([x[..., 0], -x[..., 1]], dim=-1)


def complex_abs(x: torch.Tensor) -> torch.Tensor:
    """
    Complex absolute value (magnitude).
    
    Args:
        x: Complex tensor [..., 2]
    
    Returns:
        Magnitude [...] (real-valued)
    """
    return torch.sqrt(x[..., 0]**2 + x[..., 1]**2 + 1e-12)


def complex_normalize(x: torch.Tensor, dim: int = -2) -> torch.Tensor:
    """
    Normalize complex vector to unit norm.
    
    Args:
        x: Complex tensor [..., N, 2]
        dim: Dimension to normalize over
    
    Returns:
        Normalized tensor [..., N, 2]
    """
    norm = torch.sqrt(torch.sum(x[..., 0]**2 + x[..., 1]**2, dim=dim, keepdim=True) + 1e-12)
    return x / norm.unsqueeze(-1)


def real_to_complex(x_real: torch.Tensor, x_imag: torch.Tensor) -> torch.Tensor:
    """
    Combine real and imaginary parts into complex tensor.
    
    Args:
        x_real: Real part [...]
        x_imag: Imaginary part [...]
    
    Returns:
        Complex tensor [..., 2]
    """
    return torch.stack([x_real, x_imag], dim=-1)


def complex_to_2channel(x: torch.Tensor) -> torch.Tensor:
    """
    Convert complex [..., N, 2] to 2-channel real [..., 2*N] for CNN input.
    
    Args:
        x: Complex tensor [..., N, 2]
    
    Returns:
        Real tensor [..., 2*N]
    """
    batch_shape = x.shape[:-2]
    N = x.shape[-2]
    return x.reshape(*batch_shape, 2 * N)


class ComplexLinear(nn.Module):
    """Complex-valued linear layer."""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.real = nn.Linear(in_features, out_features)
        self.imag = nn.Linear(in_features, out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Complex input [..., in_features, 2]
        
        Returns:
            Complex output [..., out_features, 2]
        """
        x_real, x_imag = x[..., 0], x[..., 1]
        
        # (a + jb) @ (c + jd) = (ac - bd) + j(ad + bc)
        out_real = self.real(x_real) - self.imag(x_imag)
        out_imag = self.real(x_imag) + self.imag(x_real)
        
        return torch.stack([out_real, out_imag], dim=-1)