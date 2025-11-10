"""
Dual-stream CNN→LSTM→Attention backbone for dynamic-ρ learning.
"""

import torch
import torch.nn as nn
from typing import Tuple


class DualStreamCNN(nn.Module):
    """Dual-stream 1D CNN stems for sensing and comm echoes."""
    
    def __init__(self, input_dim_sense: int, input_dim_comm: int, out_channels: int = 32):
        super().__init__()
        
        # Sensing stream
        self.sense_conv1 = nn.Conv1d(input_dim_sense, out_channels, kernel_size=3, padding=1)
        self.sense_bn1 = nn.BatchNorm1d(out_channels)
        self.sense_conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.sense_bn2 = nn.BatchNorm1d(out_channels)
        
        # Comm stream
        self.comm_conv1 = nn.Conv1d(input_dim_comm, out_channels, kernel_size=3, padding=1)
        self.comm_bn1 = nn.BatchNorm1d(out_channels)
        self.comm_conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.comm_bn2 = nn.BatchNorm1d(out_channels)
        
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, sensing: torch.Tensor, comm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sensing: [B, tau, D_sense]
            comm: [B, tau, D_comm]
        
        Returns:
            (sense_feat [B, C], comm_feat [B, C])
        """
        B, tau, _ = sensing.shape
        
        # Transpose for Conv1d: [B, D, tau]
        sensing = sensing.transpose(1, 2)
        comm = comm.transpose(1, 2)
        
        # Sensing stream
        s = self.relu(self.sense_bn1(self.sense_conv1(sensing)))
        s = self.relu(self.sense_bn2(self.sense_conv2(s)))
        s_feat = self.pool(s).squeeze(-1)  # [B, C]
        
        # Comm stream
        c = self.relu(self.comm_bn1(self.comm_conv1(comm)))
        c = self.relu(self.comm_bn2(self.comm_conv2(c)))
        c_feat = self.pool(c).squeeze(-1)  # [B, C]
        
        return s_feat, c_feat


class CrossAttention(nn.Module):
    """Cross-attention between sensing and comm streams."""
    
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, sense_feat: torch.Tensor, comm_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sense_feat: [B, C]
            comm_feat: [B, C]
        
        Returns:
            fused_feat: [B, 2*C]
        """
        # Add sequence dimension
        s = sense_feat.unsqueeze(1)  # [B, 1, C]
        c = comm_feat.unsqueeze(1)  # [B, 1, C]
        
        # Cross-attention: sense attends to comm
        s_attn, _ = self.attn(s, c, c)
        s_out = self.norm(s + s_attn)
        
        # Concatenate
        fused = torch.cat([s_out.squeeze(1), c.squeeze(1)], dim=-1)  # [B, 2*C]
        
        return fused


class DualStreamBackbone(nn.Module):
    """
    Full dual-stream backbone: CNN stems → cross-attention → LSTM → attention.
    """
    
    def __init__(
        self,
        input_dim_sense: int,
        input_dim_comm: int,
        cnn_channels: int = 32,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        attn_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.cnn = DualStreamCNN(input_dim_sense, input_dim_comm, cnn_channels)
        self.cross_attn = CrossAttention(cnn_channels, attn_heads)
        
        # LSTM operates on sequence of fused features
        self.lstm = nn.LSTM(
            2 * cnn_channels + 1,  # +1 for rho_prev scalar
            lstm_hidden,
            lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        
        # Self-attention pooling
        self.self_attn = nn.MultiheadAttention(lstm_hidden, attn_heads, batch_first=True)
        self.norm = nn.LayerNorm(lstm_hidden)
        
        self.dropout = nn.Dropout(dropout)
        
        self.latent_dim = lstm_hidden
    
    def forward(
        self,
        sensing_echoes: torch.Tensor,
        comm_echoes: torch.Tensor,
        rho_prev: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            sensing_echoes: [B, tau, D_sense]
            comm_echoes: [B, tau, D_comm]
            rho_prev: [B, tau]
        
        Returns:
            latent: [B, lstm_hidden]
        """
        B, tau, _ = sensing_echoes.shape
        
        # Process each time step
        fused_seq = []
        for t in range(tau):
            s_t = sensing_echoes[:, t:t+1, :]  # [B, 1, D_sense]
            c_t = comm_echoes[:, t:t+1, :]  # [B, 1, D_comm]
            
            # CNN features
            s_feat, c_feat = self.cnn(s_t, c_t)  # [B, C], [B, C]
            
            # Cross-attention fusion
            fused_t = self.cross_attn(s_feat, c_feat)  # [B, 2*C]
            
            # Append rho_prev scalar
            fused_t = torch.cat([fused_t, rho_prev[:, t:t+1]], dim=-1)  # [B, 2*C+1]
            fused_seq.append(fused_t)
        
        fused_seq = torch.stack(fused_seq, dim=1)  # [B, tau, 2*C+1]
        
        # LSTM
        lstm_out, _ = self.lstm(fused_seq)  # [B, tau, lstm_hidden]
        lstm_out = self.dropout(lstm_out)
        
        # Self-attention pooling
        attn_out, _ = self.self_attn(lstm_out, lstm_out, lstm_out)
        attn_out = self.norm(lstm_out + attn_out)
        
        # Global average pooling
        latent = torch.mean(attn_out, dim=1)  # [B, lstm_hidden]
        
        return latent