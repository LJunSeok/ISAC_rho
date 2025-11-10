"""
Dynamic-ρ loss with anti-chatter regularization.
"""

import torch
import torch.nn as nn
from ..utils.metrics import compute_sinr, compute_sum_rate
from ..sim.crb import compute_crb_trace


class DynamicRhoLoss(nn.Module):
    """
    Loss: λ·[-(1-ρ)·sum_rate] + (1-λ)·scale·tr(CRB) + α·(ρ-ρ_prev)^2 + L2
    """
    
    def __init__(
        self,
        sys_config,
        lambda_trade: float = 0.5,
        alpha_chatter: float = 1e-2,
        l2_logit: float = 1e-4,
        crb_scale: float = 1.0
    ):
        super().__init__()
        self.sys_config = sys_config
        self.lambda_trade = lambda_trade
        self.alpha_chatter = alpha_chatter
        self.l2_logit = l2_logit
        self.crb_scale = crb_scale
        self.calibrated = False
    
    def calibrate(self, calibration_batch: dict, model, auto_calibrate: bool = True, target_balance: float = 0.5):
        """
        Calibrate CRB scaling on a representative batch.
        
        Args:
            calibration_batch: Dict with outputs, H_comm, H_sense, target_params, rho_prev_last
            model: ISAC model
            auto_calibrate: If True, compute scale automatically
            target_balance: Target ratio (scaled_CRB / Rate)
        """
        if not auto_calibrate or self.calibrated:
            return
        
        with torch.no_grad():
            outputs = calibration_batch['outputs']
            F_S = outputs['F_S']
            F_C = outputs['F_C']
            W_C = outputs['W_C']
            rho = outputs['rho']
            
            # Compute raw rate
            sinr = compute_sinr(
                calibration_batch['H_comm'], F_C, W_C,
                self.sys_config.noise_power, self.sys_config.tx_power
            )
            sum_rate = compute_sum_rate(sinr, rho)
            mean_rate = sum_rate.mean().item()
            
            # Compute raw CRB
            crb_trace = compute_crb_trace(
                calibration_batch['H_sense'], F_S, rho,
                calibration_batch['target_params'],
                self.sys_config.T, self.sys_config.noise_power,
                self.sys_config.wavelength
            )
            mean_crb = crb_trace.mean().item()
            
            # Compute scale: target_balance * mean_rate / mean_crb
            if mean_crb > 1e-12:
                self.crb_scale = (target_balance * mean_rate) / mean_crb
            else:
                self.crb_scale = 1.0
            
            self.calibrated = True
            
            #print(f"\n{'='*40}")
            #print("CRB Auto-Calibration Results")
            #print(f"{'='*40}")
            #print(f"  Initial Rate:      {mean_rate:.4f} bits/s/Hz")
            #print(f"  Initial CRB:       {mean_crb:.6f}")
            #print(f"  Ratio (Rate/CRB):  {mean_rate/mean_crb:.1f}×")
            #print(f"  Target balance:    {target_balance:.2f}")
            #print(f"  → CRB Scale:       {self.crb_scale:.2f}")
            #print(f"  → Scaled CRB:      {mean_crb * self.crb_scale:.4f}")
            #print(f"{'='*40}\n")
    
    def forward(
        self,
        outputs: dict,
        H_comm: torch.Tensor,
        H_sense: torch.Tensor,
        target_params: dict,
        rho_prev_last: torch.Tensor
    ) -> dict:
        """
        Compute loss.
        
        Args:
            outputs: Model outputs (F_S, F_C, W_C, rho)
            H_comm: [B, K, Nr, Nt, 2]
            H_sense: [B, M, Nr, Nt, 2]
            target_params: Target parameters
            rho_prev_last: [B] previous slot's rho
        
        Returns:
            Dict with loss and components
        """
        F_S = outputs['F_S']
        F_C = outputs['F_C']
        W_C = outputs['W_C']
        rho = outputs['rho']
        
        # Communication term: -time_scaled_sum_rate
        sinr = compute_sinr(
            H_comm, F_C, W_C,
            self.sys_config.noise_power,
            self.sys_config.tx_power
        )
        sum_rate = compute_sum_rate(sinr, rho)
        comm_loss = -sum_rate.mean()
        
        # Sensing term: tr(CRB)
        crb_trace = compute_crb_trace(
            H_sense, F_S, rho, target_params,
            self.sys_config.T,
            self.sys_config.noise_power,
            self.sys_config.wavelength
        )
        sense_loss = crb_trace.mean()
        
        # Anti-chatter: (rho - rho_prev)^2
        chatter_loss = torch.mean((rho - rho_prev_last)**2)
        
        # L2 regularization on logits (internal weights)
        l2_loss = 0.0
        for name, param in self.named_parameters():
            if 'fc' in name and 'weight' in name:
                l2_loss += torch.sum(param**2)
        
        # Total loss
        loss = (
            self.lambda_trade * comm_loss +
            (1 - self.lambda_trade) * sense_loss * self.crb_scale +
            self.alpha_chatter * chatter_loss +
            self.l2_logit * l2_loss
        )
        
        return {
            'loss': loss,
            'comm_loss': comm_loss,
            'sense_loss': sense_loss,
            'chatter_loss': chatter_loss,
            'l2_loss': l2_loss,
            'sum_rate': -comm_loss.item(),  # For logging
            'crb_trace': sense_loss.item(),
            'crb_trace_scaled': (sense_loss * self.crb_scale).item(),
        }