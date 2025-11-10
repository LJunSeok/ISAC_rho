"""
Sanity check: quick overfit test on tiny batch.
"""

import torch
import torch.optim as optim
from tqdm import tqdm

from src.configs import SystemConfig, ModelConfig, TrainingConfig
from src.utils.seed import set_seed
from src.sim.echoes import generate_echo_history, apply_duration_normalization
from src.models.backbone_dual import DualStreamBackbone
from src.models.heads import ISACModel
from src.models.loss_dynamic import DynamicRhoLoss


def sanity_check():
    """Run sanity check to ensure model can overfit small batch."""
    
    print("="*60)
    print("Sanity Check: Overfit on Tiny Batch")
    print("="*60)
    
    set_seed(42)
    
    # Configs (small)
    sys_config = SystemConfig()
    model_config = ModelConfig(lstm_hidden=64)  # Smaller for speed
    train_config = TrainingConfig(lr=5e-3, batch_size=4)
    
    # Build model
    input_dim_sense = sys_config.M * sys_config.Nr * sys_config.Nt * 2
    input_dim_comm = sys_config.K * sys_config.Nr * sys_config.Nt * 2
    
    backbone = DualStreamBackbone(
        input_dim_sense, input_dim_comm,
        model_config.cnn_channels,
        model_config.lstm_hidden,
        model_config.lstm_layers,
        model_config.attn_heads,
        model_config.dropout
    )
    
    model = ISACModel(backbone, sys_config)
    optimizer = optim.Adam(model.parameters(), lr=train_config.lr)
    
    loss_fn = DynamicRhoLoss(
        sys_config,
        train_config.lambda_trade,
        train_config.alpha_chatter,
        train_config.l2_logit
    )
    
    # Generate single batch
    echo_data = generate_echo_history(sys_config, train_config.batch_size, sys_config.tau)
    sensing_norm, comm_norm = apply_duration_normalization(
        echo_data['sensing_echoes'],
        echo_data['comm_echoes'],
        echo_data['rho_prev']
    )
    
    print(f"\nBatch size: {train_config.batch_size}")
    print("Starting overfit test (should see loss decrease)...\n")
    
    # Overfit loop
    num_iters = 100
    initial_loss = None
    final_loss = None
    
    model.train()
    for i in tqdm(range(num_iters), desc="Overfitting"):
        outputs = model(sensing_norm, comm_norm, echo_data['rho_prev'])
        
        rho_prev_last = echo_data['rho_prev'][:, -1]
        loss_dict = loss_fn(outputs, echo_data['H_comm'], echo_data['H_sense'],
                           echo_data['target_params'], rho_prev_last)
        
        optimizer.zero_grad()
        loss_dict['loss'].backward()
        optimizer.step()
        
        if i == 0:
            initial_loss = loss_dict['loss'].item()
        if i == num_iters - 1:
            final_loss = loss_dict['loss'].item()
        
        if (i + 1) % 20 == 0:
            print(f"Iter {i+1}: Loss = {loss_dict['loss'].item():.4f}, "
                  f"Rate = {loss_dict['sum_rate']:.2f}, "
                  f"CRB = {loss_dict['crb_trace']:.4f}, "
                  f"ρ = {outputs['rho'].mean().item():.3f}")
    
    # Check if loss decreased
    print("\n" + "="*60)
    print("Sanity Check Results:")
    print("="*60)
    print(f"Initial Loss: {initial_loss:.4f}")
    print(f"Final Loss:   {final_loss:.4f}")
    print(f"Decrease:     {initial_loss - final_loss:.4f}")
    
    if final_loss < initial_loss * 0.5:
        print("\n✓ PASS: Model successfully overfits (loss decreased >50%)")
        return True
    else:
        print("\n✗ FAIL: Model did not overfit sufficiently")
        return False


if __name__ == '__main__':
    success = sanity_check()
    exit(0 if success else 1)