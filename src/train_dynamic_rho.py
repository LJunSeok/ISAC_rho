"""
Main training script for dynamic-ρ learning.
"""

import argparse
import torch
import torch.optim as optim
from pathlib import Path
import numpy as np
from tqdm import tqdm

from src.configs import SystemConfig, ModelConfig, TrainingConfig
from src.utils.seed import set_seed
from src.utils.io import save_checkpoint, setup_results_dir, save_config, get_checkpoint_name
from src.utils.metrics import batch_metrics
from src.utils.plotting import plot_training_curves
from src.sim.echoes import generate_echo_history, apply_duration_normalization
from src.models.backbone_dual import DualStreamBackbone
from src.models.heads import ISACModel
from src.models.loss_dynamic import DynamicRhoLoss


def oracle_rho_grid_search(
    model: ISACModel,
    echo_data: dict,
    sys_config: SystemConfig,
    lambda_trade: float,
    rho_grid: tuple = (0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)
) -> torch.Tensor:
    """
    Grid search over ρ to find oracle (best) value per batch sample.
    
    Args:
        model: ISAC model
        echo_data: Echo history data
        sys_config: System config
        lambda_trade: Trade-off parameter
        rho_grid: Grid of rho values to search
    
    Returns:
        oracle_rho: [B] best rho per sample
    """
    from src.utils.metrics import compute_composite_metric
    
    B = echo_data['H_comm'].shape[0]
    
    # Get beams (without rho head)
    with torch.no_grad():
        sensing_norm, comm_norm = apply_duration_normalization(
            echo_data['sensing_echoes'],
            echo_data['comm_echoes'],
            echo_data['rho_prev']
        )
        outputs = model(sensing_norm, comm_norm, echo_data['rho_prev'])
        F_S = outputs['F_S']
        F_C = outputs['F_C']
        W_C = outputs['W_C']
    
    best_rho = torch.zeros(B)
    best_composite = torch.full((B,), float('inf'))
    
    for rho_val in rho_grid:
        rho_tensor = torch.full((B,), rho_val)
        
        metrics = batch_metrics(
            echo_data['H_comm'], echo_data['H_sense'],
            F_S, F_C, W_C, rho_tensor,
            echo_data['target_params'],
            sys_config, lambda_trade
        )
        
        # Compute composite for each sample (need to recompute per-sample)
        from src.utils.metrics import compute_sinr, compute_sum_rate
        from src.sim.crb import compute_crb_trace
        
        sinr = compute_sinr(echo_data['H_comm'], F_C, W_C, sys_config.noise_power, sys_config.tx_power)
        sum_rate = compute_sum_rate(sinr, rho_tensor)
        crb_trace = compute_crb_trace(
            echo_data['H_sense'], F_S, rho_tensor,
            echo_data['target_params'], sys_config.T,
            sys_config.noise_power, sys_config.wavelength
        )
        composite = compute_composite_metric(sum_rate, crb_trace, lambda_trade)
        
        # Update best
        better_mask = composite < best_composite
        best_rho[better_mask] = rho_val
        best_composite[better_mask] = composite[better_mask]
    
    return best_rho


def pretrain_fixed_rho(
    model: ISACModel,
    optimizer: optim.Optimizer,
    sys_config: SystemConfig,
    train_config: TrainingConfig,
    fixed_rho: float
):
    """
    Pretrain with fixed ρ to stabilize beams.
    
    Args:
        model: ISAC model
        optimizer: Optimizer
        sys_config: System config
        train_config: Training config
        fixed_rho: Fixed rho value
    """
    print(f"\n=== Pretraining with fixed ρ={fixed_rho:.2f} ===")
    
    loss_fn = DynamicRhoLoss(
        sys_config,
        train_config.lambda_trade,
        alpha_chatter=0.0,  # No chatter penalty in pretrain
        l2_logit=train_config.l2_logit,
        crb_scale=train_config.crb_scale
    )
    
    model.train()
    
    for epoch in range(train_config.pretrain_epochs):
        epoch_loss = 0.0
        
        for batch_idx in tqdm(range(train_config.batches_per_epoch), desc=f"Pretrain Epoch {epoch+1}"):
            # Generate data
            rho_history = torch.ones(train_config.batch_size, sys_config.tau) * fixed_rho
            echo_data = generate_echo_history(sys_config, train_config.batch_size, sys_config.tau, rho_history)
            
            # Normalize
            sensing_norm, comm_norm = apply_duration_normalization(
                echo_data['sensing_echoes'],
                echo_data['comm_echoes'],
                echo_data['rho_prev']
            )
            
            # Forward
            outputs = model(sensing_norm, comm_norm, echo_data['rho_prev'])
            
            # Override rho with fixed value
            outputs['rho'] = torch.full_like(outputs['rho'], fixed_rho)
            
            # Loss
            rho_prev_last = echo_data['rho_prev'][:, -1]
            loss_dict = loss_fn(outputs, echo_data['H_comm'], echo_data['H_sense'],
                               echo_data['target_params'], rho_prev_last)
            
            # Backward
            optimizer.zero_grad()
            loss_dict['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            optimizer.step()
            
            epoch_loss += loss_dict['loss'].item()
        
        print(f"Pretrain Epoch {epoch+1}: Loss = {epoch_loss/train_config.batches_per_epoch:.4f}")


def train_with_distillation(
    model: ISACModel,
    optimizer: optim.Optimizer,
    sys_config: SystemConfig,
    train_config: TrainingConfig
):
    """
    Train with oracle-ρ distillation for warm-start.
    
    Args:
        model: ISAC model
        optimizer: Optimizer
        sys_config: System config
        train_config: Training config
    """
    print("\n=== Distillation from Oracle-ρ ===")
    
    loss_fn = DynamicRhoLoss(
        sys_config,
        train_config.lambda_trade,
        train_config.alpha_chatter,
        train_config.l2_logit,
        crb_scale=train_config.crb_scale
    )
    
    model.train()
    
    for epoch in range(train_config.distill_epochs):
        epoch_loss = 0.0
        epoch_distill_loss = 0.0
        
        for batch_idx in tqdm(range(train_config.batches_per_epoch), desc=f"Distill Epoch {epoch+1}"):
            # Generate data
            echo_data = generate_echo_history(sys_config, train_config.batch_size, sys_config.tau)
            
            # Normalize
            sensing_norm, comm_norm = apply_duration_normalization(
                echo_data['sensing_echoes'],
                echo_data['comm_echoes'],
                echo_data['rho_prev']
            )
            
            # Forward
            outputs = model(sensing_norm, comm_norm, echo_data['rho_prev'])
            
            # Compute oracle rho
            oracle_rho = oracle_rho_grid_search(
                model, echo_data, sys_config, train_config.lambda_trade
            )
            
            # Loss
            rho_prev_last = echo_data['rho_prev'][:, -1]
            loss_dict = loss_fn(outputs, echo_data['H_comm'], echo_data['H_sense'],
                               echo_data['target_params'], rho_prev_last)
            
            # Distillation loss
            distill_loss = torch.mean((outputs['rho'] - oracle_rho)**2)
            total_loss = loss_dict['loss'] + train_config.distill_weight * distill_loss
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            optimizer.step()
            
            epoch_loss += loss_dict['loss'].item()
            epoch_distill_loss += distill_loss.item()
        
        avg_loss = epoch_loss / train_config.batches_per_epoch
        avg_distill = epoch_distill_loss / train_config.batches_per_epoch
        print(f"Distill Epoch {epoch+1}: Loss = {avg_loss:.4f}, Distill = {avg_distill:.4f}")


def train_dynamic_rho(
    model: ISACModel,
    optimizer: optim.Optimizer,
    sys_config: SystemConfig,
    train_config: TrainingConfig,
    save_dir: Path,
    checkpoint_name: str = None
):
    """
    Main training loop for dynamic-ρ learning.
    
    Args:
        model: ISAC model
        optimizer: Optimizer
        sys_config: System config
        train_config: Training config
        save_dir: Directory to save results
    """
    print("\n=== Training Dynamic-ρ Model ===")
    
    loss_fn = DynamicRhoLoss(
        sys_config,
        train_config.lambda_trade,
        train_config.alpha_chatter,
        train_config.l2_logit,
        crb_scale=train_config.crb_scale
    )

    if train_config.auto_calibrate:
        print("Running CRB auto-calibration...")
        model.eval()
        with torch.no_grad():
            calib_data = generate_echo_history(sys_config, train_config.batch_size, sys_config.tau)
            sensing_norm, comm_norm = apply_duration_normalization(
                calib_data['sensing_echoes'],
                calib_data['comm_echoes'],
                calib_data['rho_prev']
            )
            outputs = model(sensing_norm, comm_norm, calib_data['rho_prev'])
            
            calibration_batch = {
                'outputs': outputs,
                'H_comm': calib_data['H_comm'],
                'H_sense': calib_data['H_sense'],
                'target_params': calib_data['target_params'],
                'rho_prev_last': calib_data['rho_prev'][:, -1]
            }
            
            loss_fn.calibrate(calibration_batch, model, 
                            auto_calibrate=train_config.auto_calibrate,
                            target_balance=train_config.target_balance)
        model.train()

    history = {
        'loss': [],
        'sum_rate': [],
        'crb_trace': [],
        'crb_trace_scaled': [],
        'rho_mean': [],
        'rho_std': [],
    }
    
    model.train()
    
    for epoch in range(train_config.epochs):
        epoch_loss = 0.0
        epoch_rate = 0.0
        epoch_crb = 0.0
        epoch_crb_scaled = 0.0
        epoch_chatter = 0.0
        epoch_rho = []
        
        for batch_idx in tqdm(range(train_config.batches_per_epoch), desc=f"Epoch {epoch+1}/{train_config.epochs}"):
            # Generate data
            echo_data = generate_echo_history(sys_config, train_config.batch_size, sys_config.tau)
            
            # Normalize
            sensing_norm, comm_norm = apply_duration_normalization(
                echo_data['sensing_echoes'],
                echo_data['comm_echoes'],
                echo_data['rho_prev']
            )
            
            # Forward
            outputs = model(sensing_norm, comm_norm, echo_data['rho_prev'])
            
            # Loss
            rho_prev_last = echo_data['rho_prev'][:, -1]
            loss_dict = loss_fn(outputs, echo_data['H_comm'], echo_data['H_sense'],
                               echo_data['target_params'], rho_prev_last)
            
            # Backward
            optimizer.zero_grad()
            loss_dict['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            optimizer.step()
            
            # Logging
            epoch_loss += loss_dict['loss'].item()
            epoch_rate += loss_dict['sum_rate']
            epoch_crb += loss_dict['crb_trace']
            epoch_crb_scaled += loss_dict['crb_trace_scaled']
            epoch_chatter += loss_dict['chatter_loss'].item()
            epoch_rho.extend(outputs['rho'].detach().cpu().numpy())
        
        # Epoch statistics
        avg_loss = epoch_loss / train_config.batches_per_epoch
        avg_rate = epoch_rate / train_config.batches_per_epoch
        avg_crb = epoch_crb / train_config.batches_per_epoch
        avg_crb_scaled = epoch_crb_scaled / train_config.batches_per_epoch 
        avg_chatter = epoch_chatter / train_config.batches_per_epoch
        avg_rho = np.mean(epoch_rho)
        std_rho = np.std(epoch_rho)
        
        history['loss'].append(avg_loss)
        history['sum_rate'].append(avg_rate)
        history['crb_trace'].append(avg_crb)
        history['crb_trace_scaled'].append(avg_crb_scaled)
        history['rho_mean'].append(avg_rho)
        history['rho_std'].append(std_rho)
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Rate={avg_rate:.2f}, "
              f"CRB_raw={avg_crb:.4f}, CRB_scaled={avg_crb_scaled:.2f}, "
              f"Chatter={avg_chatter:.6f}, ρ={avg_rho:.3f}±{std_rho:.3f}")
    
    # Save final checkpoint
    # Only generate checkpoint name if not provided
    if checkpoint_name is None:
        checkpoint_name = get_checkpoint_name(
            alpha_chatter=train_config.alpha_chatter,
            lambda_trade=train_config.lambda_trade,
            seed=train_config.seed
        )
    
    # Save final checkpoint
    save_checkpoint(model, optimizer, train_config.epochs, history,
                   save_dir / 'checkpoints' / checkpoint_name)
    
    # Plot training curves
    plot_training_curves(history, save_dir / 'plots' / 'training_curves.png')
    
    return history


def main():
    parser = argparse.ArgumentParser(description="Train dynamic-ρ ISAC model")
    parser.add_argument('--lambda', dest='lambda_trade', type=float, default=0.5,
                       help='Trade-off parameter (0=rate, 1=sensing)')
    parser.add_argument('--alpha', dest='alpha_chatter', type=float, default=1e-2,
                       help='Anti-chatter regularization weight')
    parser.add_argument('--rho-min', type=float, default=0.0, help='Minimum rho')
    parser.add_argument('--rho-max', type=float, default=0.4, help='Maximum rho')
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--pretrain-rho', type=float, default=None,
                       help='Fixed rho for pretraining (None=skip)')
    parser.add_argument('--use-distill', action='store_true',
                       help='Use oracle-rho distillation')
    parser.add_argument('--frozen-wc', action='store_true',
                       help='Freeze W_C combiner')
    parser.add_argument('--no-auto-calibrate', action='store_true',
                       help='Disable CRB auto-calibration')
    parser.add_argument('--target-balance', type=float, default=0.5,
                       help='Target balance ratio for CRB scaling')
    parser.add_argument('--checkpoint-name', type=str, default=None,
                       help='Custom checkpoint name (default: auto-generated)')
    parser.add_argument('--sys-config', type=str, default=None, help='Path to custom system config')
    parser.add_argument('--suffix', type=str, default='', help='Suffix for checkpoint name (e.g., _M16)')
        
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Setup results directory
    save_dir = Path("results")
    paths = setup_results_dir(save_dir)

    # Configs
    if args.sys_config:
        import json
        import dataclasses
        from typing import get_origin, get_args
        
        print(f"Loading custom system config from: {args.sys_config}")
        
        with open(args.sys_config, 'r') as f:
            config_dict = json.load(f)
        
        # Filter out non-init fields (like wavelength which is computed in __post_init__)
        init_fields = {f.name for f in dataclasses.fields(SystemConfig) if f.init}
        filtered_dict = {k: v for k, v in config_dict.items() if k in init_fields}
        
        # Convert lists back to tuples where needed
        for field in dataclasses.fields(SystemConfig):
            if field.name in filtered_dict:
                # Check if field type is a Tuple
                origin = get_origin(field.type)
                if origin is tuple:
                    # Convert list to tuple
                    if isinstance(filtered_dict[field.name], list):
                        filtered_dict[field.name] = tuple(filtered_dict[field.name])
        
        sys_config = SystemConfig(**filtered_dict)
        print(f"  M={sys_config.M}, K={sys_config.K}, Nt={sys_config.Nt}, Nr={sys_config.Nr}")
    else:
        sys_config = SystemConfig()
    model_config = ModelConfig()
    train_config = TrainingConfig(
        lambda_trade=args.lambda_trade,
        alpha_chatter=args.alpha_chatter,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        pretrain_rho=args.pretrain_rho,
        use_distill=args.use_distill,
        auto_calibrate=not args.no_auto_calibrate,
        target_balance=args.target_balance
    )

    # Store checkpoint name preference
    train_config.checkpoint_name = args.checkpoint_name
    
    # Save configs
    save_config(sys_config, paths['base'] / 'sys_config.json')
    save_config(model_config, paths['base'] / 'model_config.json')
    save_config(train_config, paths['base'] / 'train_config.json')
    
    # Compute input dimensions
    input_dim_sense = sys_config.M * sys_config.Nr * sys_config.Nt * 2
    input_dim_comm = sys_config.K * sys_config.Nr * sys_config.Nt * 2
    
    # Build model
    backbone = DualStreamBackbone(
        input_dim_sense, input_dim_comm,
        model_config.cnn_channels,
        model_config.lstm_hidden,
        model_config.lstm_layers,
        model_config.attn_heads,
        model_config.dropout
    )
    
    model = ISACModel(backbone, sys_config, frozen_wc=args.frozen_wc)
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay
    )
    
    #print("="*60)
    #print("ISAC Dynamic-ρ Training")
    #print("="*60)
    #print(f"System: Nt={sys_config.Nt}, Nr={sys_config.Nr}, K={sys_config.K}, M={sys_config.M}")
    #print(f"ρ bounds: [{sys_config.rho_min}, {sys_config.rho_max}]")
    #print(f"Trade-off λ: {train_config.lambda_trade}")
    #print(f"Anti-chatter α: {train_config.alpha_chatter}")
    #print(f"Auto-calibrate: {train_config.auto_calibrate}")
    #if train_config.auto_calibrate:
    #    print(f"Target balance: {train_config.target_balance}")
    #print(f"Epochs: {train_config.epochs}, LR: {train_config.lr}")
    #print(f"Seed: {args.seed}")
    #print("="*60)
    
    # Pretrain if requested
    if args.pretrain_rho is not None:
        pretrain_fixed_rho(model, optimizer, sys_config, train_config, args.pretrain_rho)
    
    # Distillation if requested
    if args.use_distill:
        train_with_distillation(model, optimizer, sys_config, train_config)
    
    checkpoint_name = get_checkpoint_name(
        alpha_chatter=train_config.alpha_chatter,
        lambda_trade=train_config.lambda_trade,
        seed=train_config.seed,
        use_distill=args.use_distill,
        frozen_wc=args.frozen_wc,
        pretrain_rho=train_config.pretrain_rho
    )
    
    # Add suffix for sweep models
    if args.suffix:
        checkpoint_name = checkpoint_name.replace('.pt', f'{args.suffix}.pt')
    
    # Main training - PASS checkpoint_name
    history = train_dynamic_rho(
        model, optimizer, sys_config, train_config, save_dir,
        checkpoint_name=checkpoint_name
    )

    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Model saved to: {paths['checkpoints'] / checkpoint_name}")
    print("="*60)


if __name__ == '__main__':
    main()