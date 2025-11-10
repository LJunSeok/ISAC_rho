"""
Generate real Pareto sweep data by evaluating fixed-ρ across grid.
"""

print("DEBUG: Script starting...")

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy

print("DEBUG: Basic imports done")

from src.configs import SystemConfig, ModelConfig, TrainingConfig, AblationConfig
from src.utils.seed import set_seed
from src.utils.io import load_checkpoint, get_checkpoint_name, save_metrics_csv, load_config
from src.utils.metrics import batch_metrics
from src.sim.echoes import generate_echo_history, apply_duration_normalization
from src.models.backbone_dual import DualStreamBackbone
from src.models.heads import ISACModel

print("DEBUG: All imports successful")


def run_fixed_rho_sweep(
    model: ISACModel,
    sys_config: SystemConfig,
    train_config: TrainingConfig,
    rho_grid: list,
    num_batches: int = 50
):
    """
    Run real evaluation with fixed rho values across a grid.
    
    Args:
        model: Trained model
        sys_config: System config
        train_config: Training config
        rho_grid: List of rho values to evaluate
        num_batches: Number of batches per rho value
    
    Returns:
        DataFrame with columns [rho, sum_rate, crb_trace, composite]
    """
    print(f"DEBUG: Starting sweep with {len(rho_grid)} rho values, {num_batches} batches each")
    model.eval()
    results = []
    
    for rho_val in tqdm(rho_grid, desc="Fixed-ρ Sweep"):
        metrics_list = []
        
        # Create ablation config for this fixed rho
        abl_config = AblationConfig(fixed_rho=True, fixed_rho_value=rho_val)
        
        with torch.no_grad():
            for batch_idx in range(num_batches):
                echo_data = generate_echo_history(sys_config, train_config.batch_size, sys_config.tau)
                
                sensing_norm, comm_norm = apply_duration_normalization(
                    echo_data['sensing_echoes'],
                    echo_data['comm_echoes'],
                    echo_data['rho_prev']
                )
                
                outputs = model(sensing_norm, comm_norm, echo_data['rho_prev'])
                
                # Override rho with fixed value
                outputs['rho'] = torch.full_like(outputs['rho'], rho_val)
                
                metrics = batch_metrics(
                    echo_data['H_comm'], echo_data['H_sense'],
                    outputs['F_S'], outputs['F_C'], outputs['W_C'], outputs['rho'],
                    echo_data['target_params'],
                    sys_config, train_config.lambda_trade
                )
                metrics_list.append(metrics)
        
        # Aggregate over batches
        metrics_df = pd.DataFrame(metrics_list)
        results.append({
            'rho': rho_val,
            'sum_rate': metrics_df['sum_rate'].mean(),
            'sum_rate_std': metrics_df['sum_rate'].std(),
            'crb_trace': metrics_df['crb_trace'].mean(),
            'crb_trace_std': metrics_df['crb_trace'].std(),
            'composite': metrics_df['composite'].mean(),
        })
    
    print(f"DEBUG: Sweep complete, {len(results)} results collected")
    return pd.DataFrame(results)


def main():
    print("DEBUG: Entering main()")
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Pareto sweep data")
    parser.add_argument('--lambda-vals', type=float, nargs='+', default=[0.01, 0.50, 0.99],
                       help='Lambda values to sweep')
    parser.add_argument('--rho-min', type=float, default=0.0, help='Min rho value')
    parser.add_argument('--rho-max', type=float, default=0.95, help='Max rho value')
    parser.add_argument('--rho-points', type=int, default=20, help='Number of rho points')
    parser.add_argument('--num-batches', type=int, default=50, help='Batches per rho')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    print(f"DEBUG: Args parsed - lambda_vals={args.lambda_vals}, rho_points={args.rho_points}")
    
    print("="*60)
    print("Generating Real Pareto Sweep Data")
    print("="*60)
    
    base_path = Path("results")
    print(f"DEBUG: Base path = {base_path.absolute()}")
    
    set_seed(args.seed)
    
    # Generate rho grid
    rho_grid = np.linspace(args.rho_min, args.rho_max, args.rho_points).tolist()
    print(f"ρ grid: {args.rho_points} points from {args.rho_min} to {args.rho_max}")
    print(f"DEBUG: rho_grid = {rho_grid[:3]}...{rho_grid[-3:]}")
    
    # Load base configs
    print("DEBUG: Loading configs...")
    sys_config = load_config(SystemConfig, base_path / 'sys_config.json')
    model_config = load_config(ModelConfig, base_path / 'model_config.json')
    train_config = load_config(TrainingConfig, base_path / 'train_config.json')
    print("DEBUG: Configs loaded")
    
    # Build model architecture
    input_dim_sense = sys_config.M * sys_config.Nr * sys_config.Nt * 2
    input_dim_comm = sys_config.K * sys_config.Nr * sys_config.Nt * 2
    print(f"DEBUG: Building model - sense_dim={input_dim_sense}, comm_dim={input_dim_comm}")
    
    backbone = DualStreamBackbone(
        input_dim_sense, input_dim_comm,
        model_config.cnn_channels,
        model_config.lstm_hidden,
        model_config.lstm_layers,
        model_config.attn_heads,
        model_config.dropout
    )
    print("DEBUG: Backbone created")
    
    # Process each lambda value
    for lambda_val in args.lambda_vals:
        print(f"\n{'='*60}")
        print(f"Processing λ={lambda_val}")
        print(f"{'='*60}")
        
        # Get checkpoint name
        checkpoint_name = get_checkpoint_name(
            alpha_chatter=train_config.alpha_chatter,
            lambda_trade=lambda_val,
            seed=args.seed
        )
        checkpoint_path = base_path / 'checkpoints' / checkpoint_name
        
        print(f"DEBUG: Looking for checkpoint: {checkpoint_path}")
        
        if not checkpoint_path.exists():
            print(f"ERROR: Checkpoint not found: {checkpoint_name}")
            print(f"       Expected at: {checkpoint_path.absolute()}")
            print(f"Train with: python -m src.train_dynamic_rho --lambda {lambda_val}")
            continue
        
        # Load model
        print("DEBUG: Loading model...")
        model = ISACModel(backbone, sys_config)
        load_checkpoint(model, None, checkpoint_path)
        print(f"Loaded: {checkpoint_name}")
        
        # Run sweep
        print("DEBUG: Starting sweep evaluation...")
        sweep_df = run_fixed_rho_sweep(
            model, sys_config, train_config, rho_grid, args.num_batches
        )
        
        print(f"DEBUG: Sweep complete, got {len(sweep_df)} rows")
        
        # Save results
        save_dir = f"{lambda_val:.2f}".replace('.', 'p')
        output_name = f"pareto_sweep_l{save_dir}_s{args.seed}.csv"
        output_path = base_path / output_name
        
        print(f"DEBUG: Saving to {output_path.absolute()}")
        save_metrics_csv(sweep_df, output_path)
        
        print(f"✓ Saved: {output_path}")
        print(f"  Rate range: [{sweep_df['sum_rate'].min():.2f}, {sweep_df['sum_rate'].max():.2f}]")
        print(f"  CRB range:  [{sweep_df['crb_trace'].min():.6f}, {sweep_df['crb_trace'].max():.6f}]")
    
    print("\n" + "="*60)
    print("Pareto sweep complete!")
    print("="*60)
    print("Next: python -m src.plot_analysis")


if __name__ == '__main__':
    print("DEBUG: __main__ check passed")
    main()
    print("DEBUG: Script completed successfully")