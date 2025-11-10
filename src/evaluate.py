"""
Evaluate trained dynamic-ρ model on test set.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.configs import SystemConfig, ModelConfig, TrainingConfig
from src.utils.seed import set_seed
from src.utils.io import load_checkpoint, save_metrics_csv, get_checkpoint_name, get_metrics_name
from src.utils.metrics import batch_metrics
from src.sim.echoes import generate_echo_history, apply_duration_normalization
from src.models.backbone_dual import DualStreamBackbone
from src.models.heads import ISACModel


def evaluate_model(
    model: ISACModel,
    sys_config: SystemConfig,
    train_config: TrainingConfig,
    num_batches: int = 50
):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained ISAC model
        sys_config: System config
        train_config: Training config
        num_batches: Number of test batches
    
    Returns:
        (metrics_df, analysis_df)
    """
    model.eval()
    
    metrics_list = []
    analysis_list = []
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Evaluating"):
            # Generate test data
            echo_data = generate_echo_history(sys_config, train_config.batch_size, sys_config.tau)
            
            # Normalize
            sensing_norm, comm_norm = apply_duration_normalization(
                echo_data['sensing_echoes'],
                echo_data['comm_echoes'],
                echo_data['rho_prev']
            )
            
            # Forward
            outputs = model(sensing_norm, comm_norm, echo_data['rho_prev'])
            
            # Compute metrics
            metrics = batch_metrics(
                echo_data['H_comm'], echo_data['H_sense'],
                outputs['F_S'], outputs['F_C'], outputs['W_C'], outputs['rho'],
                echo_data['target_params'],
                sys_config, train_config.lambda_trade
            )
            metrics_list.append(metrics)
            
            # Per-sample analysis
            for i in range(train_config.batch_size):
                analysis_list.append({
                    'batch': batch_idx,
                    'sample': i,
                    'rho': outputs['rho'][i].item(),
                    'doppler_mean': echo_data['doppler'][i].mean().item(),
                    'aod_change_mean': echo_data['aod_change'][i].mean().item(),
                })
    
    # Aggregate metrics
    metrics_df = pd.DataFrame(metrics_list)
    analysis_df = pd.DataFrame(analysis_list)
    
    return metrics_df, analysis_df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument('--checkpoint', type=str, default='dynamic_rho_model.pt',
                       help='Checkpoint filename to evaluate')
    args = parser.parse_args()
    
    print("="*60)
    print("Evaluating Dynamic-ρ Model")
    print("="*60)
    
    # Load configs
    from src.utils.io import load_config
    
    base_path = Path("results")
    sys_config = load_config(SystemConfig, base_path / 'sys_config.json')
    model_config = load_config(ModelConfig, base_path / 'model_config.json')
    train_config = load_config(TrainingConfig, base_path / 'train_config.json')
    
    set_seed(train_config.seed)
    
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
    
    # Load checkpoint
    checkpoint_path = base_path / 'checkpoints' / args.checkpoint
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return
    
    epoch, history = load_checkpoint(model, None, checkpoint_path)
    print(f"Loaded model from: {args.checkpoint}")
    print(f"Trained for {epoch} epochs")
    
    # Evaluate
    metrics_df, analysis_df = evaluate_model(
        model, sys_config, train_config, num_batches=train_config.test_batches
    )
    
    # Print summary
    print("\n" + "="*60)
    print("Test Results:")
    print("="*60)
    print(f"Sum Rate:      {metrics_df['sum_rate'].mean():.4f} ± {metrics_df['sum_rate'].std():.4f}")
    print(f"CRB Trace:     {metrics_df['crb_trace'].mean():.4f} ± {metrics_df['crb_trace'].std():.4f}")
    print(f"Composite:     {metrics_df['composite'].mean():.4f} ± {metrics_df['composite'].std():.4f}")
    print(f"ρ Mean:        {metrics_df['rho_mean'].mean():.4f}")
    print(f"ρ Std:         {metrics_df['rho_std'].mean():.4f}")
    print("="*60)
    
    # Save results with matching names
    metrics_name = get_metrics_name(args.checkpoint, 'test_metrics')
    analysis_name = get_metrics_name(args.checkpoint, 'test_rho_analysis')
    
    save_metrics_csv(metrics_df, base_path / metrics_name)
    save_metrics_csv(analysis_df, base_path / analysis_name)
    
    print(f"\nResults saved to:")
    print(f"  - {base_path / metrics_name}")
    print(f"  - {base_path / analysis_name}")

if __name__ == '__main__':
    main()