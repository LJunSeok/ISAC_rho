"""
Run ablation studies.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy

from src.configs import SystemConfig, ModelConfig, TrainingConfig, AblationConfig
from src.utils.seed import set_seed
from src.utils.io import save_metrics_csv, load_config, get_checkpoint_name, load_checkpoint
from src.utils.metrics import batch_metrics
from src.sim.echoes import generate_echo_history, apply_duration_normalization
from src.models.backbone_dual import DualStreamBackbone
from src.models.heads import ISACModel
from src.train_dynamic_rho import oracle_rho_grid_search


def run_ablation(
    ablation_name: str,
    model: ISACModel,
    sys_config: SystemConfig,
    train_config: TrainingConfig,
    abl_config: AblationConfig,
    num_batches: int = 50
):
    """
    Run a single ablation experiment.
    
    Args:
        ablation_name: Name of ablation
        model: Base model
        sys_config: System config
        train_config: Training config
        abl_config: Ablation config
        num_batches: Number of test batches
    
    Returns:
        Metrics dictionary
    """
    print(f"\nRunning ablation: {ablation_name}")
    
    model_abl = deepcopy(model)
    model_abl.eval()
    
    metrics_list = []
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc=ablation_name):
            echo_data = generate_echo_history(sys_config, train_config.batch_size, sys_config.tau)
            
            # Apply ablation-specific modifications
            if abl_config.no_dual_stream:
                # No dual stream: use comm echoes as source, appropriately sized for each CNN
                comm_echoes_orig = echo_data['comm_echoes']   # [B, tau, 8192]
                
                # Keep comm echoes at original size for comm CNN
                comm_echoes = comm_echoes_orig  # [B, tau, 8192]
                
                # Tile comm echoes to sensing dimension for sensing CNN
                B, tau, comm_dim = comm_echoes_orig.shape
                sense_dim = echo_data['sensing_echoes'].shape[-1]  # 32768
                repeat_factor = (sense_dim + comm_dim - 1) // comm_dim
                
                sensing_echoes_tiled = comm_echoes_orig.repeat(1, 1, repeat_factor)
                sensing_echoes = sensing_echoes_tiled[:, :, :sense_dim]  # [B, tau, 32768]
            else:
                sensing_echoes = echo_data['sensing_echoes']
                comm_echoes = echo_data['comm_echoes']
            
            sensing_norm, comm_norm = apply_duration_normalization(
                sensing_echoes, comm_echoes, echo_data['rho_prev']
            )
            
            outputs = model_abl(sensing_norm, comm_norm, echo_data['rho_prev'])
            
            # Fixed-rho override
            if abl_config.fixed_rho:
                outputs['rho'] = torch.full_like(outputs['rho'], abl_config.fixed_rho_value)
            
            metrics = batch_metrics(
                echo_data['H_comm'], echo_data['H_sense'],
                outputs['F_S'], outputs['F_C'], outputs['W_C'], outputs['rho'],
                echo_data['target_params'],
                sys_config, train_config.lambda_trade
            )
            metrics_list.append(metrics)
    
    # Aggregate
    metrics_df = pd.DataFrame(metrics_list)
    return {
        'ablation': ablation_name,
        'sum_rate': metrics_df['sum_rate'].mean(),
        'crb_trace': metrics_df['crb_trace'].mean(),
        'composite': metrics_df['composite'].mean(),
        'efficiency': metrics_df['sum_rate'].mean() / (metrics_df['crb_trace'].mean() + 1e-6),
        'rho_mean': metrics_df['rho_mean'].mean(),
        'rho_std': metrics_df['rho_std'].mean(),
    }


def run_oracle_rho_analysis(
    model: ISACModel,
    sys_config: SystemConfig,
    train_config: TrainingConfig,
    abl_config: AblationConfig,
    num_batches: int = 20
):
    """
    Compute oracle-ρ upper bound via grid search.
    
    Args:
        model: Base model
        sys_config: System config
        train_config: Training config
        abl_config: Ablation config
        num_batches: Number of batches
    
    Returns:
        Metrics dictionary
    """
    print("\nRunning oracle-ρ analysis (upper bound)...")
    
    model.eval()
    metrics_list = []
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Oracle-ρ"):
            echo_data = generate_echo_history(sys_config, train_config.batch_size, sys_config.tau)
            
            # Find oracle rho per sample
            oracle_rho = oracle_rho_grid_search(
                model, echo_data, sys_config, train_config.lambda_trade,
                rho_grid=abl_config.oracle_rho_grid
            )
            
            sensing_norm, comm_norm = apply_duration_normalization(
                echo_data['sensing_echoes'],
                echo_data['comm_echoes'],
                echo_data['rho_prev']
            )
            
            outputs = model(sensing_norm, comm_norm, echo_data['rho_prev'])
            outputs['rho'] = oracle_rho  # Use oracle
            
            metrics = batch_metrics(
                echo_data['H_comm'], echo_data['H_sense'],
                outputs['F_S'], outputs['F_C'], outputs['W_C'], outputs['rho'],
                echo_data['target_params'],
                sys_config, train_config.lambda_trade
            )
            metrics_list.append(metrics)
    
    metrics_df = pd.DataFrame(metrics_list)
    return {
        'ablation': 'Oracle-ρ',
        'sum_rate': metrics_df['sum_rate'].mean(),
        'crb_trace': metrics_df['crb_trace'].mean(),
        'composite': metrics_df['composite'].mean(),
        'efficiency': metrics_df['sum_rate'].mean() / (metrics_df['crb_trace'].mean() + 1e-6),
        'rho_mean': metrics_df['rho_mean'].mean(),
        'rho_std': metrics_df['rho_std'].mean(),
    }


def main():
    print("="*60)
    print("Running Ablation Studies")
    print("="*60)
    
    # Load configs
    base_path = Path("results")
    sys_config = load_config(SystemConfig, base_path / 'sys_config.json')
    model_config = load_config(ModelConfig, base_path / 'model_config.json')
    train_config = load_config(TrainingConfig, base_path / 'train_config.json')
    abl_config = AblationConfig()
    
    set_seed(train_config.seed)
    
    # Build backbone
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
    
    # Determine baseline checkpoint
    baseline_checkpoint = get_checkpoint_name(
        alpha_chatter=train_config.alpha_chatter,
        lambda_trade=train_config.lambda_trade,
        seed=train_config.seed
    )
    baseline_path = base_path / 'checkpoints' / baseline_checkpoint
    
    if not baseline_path.exists():
        print(f"ERROR: Baseline checkpoint not found: {baseline_checkpoint}")
        print("Please run training first: python -m src.train_dynamic_rho")
        return
    
    print(f"Using baseline checkpoint: {baseline_checkpoint}")
    
    # Load baseline model
    model = ISACModel(backbone, sys_config)
    load_checkpoint(model, None, baseline_path)
    
    results = []
    
    # ===== Order: Proposed Method, Proposed-With-Distill, Proposed-Frozen-WC, 
    #              No-Dual-Stream, No-Anti-Chatter, Fixed-ρ, Oracle-ρ =====
    
    # 1. Proposed Method (Baseline)
    print("\n1. Proposed Method")
    baseline_result = run_ablation("Proposed Method", model, sys_config, train_config, AblationConfig())
    results.append(baseline_result)
    
    # 2. Proposed-With-Distill (trained variant)
    print("\n2. Proposed-With-Distill")
    distill_checkpoint = get_checkpoint_name(
        alpha_chatter=train_config.alpha_chatter,
        lambda_trade=train_config.lambda_trade,
        seed=train_config.seed,
        use_distill=True
    )
    distill_path = base_path / 'checkpoints' / distill_checkpoint
    
    if distill_path.exists():
        print(f"   Found checkpoint: {distill_checkpoint}")
        model_distill = ISACModel(deepcopy(backbone), sys_config)
        load_checkpoint(model_distill, None, distill_path)
        distill_result = run_ablation("Proposed-With-Distill", model_distill, sys_config, train_config, AblationConfig())
        results.append(distill_result)
    else:
        print(f"   Checkpoint not found: {distill_checkpoint}")
        print("   Train with: python -m src.train_dynamic_rho --use-distill")
        print("   Skipping this ablation.")
    
    # 3. Proposed-Frozen-WC (trained variant)
    print("\n3. Proposed-Frozen-WC")
    frozen_checkpoint = get_checkpoint_name(
        alpha_chatter=train_config.alpha_chatter,
        lambda_trade=train_config.lambda_trade,
        seed=train_config.seed,
        frozen_wc=True
    )
    frozen_path = base_path / 'checkpoints' / frozen_checkpoint
    
    if frozen_path.exists():
        print(f"   Found checkpoint: {frozen_checkpoint}")
        model_frozen = ISACModel(deepcopy(backbone), sys_config, frozen_wc=True)
        load_checkpoint(model_frozen, None, frozen_path)
        frozen_result = run_ablation("Proposed-Frozen-WC", model_frozen, sys_config, train_config, AblationConfig())
        results.append(frozen_result)
    else:
        print(f"   Checkpoint not found: {frozen_checkpoint}")
        print("   Train with: python -m src.train_dynamic_rho --frozen-wc")
        print("   Skipping this ablation.")
    
    # 4. No-Dual-Stream
    print("\n4. No-Dual-Stream")
    no_dual_config = AblationConfig(no_dual_stream=True)
    no_dual_result = run_ablation("No-Dual-Stream", model, sys_config, train_config, no_dual_config)
    results.append(no_dual_result)
    
    # 5. No-Anti-Chatter (trained variant with α=0)
    print("\n5. No-Anti-Chatter")
    no_chatter_checkpoint = get_checkpoint_name(
        alpha_chatter=0.0,
        lambda_trade=train_config.lambda_trade,
        seed=train_config.seed
    )
    no_chatter_path = base_path / 'checkpoints' / no_chatter_checkpoint
    
    if no_chatter_path.exists():
        print(f"   Found checkpoint: {no_chatter_checkpoint}")
        model_no_chatter = ISACModel(deepcopy(backbone), sys_config)
        load_checkpoint(model_no_chatter, None, no_chatter_path)
        no_chatter_result = run_ablation("No-Anti-Chatter", model_no_chatter, sys_config, train_config, AblationConfig())
        results.append(no_chatter_result)
    else:
        print(f"   Checkpoint not found: {no_chatter_checkpoint}")
        print("   Train with: python -m src.train_dynamic_rho --alpha 0.0")
        print("   Or run: python -m scripts.compare_anti_chatter")
        print("   Skipping this ablation.")
    
    # 6. Fixed-ρ
    print("\n6. Fixed-ρ")
    fixed_rho_config = AblationConfig(fixed_rho=True, fixed_rho_value=0.35)
    fixed_rho_result = run_ablation("Fixed-ρ", model, sys_config, train_config, fixed_rho_config)
    results.append(fixed_rho_result)
    
    # 7. Oracle-ρ (Upper Bound)
    print("\n7. Oracle-ρ (Upper Bound)")
    oracle_result = run_oracle_rho_analysis(model, sys_config, train_config, abl_config, num_batches=20)
    results.append(oracle_result)
    
    # Save results
    results_df = pd.DataFrame(results)
    save_metrics_csv(results_df, base_path / 'ablations.csv')
    
    print("\n" + "="*60)
    print("Ablation Results:")
    print("="*60)
    print(results_df.to_string(index=False))
    print("="*60)
    print(f"\nResults saved to: {base_path / 'ablations.csv'}")


if __name__ == '__main__':
    main()