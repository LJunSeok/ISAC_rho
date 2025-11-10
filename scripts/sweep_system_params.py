"""
Sweep system parameters (M, K, Ptx) and compare methods.
Each M and K value uses its own trained model.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy

from src.configs import SystemConfig, ModelConfig, TrainingConfig
from src.utils.seed import set_seed
from src.utils.io import load_checkpoint, get_checkpoint_name, save_metrics_csv, load_config
from src.sim.echoes import generate_echo_history, apply_duration_normalization
from src.models.backbone_dual import DualStreamBackbone
from src.models.heads import ISACModel
from src.baselines.ekf_beamforming import EKFBaseline
from src.train_dynamic_rho import oracle_rho_grid_search
from src.utils.metrics import batch_metrics
from typing import Dict
import json


def load_model_for_params(M: int = None, K: int = None, base_path: Path = Path("results")):
    """
    Load model trained for specific M or K value.
    
    Args:
        M: Number of sensing targets (if sweeping M)
        K: Number of communication users (if sweeping K)
        base_path: Results directory
    
    Returns:
        (model, sys_config, train_config) or (None, None, None) if failed
    """
    import dataclasses
    from typing import get_origin
    
    # Determine suffix and config path
    if M is not None:
        config_path = base_path / "sweep_configs" / f"sys_config_M{M}.json"
        suffix = f"_M{M}"
    elif K is not None:
        config_path = base_path / "sweep_configs" / f"sys_config_K{K}.json"
        suffix = f"_K{K}"
    else:
        # Default model
        suffix = ""
        config_path = None
    
    # Load system config
    if config_path and config_path.exists():
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Filter out non-init fields (like wavelength)
        init_fields = {f.name for f in dataclasses.fields(SystemConfig) if f.init}
        filtered_dict = {k: v for k, v in config_dict.items() if k in init_fields}
        
        # Convert lists back to tuples where needed
        for field in dataclasses.fields(SystemConfig):
            if field.name in filtered_dict:
                origin = get_origin(field.type)
                if origin is tuple:
                    if isinstance(filtered_dict[field.name], list):
                        filtered_dict[field.name] = tuple(filtered_dict[field.name])
        
        sys_config = SystemConfig(**filtered_dict)
    else:
        sys_config = SystemConfig()
        if M is not None:
            sys_config.M = M
        if K is not None:
            sys_config.K = K
    
    model_config = ModelConfig()
    train_config = TrainingConfig()
    
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
    checkpoint_name = get_checkpoint_name(
        alpha_chatter=train_config.alpha_chatter,
        lambda_trade=train_config.lambda_trade,
        seed=train_config.seed
    )
    
    # Add suffix
    if suffix:
        checkpoint_name = checkpoint_name.replace('.pt', f'{suffix}.pt')
    
    checkpoint_path = base_path / 'checkpoints' / checkpoint_name
    
    if checkpoint_path.exists():
        load_checkpoint(model, None, checkpoint_path)
        print(f"  Loaded: {checkpoint_name}")
        return model, sys_config, train_config
    else:
        print(f"  WARNING: Checkpoint not found: {checkpoint_name}")
        print(f"           Expected at: {checkpoint_path}")
        print(f"  Train with: python -m scripts.train_sweep_models")
        return None, None, None

def evaluate_method_single(
    method_name: str,
    model: ISACModel,
    sys_config: SystemConfig,
    train_config: TrainingConfig,
    num_batches: int = 20
) -> Dict[str, float]:
    """Evaluate a single method with given model and config."""
    
    # Safety check
    if train_config is None:
        return {'sum_rate': 0.0, 'crb_trace': 0.0, 'rho_mean': 0.0}
    
    if model is not None:
        model.eval()
    
    metrics_list = []
    
    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc=f"    {method_name}", leave=False):
            echo_data = generate_echo_history(sys_config, train_config.batch_size, sys_config.tau)
            
            if method_name == "EKF Baseline":
                ekf = EKFBaseline(sys_config)
                metrics = ekf.evaluate(
                    echo_data['H_comm'],
                    echo_data['H_sense'],
                    echo_data['target_params']
                )
                metrics_list.append(metrics)
                
            elif method_name == "Oracle-ρ":
                if model is None:
                    continue
                sensing_norm, comm_norm = apply_duration_normalization(
                    echo_data['sensing_echoes'],
                    echo_data['comm_echoes'],
                    echo_data['rho_prev']
                )
                outputs = model(sensing_norm, comm_norm, echo_data['rho_prev'])
                
                oracle_rho = oracle_rho_grid_search(
                    model, echo_data, sys_config, train_config.lambda_trade
                )
                outputs['rho'] = oracle_rho
                
                metrics = batch_metrics(
                    echo_data['H_comm'], echo_data['H_sense'],
                    outputs['F_S'], outputs['F_C'], outputs['W_C'], outputs['rho'],
                    echo_data['target_params'], sys_config, train_config.lambda_trade
                )
                metrics_list.append(metrics)
                
            elif method_name == "Fixed-ρ":
                if model is None:
                    continue
                sensing_norm, comm_norm = apply_duration_normalization(
                    echo_data['sensing_echoes'],
                    echo_data['comm_echoes'],
                    echo_data['rho_prev']
                )
                outputs = model(sensing_norm, comm_norm, echo_data['rho_prev'])
                outputs['rho'] = torch.full_like(outputs['rho'], 0.35)
                
                metrics = batch_metrics(
                    echo_data['H_comm'], echo_data['H_sense'],
                    outputs['F_S'], outputs['F_C'], outputs['W_C'], outputs['rho'],
                    echo_data['target_params'], sys_config, train_config.lambda_trade
                )
                metrics_list.append(metrics)
                
            else:  # Proposed Method
                if model is None:
                    continue
                sensing_norm, comm_norm = apply_duration_normalization(
                    echo_data['sensing_echoes'],
                    echo_data['comm_echoes'],
                    echo_data['rho_prev']
                )
                outputs = model(sensing_norm, comm_norm, echo_data['rho_prev'])
                
                metrics = batch_metrics(
                    echo_data['H_comm'], echo_data['H_sense'],
                    outputs['F_S'], outputs['F_C'], outputs['W_C'], outputs['rho'],
                    echo_data['target_params'], sys_config, train_config.lambda_trade
                )
                metrics_list.append(metrics)
    
    if not metrics_list:
        return {'sum_rate': 0.0, 'crb_trace': 0.0, 'rho_mean': 0.0}
    
    # Aggregate
    metrics_df = pd.DataFrame(metrics_list)
    return {
        'sum_rate': metrics_df['sum_rate'].mean(),
        'crb_trace': metrics_df['crb_trace'].mean(),
        'rho_mean': metrics_df['rho_mean'].mean(),
    }

def run_M_sweep():
    """Sweep number of sensing targets M."""
    print("\n" + "="*60)
    print("Sweeping M (Number of Sensing Targets)")
    print("="*60)
    
    M_values = [4, 8, 12, 16, 20, 24]
    methods = ["Proposed Method", "Fixed-ρ", "Oracle-ρ", "EKF Baseline"]
    base_path = Path("results")
    
    # Run sweep
    results = {method: {'sum_rate': [], 'crb_trace': [], 'M_values': []} for method in methods}
    
    for M in M_values:
        print(f"\nM = {M}")
        
            
        # Special case: M=16 uses baseline checkpoint
        if M == 16:
            print("  Using baseline checkpoint (M=16 is default)")
            checkpoint_name = get_checkpoint_name(
                alpha_chatter=0.05,
                lambda_trade=0.5,
                seed=42
            )
            checkpoint_path = base_path / 'checkpoints' / checkpoint_name
            
            if checkpoint_path.exists():
                sys_config = SystemConfig()  # M=16 by default
                model_config = ModelConfig()
                train_config = TrainingConfig()
                
                input_dim_sense = sys_config.M * sys_config.Nr * sys_config.Nt * 2
                input_dim_comm = sys_config.K * sys_config.Nr * sys_config.Nt * 2
                
                backbone = DualStreamBackbone(
                    input_dim_sense, input_dim_comm,
                    model_config.cnn_channels, model_config.lstm_hidden,
                    model_config.lstm_layers, model_config.attn_heads,
                    model_config.dropout
                )
                
                model = ISACModel(backbone, sys_config)
                load_checkpoint(model, None, checkpoint_path)
                print(f"  Loaded: {checkpoint_name}")
            else:
                print(f"  ERROR: Baseline not found")
                model = None
                sys_config = None
                train_config = None
        else:
            # Load model trained for this M
            model, sys_config, train_config = load_model_for_params(M=M, base_path=base_path)
        
        # Skip this M value if model not found (except for EKF which doesn't need a model)
        if model is None:
            print(f"  Model not found. Only evaluating EKF Baseline.")
            # Still evaluate EKF
            if "EKF Baseline" in methods:
                # Create configs manually for EKF
                sys_config = SystemConfig()
                sys_config.M = M
                train_config = TrainingConfig()
                
                ekf_metrics = evaluate_method_single("EKF Baseline", None, sys_config, train_config, num_batches=20)
                results["EKF Baseline"]['sum_rate'].append(ekf_metrics['sum_rate'])
                results["EKF Baseline"]['crb_trace'].append(ekf_metrics['crb_trace'])
                results["EKF Baseline"]['M_values'].append(M)
            continue
        
        # Evaluate all methods
        for method in methods:
            metrics = evaluate_method_single(method, model, sys_config, train_config, num_batches=20)
            results[method]['sum_rate'].append(metrics['sum_rate'])
            results[method]['crb_trace'].append(metrics['crb_trace'])
            results[method]['M_values'].append(M)
    
    # Save results
    for method in methods:
        if len(results[method]['sum_rate']) > 0:
            df = pd.DataFrame({
                'M': results[method]['M_values'],
                'sum_rate': results[method]['sum_rate'],
                'crb_trace': results[method]['crb_trace']
            })
            filename = f"sweep_M_{method.replace(' ', '_').replace('-', '_').lower()}.csv"
            save_metrics_csv(df, base_path / filename)
            print(f"  Saved: {filename}")
    
    print(f"\n✓ M sweep complete. Results saved to results/sweep_M_*.csv")


def run_K_sweep():
    """Sweep number of communication users K."""
    print("\n" + "="*60)
    print("Sweeping K (Number of Communication Users)")
    print("="*60)
    
    K_values = [2, 4, 6, 8, 10]
    methods = ["Proposed Method", "Fixed-ρ", "Oracle-ρ", "EKF Baseline"]
    base_path = Path("results")
    
    # Run sweep
    results = {method: {'sum_rate': [], 'crb_trace': [], 'K_values': []} for method in methods}
    
    for K in K_values:
        print(f"\nK = {K}")
        
        if K == 4:
            print("  Using baseline checkpoint (K=4 is default)")
            checkpoint_name = get_checkpoint_name(
                alpha_chatter=0.05,
                lambda_trade=0.5,
                seed=42
            )
            checkpoint_path = base_path / 'checkpoints' / checkpoint_name
            
            if checkpoint_path.exists():
                sys_config = SystemConfig()
                model_config = ModelConfig()
                train_config = TrainingConfig()
                
                input_dim_sense = sys_config.M * sys_config.Nr * sys_config.Nt * 2
                input_dim_comm = sys_config.K * sys_config.Nr * sys_config.Nt * 2
                
                backbone = DualStreamBackbone(
                    input_dim_sense, input_dim_comm,
                    model_config.cnn_channels, model_config.lstm_hidden,
                    model_config.lstm_layers, model_config.attn_heads,
                    model_config.dropout
                )
                
                model = ISACModel(backbone, sys_config)
                load_checkpoint(model, None, checkpoint_path)
                print(f"  Loaded: {checkpoint_name}")
            else:
                print(f"  ERROR: Baseline not found")
                model = None
                sys_config = None
                train_config = None
        else:
            # Load model trained for this K
            model, sys_config, train_config = load_model_for_params(K=K, base_path=base_path)
        
        # Skip this K value if model not found (except for EKF)
        if model is None:
            print(f"  Model not found. Only evaluating EKF Baseline.")
            if "EKF Baseline" in methods:
                # Create configs manually for EKF
                sys_config = SystemConfig()
                sys_config.K = K
                train_config = TrainingConfig()
                
                ekf_metrics = evaluate_method_single("EKF Baseline", None, sys_config, train_config, num_batches=20)
                results["EKF Baseline"]['sum_rate'].append(ekf_metrics['sum_rate'])
                results["EKF Baseline"]['crb_trace'].append(ekf_metrics['crb_trace'])
                results["EKF Baseline"]['K_values'].append(K)
            continue
        
        # Evaluate all methods
        for method in methods:
            metrics = evaluate_method_single(method, model, sys_config, train_config, num_batches=20)
            results[method]['sum_rate'].append(metrics['sum_rate'])
            results[method]['crb_trace'].append(metrics['crb_trace'])
            results[method]['K_values'].append(K)
    
    # Save results
    for method in methods:
        if len(results[method]['sum_rate']) > 0:
            df = pd.DataFrame({
                'K': results[method]['K_values'],
                'sum_rate': results[method]['sum_rate'],
                'crb_trace': results[method]['crb_trace']
            })
            filename = f"sweep_K_{method.replace(' ', '_').replace('-', '_').lower()}.csv"
            save_metrics_csv(df, base_path / filename)
            print(f"  Saved: {filename}")
    
    print(f"\n✓ K sweep complete. Results saved to results/sweep_K_*.csv")

def run_Ptx_sweep():
    """Sweep transmit power (uses single default model)."""
    print("\n" + "="*60)
    print("Sweeping Ptx (Transmit Power)")
    print("="*60)
    
    Ptx_values = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    methods = ["Proposed Method", "Fixed-ρ", "Oracle-ρ", "EKF Baseline"]
    base_path = Path("results")
    
    # Load default model (M=16, K=4)
    model, base_sys_config, train_config = load_model_for_params(K=4, M=16, base_path=base_path)
    
    if model is None:
        print("ERROR: Default model not found. Train with: python -m src.train_dynamic_rho")
        return
    
    # Run sweep
    results = {method: {'sum_rate': [], 'crb_trace': []} for method in methods}
    
    for Ptx in Ptx_values:
        print(f"\nPtx = {Ptx}")
        sys_config = deepcopy(base_sys_config)
        sys_config.tx_power = Ptx
        
        for method in methods:
            metrics = evaluate_method_single(method, model, sys_config, train_config, num_batches=20)
            results[method]['sum_rate'].append(metrics['sum_rate'])
            results[method]['crb_trace'].append(metrics['crb_trace'])
    
    # Save results
    for method in methods:
        df = pd.DataFrame({
            'Ptx': Ptx_values,
            'sum_rate': results[method]['sum_rate'],
            'crb_trace': results[method]['crb_trace']
        })
        filename = f"sweep_Ptx_{method.replace(' ', '_').replace('-', '_').lower()}.csv"
        save_metrics_csv(df, base_path / filename)
    
    print(f"\n✓ Ptx sweep complete. Results saved to results/sweep_Ptx_*.csv")


def main():
    set_seed(42)
    
    print("="*60)
    print("System Parameter Sweeps")
    print("="*60)
    print("NOTE: M and K sweeps require pre-trained models")
    print("  Run: python -m scripts.train_sweep_models")
    print("="*60)
    
    # Run all sweeps
    #run_M_sweep()
    #run_K_sweep()
    run_Ptx_sweep()
    
    print("\n" + "="*60)
    print("All sweeps complete!")
    print("="*60)
    print("Run: python -m src.plot_analysis")


if __name__ == '__main__':
    main()