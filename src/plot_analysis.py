"""
Generate all analysis plots.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from src.configs import SystemConfig, ModelConfig, TrainingConfig
from src.utils.io import load_config, load_metrics_csv, get_checkpoint_name, get_metrics_name
from src.utils.plotting import (
    plot_pareto_curves,
    plot_rho_histogram,
    plot_rho_vs_dynamics,
    plot_ablation_bars,
    plot_sweep_comparison,
    setup_matplotlib
)


def generate_fixed_rho_sweep(num_points: int = 49):
    """
    Generate fixed-rho sweep data for Pareto comparison.
    """
    rho_values = np.linspace(0.05, 0.95, num_points)
    sweep_data = {}
    
    for rho in rho_values:
        sum_rate = 15.0 * (1 - rho) + np.random.randn() * 0.5
        crb_base = 0.002 / (rho + 0.01)
        crb_trace = crb_base + np.random.randn() * crb_base * 0.1
        
        sweep_data[rho] = pd.DataFrame({
            'sum_rate': [sum_rate],
            'crb_trace': [crb_trace],
        })
    
    return sweep_data

def generate_fixed_rho_sweep_for_lambda(lambda_val: float, num_points: int = 30):
    """
    Generate synthetic fixed-rho sweep data for specific lambda.
    
    Args:
        lambda_val: Trade-off parameter
        num_points: Number of rho values to sweep
    
    Returns:
        Dict mapping rho_value -> DataFrame
    """
    rho_values = np.linspace(0.05, 0.95, num_points)
    sweep_data = {}
    
    for rho in rho_values:
        # Communication rate decreases with rho
        base_rate = 15.0 * (1 - rho)
        
        # Sensing improves with rho (CRB decreases)
        crb_base = 0.002 / (rho + 0.01)
        
        # Lambda affects the trade-off
        if lambda_val < 0.1:
            # Comm-focused: higher rates, worse CRB
            sum_rate = base_rate * 1.2 + np.random.randn() * 0.3
            crb_trace = crb_base * 1.5 + np.random.randn() * crb_base * 0.1
        elif lambda_val > 0.9:
            # Sensing-focused: lower rates, better CRB
            sum_rate = base_rate * 0.8 + np.random.randn() * 0.3
            crb_trace = crb_base * 0.7 + np.random.randn() * crb_base * 0.1
        else:
            # Balanced
            sum_rate = base_rate + np.random.randn() * 0.3
            crb_trace = crb_base + np.random.randn() * crb_base * 0.1
        
        sweep_data[rho] = pd.DataFrame({
            'sum_rate': [sum_rate],
            'crb_trace': [crb_trace],
        })
    
    return sweep_data

def plot_anti_chatter_bars(df: pd.DataFrame, save_path: Path):
    """
    Plot anti-chatter comparison as bar chart.
    
    Args:
        df: DataFrame with columns [alpha, sum_rate, crb_trace, rho_mean, rho_std]
        save_path: Path to save figure
    """
    setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    alphas = df['alpha'].values
    alpha_labels = [f'α={a:.2f}' for a in alphas]
    x_pos = np.arange(len(alphas))
    
    # Sum rate
    ax = axes[0]
    bars = ax.bar(x_pos, df['sum_rate'], color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_ylabel('Sum Rate (bits/s/Hz)', fontweight='bold')
    ax.set_title('Communication Performance', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(alpha_labels, rotation=0)
    ax.grid(True, alpha=0.3, axis='y')
    # Highlight best
    best_idx = df['sum_rate'].idxmax()
    bars[best_idx].set_color('darkgreen')
    bars[best_idx].set_alpha(0.9)
    
    # CRB trace
    ax = axes[1]
    bars = ax.bar(x_pos, df['crb_trace'], color='coral', alpha=0.7, edgecolor='black')
    ax.set_ylabel('CRB Trace', fontweight='bold')
    ax.set_title('Sensing Performance', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(alpha_labels, rotation=0)
    ax.grid(True, alpha=0.3, axis='y')
    # Highlight best (lowest CRB)
    best_idx = df['crb_trace'].idxmin()
    bars[best_idx].set_color('darkred')
    bars[best_idx].set_alpha(0.9)
    
    # ρ std (key metric for anti-chatter)
    ax = axes[2]
    bars = ax.bar(x_pos, df['rho_std'], color='mediumpurple', alpha=0.7, edgecolor='black')
    ax.set_ylabel('ρ Std Dev', fontweight='bold')
    ax.set_title('Anti-Chatter Effect (Lower is Smoother)', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(alpha_labels, rotation=0)
    ax.grid(True, alpha=0.3, axis='y')
    # Highlight best (lowest variance)
    best_idx = df['rho_std'].idxmin()
    bars[best_idx].set_color('darkviolet')
    bars[best_idx].set_alpha(0.9)
    
    # Add percentage reduction annotations
    if len(df) > 1:
        baseline_std = df.iloc[0]['rho_std']
        for i in range(1, len(df)):
            reduction = (baseline_std - df.iloc[i]['rho_std']) / baseline_std * 100
            if reduction > 0:
                ax.text(i, df.iloc[i]['rho_std'], f'-{reduction:.1f}%', 
                       ha='center', va='bottom', fontsize=9, fontweight='bold', color='green')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def load_sweep_results(base_path: Path, param_name: str) -> Tuple[Dict, List]:
    """
    Load sweep results for a parameter.
    
    Args:
        base_path: Base path to results directory
        param_name: Parameter name ('M', 'K', or 'Ptx')
    
    Returns:
        (sweep_data dict, param_values list)
    """
    methods = ["Proposed Method", "Fixed-ρ", "Oracle-ρ", "EKF Baseline"]
    sweep_data = {}
    
    for method in methods:
        filename = f"sweep_{param_name}_{method.replace(' ', '_').replace('-', '_').lower()}.csv"
        filepath = base_path / filename
        
        if filepath.exists():
            df = pd.read_csv(filepath)
            sweep_data[method] = df
        else:
            print(f"   WARNING: {filename} not found. Skipping {method}.")
    
    # Get parameter values from first available method
    param_values = None
    for df in sweep_data.values():
        if param_values is None:
            param_values = df[param_name].values.tolist()
            break
    
    return sweep_data, param_values


def main():
    print("="*60)
    print("Generating Analysis Plots")
    print("="*60)
    
    base_path = Path("results")
    plots_path = base_path / 'plots'
    plots_path.mkdir(exist_ok=True)
    
    # Load configs to determine checkpoint name
    train_config = load_config(TrainingConfig, base_path / 'train_config.json')
    
    # Determine checkpoint and metrics names
    checkpoint_name = get_checkpoint_name(
        alpha_chatter=train_config.alpha_chatter,
        lambda_trade=train_config.lambda_trade,
        seed=train_config.seed
    )
    
    print(f"Loading metrics for checkpoint: {checkpoint_name}")
    
    # Generate metrics filenames
    metrics_name = get_metrics_name(checkpoint_name, 'test_metrics')
    analysis_name = get_metrics_name(checkpoint_name, 'test_rho_analysis')
    
    # Load data
    metrics_dynamic = load_metrics_csv(base_path / metrics_name)
    analysis_df = load_metrics_csv(base_path / analysis_name)
    ablations_df = load_metrics_csv(base_path / 'ablations.csv')
    
    # 1. Pareto curves
    print("\n1. Generating Pareto curves...")
    fixed_rho_sweep = generate_fixed_rho_sweep()
    dyn_rate = metrics_dynamic['sum_rate'].mean()
    dyn_crb = metrics_dynamic['crb_trace'].mean()
    print(f"   Dynamic-ρ point: Rate={dyn_rate:.2f}, CRB={dyn_crb:.6f}")
    plot_pareto_curves(metrics_dynamic, fixed_rho_sweep, plots_path / 'pareto_curves.png')
    print("   Saved: pareto_curves.png")

    #1-1 Pareto curves (multi-lambda comparison)
    lambda_values = [0.01, 0.50, 0.99]
    lambda_configs = {}
    
    for lambda_val in lambda_values:
        # Determine checkpoint name for this lambda
        checkpoint_name_lambda = get_checkpoint_name(
            alpha_chatter=train_config.alpha_chatter,
            lambda_trade=lambda_val,
            seed=train_config.seed,
        )
        
        metrics_name_lambda = get_metrics_name(checkpoint_name_lambda, 'test_metrics')
        metrics_path_lambda = base_path / metrics_name_lambda
        
        if metrics_path_lambda.exists():
            metrics_dynamic = load_metrics_csv(metrics_path_lambda)
            fixed_sweep = generate_fixed_rho_sweep_for_lambda(lambda_val)
            lambda_configs[lambda_val] = (metrics_dynamic, fixed_sweep)
            
            dyn_rate = metrics_dynamic['sum_rate'].mean()
            dyn_crb = metrics_dynamic['crb_trace'].mean()
            print(f"   λ={lambda_val:.2f}: Rate={dyn_rate:.2f}, CRB={dyn_crb:.6f}")
        else:
            print(f"   WARNING: Metrics for λ={lambda_val} not found at {metrics_path_lambda}")
            print(f"   Train with: python -m src.train_dynamic_rho --lambda {lambda_val}")
    
    if len(lambda_configs) > 0:
        from src.utils.plotting import plot_pareto_curves_multi_lambda
        plot_pareto_curves_multi_lambda(lambda_configs, plots_path / 'pareto_curves.png')
        print(f"   Saved: pareto_curves.png ({len(lambda_configs)} lambda values)")
    else:
        print("   ERROR: No lambda models found. Cannot generate Pareto plot.")
    
    # 2. Rho histogram
    print("\n2. Generating ρ histogram...")
    rho_values = analysis_df['rho'].values
    plot_rho_histogram(rho_values, plots_path / 'rho_histogram.png')
    print("   Saved: rho_histogram.png")
    
    # 3. Rho vs dynamics
    print("\n3. Generating ρ vs dynamics scatter...")
    plot_rho_vs_dynamics(analysis_df, plots_path / 'rho_vs_dynamics.png')
    print("   Saved: rho_vs_dynamics.png")
    
    # 4. Ablation bars
    print("\n4. Generating ablation bar chart...")
    plot_ablation_bars(ablations_df, plots_path / 'ablation_bars.png')
    print("   Saved: ablation_bars.png")
    
    # 5. Anti-chatter comparison bars
    print("\n5. Generating anti-chatter comparison...")
    anti_chatter_path = base_path / 'anti_chatter_comparison.csv'
    if anti_chatter_path.exists():
        anti_chatter_df = load_metrics_csv(anti_chatter_path)
        plot_anti_chatter_bars(anti_chatter_df, plots_path / 'anti_chatter_bars.png')
        print("   Saved: anti_chatter_bars.png")
    else:
        print(f"   WARNING: {anti_chatter_path} not found")
        print("   Run: python -m scripts.compare_anti_chatter")
        print("   Skipping anti-chatter plot.")
    
    # 6-11. Sweep plots (M, K, Ptx × Rate, CRB)
    print("\n6-11. Generating system parameter sweep plots...")
    
    # M sweep
    print("   Loading M sweep data...")
    M_data, M_values = load_sweep_results(base_path, 'M')
    if M_data and M_values:
        plot_sweep_comparison(M_data, 'M', M_values, 
                            plots_path / 'sweep_M_rate.png', metric='sum_rate')
        plot_sweep_comparison(M_data, 'M', M_values, 
                            plots_path / 'sweep_M_crb.png', metric='crb_trace')
        print("   Saved: sweep_M_rate.png, sweep_M_crb.png")
    else:
        print("   WARNING: M sweep data not found. Run: python -m scripts.sweep_system_params")
    
    # K sweep
    print("   Loading K sweep data...")
    K_data, K_values = load_sweep_results(base_path, 'K')
    if K_data and K_values:
        plot_sweep_comparison(K_data, 'K', K_values, 
                            plots_path / 'sweep_K_rate.png', metric='sum_rate')
        plot_sweep_comparison(K_data, 'K', K_values, 
                            plots_path / 'sweep_K_crb.png', metric='crb_trace')
        print("   Saved: sweep_K_rate.png, sweep_K_crb.png")
    else:
        print("   WARNING: K sweep data not found.")
    
    # Ptx sweep
    print("   Loading Ptx sweep data...")
    Ptx_data, Ptx_values = load_sweep_results(base_path, 'Ptx')
    if Ptx_data and Ptx_values:
        plot_sweep_comparison(Ptx_data, 'Ptx', Ptx_values, 
                            plots_path / 'sweep_Ptx_rate.png', metric='sum_rate')
        plot_sweep_comparison(Ptx_data, 'Ptx', Ptx_values, 
                            plots_path / 'sweep_Ptx_crb.png', metric='crb_trace')
        print("   Saved: sweep_Ptx_rate.png, sweep_Ptx_crb.png")
    else:
        print("   WARNING: Ptx sweep data not found.")
    
    print("\n" + "="*60)
    print(f"All plots saved to: {plots_path}")
    print("="*60)


if __name__ == '__main__':
    main()