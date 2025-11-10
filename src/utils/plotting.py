"""
Plotting utilities for analysis and visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd


def setup_matplotlib():
    """Configure matplotlib for publication-quality plots."""
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['figure.dpi'] = 100


def plot_pareto_curves_multi_lambda_real(
    lambda_configs: Dict[float, Dict],
    save_path: Path
):
    """
    Plot Pareto curves for multiple lambda values with REAL evaluation data.
    
    Args:
        lambda_configs: Dict mapping lambda -> {
            'all_points': DataFrame with all evaluated points,
            'pareto': DataFrame with Pareto frontier points,
            'dynamic': DataFrame with dynamic-rho metrics
        }
        save_path: Path to save figure
    """
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(11, 8))
    
    # Define colors for each lambda
    colors = {
        0.01: '#1f77b4',  # Blue (comm-focused)
        0.50: '#2ca02c',  # Green (balanced)
        0.99: '#d62728',  # Red (sensing-focused)
    }
    
    # Plot each lambda configuration
    for lambda_val in sorted(lambda_configs.keys()):
        data = lambda_configs[lambda_val]
        color = colors.get(lambda_val, 'gray')
        
        all_points = data['all_points']
        pareto_points = data['pareto']
        dynamic_metrics = data['dynamic']
        
        # Plot all evaluated fixed-rho points (small dots)
        ax.scatter(all_points['crb_trace'], all_points['sum_rate'], 
                  c=color, marker='o', s=30, alpha=0.3, edgecolors='none')
        
        # Plot Pareto frontier (connected line with larger dots)
        if len(pareto_points) > 0:
            ax.plot(pareto_points['crb_trace'], pareto_points['sum_rate'],
                   color=color, linestyle='-', linewidth=2.5, alpha=0.8,
                   label=f'Pareto (λ={lambda_val})')
            ax.scatter(pareto_points['crb_trace'], pareto_points['sum_rate'],
                      c=color, marker='o', s=100, alpha=0.9, 
                      edgecolors='black', linewidth=1)
        
        # Plot dynamic-rho point (star)
        dyn_rate = dynamic_metrics['sum_rate'].mean()
        dyn_crb = dynamic_metrics['crb_trace'].mean()
        ax.scatter([dyn_crb], [dyn_rate], c=color, marker='*', s=500,
                  label=f'Dynamic-ρ (λ={lambda_val})',
                  edgecolors='black', linewidth=2, zorder=10)
    
    # Invert x-axis so (0,0) is at bottom-right
    ax.invert_xaxis()
    
    # Labels and styling
    ax.set_xlabel('CRB Trace (sensing cost) ←', fontweight='bold', fontsize=13)
    ax.set_ylabel('Sum Rate (bits/s/Hz) →', fontweight='bold', fontsize=13)
    ax.set_title('Pareto Frontier: Dynamic-ρ vs Fixed-ρ\n(Real Evaluations, Multiple Trade-offs)', 
                fontweight='bold', fontsize=14)
    
    # Legend with better organization
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right', fontsize=10, 
             framealpha=0.95, ncol=1, markerscale=0.8)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add arrow annotation
    ax.annotate('Better\n(minimize CRB,\nmaximize Rate)', 
               xy=(0.98, 0.02), xycoords='axes fraction',
               fontsize=9, style='italic', alpha=0.6,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.2))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_rho_histogram(rho_values: np.ndarray, save_path: Path):
    """
    Plot histogram of learned ρ(n) values.
    
    Args:
        rho_values: Array of rho values
        save_path: Path to save figure
    """
    setup_matplotlib()
    fig, ax = plt.subplots()
    
    ax.hist(rho_values, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(rho_values.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {rho_values.mean():.3f}')
    ax.axvline(np.median(rho_values), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {np.median(rho_values):.3f}')
    
    ax.set_xlabel('ρ(n) - Sensing Time Fraction', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Distribution of Learned Time-Split ρ(n)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_rho_vs_dynamics(
    df: pd.DataFrame,
    save_path: Path
):
    """
    Plot ρ vs Doppler and ρ vs AoD-change scatter plots.
    
    Args:
        df: DataFrame with columns [rho, doppler_mean, aod_change_mean]
        save_path: Path to save figure
    """
    setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ρ vs Doppler
    ax = axes[0]
    scatter1 = ax.scatter(df['doppler_mean'].abs(), df['rho'],
                         c=df['rho'], cmap='viridis', alpha=0.6, s=30)
    
    # Trend line
    z = np.polyfit(df['doppler_mean'].abs(), df['rho'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df['doppler_mean'].abs().min(), df['doppler_mean'].abs().max(), 100)
    ax.plot(x_trend, p(x_trend), "r--", linewidth=2, label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
    
    ax.set_xlabel('|Doppler| (Hz)', fontweight='bold')
    ax.set_ylabel('ρ(n)', fontweight='bold')
    ax.set_title('Time-Split vs Doppler Shift', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax, label='ρ(n)')
    
    # ρ vs AoD-change
    ax = axes[1]
    scatter2 = ax.scatter(df['aod_change_mean'], df['rho'],
                         c=df['rho'], cmap='plasma', alpha=0.6, s=30)
    
    # Trend line
    z2 = np.polyfit(df['aod_change_mean'], df['rho'], 1)
    p2 = np.poly1d(z2)
    x_trend2 = np.linspace(df['aod_change_mean'].min(), df['aod_change_mean'].max(), 100)
    ax.plot(x_trend2, p2(x_trend2), "r--", linewidth=2, label=f'Trend: y={z2[0]:.3f}x+{z2[1]:.3f}')
    
    ax.set_xlabel('|ΔAoD| (radians)', fontweight='bold')
    ax.set_ylabel('ρ(n)', fontweight='bold')
    ax.set_title('Time-Split vs AoD Change', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax, label='ρ(n)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_ablation_bars(df: pd.DataFrame, save_path: Path):
    """
    Plot ablation study results as bar chart (2 subplots: Rate + CRB only).
    
    Args:
        df: DataFrame with columns [ablation, sum_rate, crb_trace]
        save_path: Path to save figure
    """
    setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Changed from 3 to 2
    
    ablations = df['ablation'].values
    x_pos = np.arange(len(ablations))
    
    # Sum rate
    ax = axes[0]
    bars = ax.bar(x_pos, df['sum_rate'], color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_ylabel('Sum Rate (bits/s/Hz)', fontweight='bold')
    ax.set_title('Communication Performance', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ablations, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Highlight baseline (first bar)
    bars[0].set_color('darkgreen')
    bars[0].set_alpha(0.9)
    
    # CRB trace
    ax = axes[1]
    bars = ax.bar(x_pos, df['crb_trace'], color='coral', alpha=0.7, edgecolor='black')
    ax.set_ylabel('CRB Trace', fontweight='bold')
    ax.set_title('Sensing Performance', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ablations, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    bars[0].set_color('darkred')
    bars[0].set_alpha(0.9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_curves(history: Dict[str, List[float]], save_path: Path):
    """
    Plot training curves (loss, rate, CRB over epochs).
    
    Args:
        history: Dict with keys [loss, sum_rate, crb_trace, rho_mean]
        save_path: Path to save figure
    """
    setup_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['loss'], 'b-', linewidth=2, marker='o')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Loss', fontweight='bold')
    ax.set_title('Training Loss', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Sum rate
    ax = axes[0, 1]
    ax.plot(epochs, history['sum_rate'], 'g-', linewidth=2, marker='s')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Sum Rate', fontweight='bold')
    ax.set_title('Communication Rate', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # CRB trace
    ax = axes[1, 0]
    ax.plot(epochs, history['crb_trace'], 'r-', linewidth=2, marker='^')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('CRB Trace', fontweight='bold')
    ax.set_title('Sensing Cost', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Rho mean
    ax = axes[1, 1]
    ax.plot(epochs, history['rho_mean'], 'm-', linewidth=2, marker='d')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Mean ρ', fontweight='bold')
    ax.set_title('Average Time-Split', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_crb_scaling_comparison(
    history_no_scale: Dict[str, List[float]],
    history_with_scale: Dict[str, List[float]],
    save_path: Path
):
    """
    Plot before/after CRB scaling comparison.
    
    Args:
        history_no_scale: Training history without CRB scaling
        history_with_scale: Training history with CRB scaling
        save_path: Path to save figure
    """
    setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs_no = range(1, len(history_no_scale['rho_mean']) + 1)
    epochs_yes = range(1, len(history_with_scale['rho_mean']) + 1)
    
    # ρ evolution
    ax = axes[0]
    ax.plot(epochs_no, history_no_scale['rho_mean'], 'r--', linewidth=2, marker='x', label='Without Rescaling')
    ax.plot(epochs_yes, history_with_scale['rho_mean'], 'g-', linewidth=2, marker='o', label='With Rescaling')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Mean ρ', fontweight='bold')
    ax.set_title('Effect of CRB Rescaling on ρ', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.01, color='gray', linestyle=':', label='Collapse threshold')
    
    # CRB improvement
    ax = axes[1]
    ax.plot(epochs_no, history_no_scale['crb_trace'], 'r--', linewidth=2, marker='x', label='Without Rescaling')
    ax.plot(epochs_yes, history_with_scale['crb_trace'], 'g-', linewidth=2, marker='o', label='With Rescaling')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('CRB Trace', fontweight='bold')
    ax.set_title('Sensing Performance', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Rate comparison
    ax = axes[2]
    ax.plot(epochs_no, history_no_scale['sum_rate'], 'r--', linewidth=2, marker='x', label='Without Rescaling')
    ax.plot(epochs_yes, history_with_scale['sum_rate'], 'g-', linewidth=2, marker='o', label='With Rescaling')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Sum Rate (bits/s/Hz)', fontweight='bold')
    ax.set_title('Communication Performance', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_rho_trajectory_comparison(
    history_no_chatter: Dict[str, List[float]],
    history_with_chatter: Dict[str, List[float]],
    save_path: Path
):
    """
    Plot ρ trajectory showing anti-chatter effect.
    
    Args:
        history_no_chatter: Training history with α=0
        history_with_chatter: Training history with α>0
        save_path: Path to save figure
    """
    setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs_no = range(1, len(history_no_chatter['rho_mean']) + 1)
    epochs_yes = range(1, len(history_with_chatter['rho_mean']) + 1)
    
    # ρ mean evolution
    ax = axes[0]
    ax.plot(epochs_no, history_no_chatter['rho_mean'], 'r--', linewidth=2, marker='x', label='α=0 (No Anti-Chatter)')
    ax.plot(epochs_yes, history_with_chatter['rho_mean'], 'g-', linewidth=2, marker='o', label='α=0.05 (With Anti-Chatter)')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Mean ρ', fontweight='bold')
    ax.set_title('ρ Trajectory Over Training', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ρ std evolution
    ax = axes[1]
    ax.plot(epochs_no, history_no_chatter['rho_std'], 'r--', linewidth=2, marker='x', label='α=0 (No Anti-Chatter)')
    ax.plot(epochs_yes, history_with_chatter['rho_std'], 'g-', linewidth=2, marker='o', label='α=0.05 (With Anti-Chatter)')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('ρ Std Dev', fontweight='bold')
    ax.set_title('Anti-Chatter Regularization Effect', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.fill_between(epochs_yes, 0, history_with_chatter['rho_std'], alpha=0.2, color='green', label='Lower variance')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_sweep_comparison(
    sweep_data: Dict[str, pd.DataFrame],
    param_name: str,
    param_values: List[float],
    save_path: Path,
    metric: str = 'sum_rate'
):
    """
    Plot performance vs system parameter sweep.
    
    Args:
        sweep_data: Dict mapping method_name -> DataFrame with sweep results
        param_name: Parameter name (e.g., 'M', 'K', 'Ptx')
        param_values: List of parameter values swept
        save_path: Path to save figure
        metric: Metric to plot ('sum_rate' or 'crb_trace')
    """
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define styles for each method
    styles = {
        'Proposed Method': {'color': 'darkgreen', 'marker': 'o', 'linestyle': '-', 'linewidth': 2.5},
        'Fixed-ρ': {'color': 'steelblue', 'marker': 's', 'linestyle': '--', 'linewidth': 2},
        'Oracle-ρ': {'color': 'orange', 'marker': '^', 'linestyle': '-.', 'linewidth': 2},
        'EKF Baseline': {'color': 'crimson', 'marker': 'd', 'linestyle': ':', 'linewidth': 2},
    }
    
    # Plot each method
    for method_name, df in sweep_data.items():
        if method_name in styles:
            style = styles[method_name]
            ax.plot(param_values, df[metric], 
                   label=method_name, 
                   marker=style['marker'],
                   color=style['color'],
                   linestyle=style['linestyle'],
                   linewidth=style['linewidth'],
                   markersize=8)
    
    # Labels
    param_labels = {
        'M': 'Number of Sensing Targets (M)',
        'K': 'Number of Communication Users (K)',
        'Ptx': 'Transmit Power (W)'
    }
    metric_labels = {
        'sum_rate': 'Sum Rate (bits/s/Hz)',
        'crb_trace': 'CRB Trace'
    }
    
    ax.set_xlabel(param_labels.get(param_name, param_name), fontweight='bold', fontsize=13)
    ax.set_ylabel(metric_labels.get(metric, metric), fontweight='bold', fontsize=13)
    
    title_map = {
        ('M', 'sum_rate'): 'Achievable Rate vs Number of Targets',
        ('M', 'crb_trace'): 'Sensing Accuracy vs Number of Targets',
        ('K', 'sum_rate'): 'Achievable Rate vs Number of Users',
        ('K', 'crb_trace'): 'Sensing Accuracy vs Number of Users',
        ('Ptx', 'sum_rate'): 'Achievable Rate vs Transmit Power',
        ('Ptx', 'crb_trace'): 'Sensing Accuracy vs Transmit Power',
    }
    
    ax.set_title(title_map.get((param_name, metric), f'{metric} vs {param_name}'), 
                fontweight='bold', fontsize=14)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Log scale for Ptx if applicable
    if param_name == 'Ptx':
        ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()