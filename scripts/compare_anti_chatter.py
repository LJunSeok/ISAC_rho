"""
Compare models trained with different α (anti-chatter) values.

Expected result: α=0.05 should produce LOWER ρ variance (smoother transitions).
"""

import subprocess
from pathlib import Path
import pandas as pd


def train_and_evaluate(alpha_value, lambda_trade=0.5, seed=42):
    """Train a model with specific α value and return metrics."""
    print(f"\n{'='*60}")
    print(f"Training with α={alpha_value}")
    print(f"{'='*60}")
    
    from src.utils.io import get_checkpoint_name, get_metrics_name
    checkpoint_name = get_checkpoint_name(
        alpha_chatter=alpha_value,
        lambda_trade=lambda_trade,
        seed=seed
    )
    print(f"Checkpoint will be: {checkpoint_name}")

    # Train
    result = subprocess.run([
        "python", "-m", "src.train_dynamic_rho",
        "--alpha", str(alpha_value),
        "--lambda", str(lambda_trade),
        "--epochs", "20",
        "--seed", str(seed)
    ], capture_output=False)
    
    if result.returncode != 0:
        print(f"ERROR: Training failed for α={alpha_value}")
        return None
    
    # Evaluate
    result = subprocess.run([
        "python", "-m", "src.evaluate",
        "--checkpoint", checkpoint_name
    ], capture_output=False)
    
    if result.returncode != 0:
        print(f"ERROR: Evaluation failed for α={alpha_value}")
        return None
    
    # Load results
    metrics_name = get_metrics_name(checkpoint_name, 'test_metrics')
    metrics_path = Path("results") / metrics_name
    
    if not metrics_path.exists():
        print(f"ERROR: Metrics not found: {metrics_path}")
        return None
    
    metrics_df = pd.read_csv(metrics_path)
    
    # Extract key metrics
    metrics = {
        'alpha': alpha_value,
        'sum_rate': metrics_df['sum_rate'].mean(),
        'crb_trace': metrics_df['crb_trace'].mean(),
        'rho_mean': metrics_df['rho_mean'].mean(),
        'rho_std': metrics_df['rho_std'].mean(),
        'checkpoint': checkpoint_name
    }
    
    return metrics


def main():
    print("="*60)
    print("Anti-Chatter Ablation Study")
    print("="*60)
    print("Hypothesis: Higher α → Lower ρ variance (smoother transitions)")
    print("="*60)
    
    results = []
    seed = 42  # Use consistent seed
    
    # Test α=0.0 (no anti-chatter)
    print("\n[1/3] Training α=0.0 (No Anti-Chatter)...")
    metrics_0 = train_and_evaluate(alpha_value=0.0, seed=seed)
    if metrics_0:
        results.append(metrics_0)
    
    # Test α=0.01 (weak anti-chatter)
    print("\n[2/3] Training α=0.01 (Weak Anti-Chatter)...")
    metrics_001 = train_and_evaluate(alpha_value=0.01, seed=seed)
    if metrics_001:
        results.append(metrics_001)
    
    # Test α=0.05 (strong anti-chatter)
    print("\n[3/3] Training α=0.05 (Strong Anti-Chatter)...")
    metrics_005 = train_and_evaluate(alpha_value=0.05, seed=seed)
    if metrics_005:
        results.append(metrics_005)
    
    # Save and display results
    if len(results) >= 2:
        results_df = pd.DataFrame(results)
        
        # Save to CSV for plotting
        output_path = Path("results") / "anti_chatter_comparison.csv"
        results_df.to_csv(output_path, index=False)
        
        print("\n" + "="*60)
        print("Anti-Chatter Comparison Results")
        print("="*60)
        print(results_df.to_string(index=False))
        print("="*60)
        
        print("\nAnalysis:")
        for i, row in results_df.iterrows():
            print(f"  α={row['alpha']:.2f}: ρ_std={row['rho_std']:.4f}, Rate={row['sum_rate']:.2f}, CRB={row['crb_trace']:.6f}")
        
        # Calculate improvements
        if len(results) > 1:
            baseline_std = results_df.iloc[0]['rho_std']
            print("\nVariance Reduction:")
            for i in range(1, len(results_df)):
                alpha = results_df.iloc[i]['alpha']
                current_std = results_df.iloc[i]['rho_std']
                reduction = (baseline_std - current_std) / baseline_std * 100
                print(f"  α={alpha:.2f} vs α=0.0: {reduction:+.1f}% (ρ_std: {baseline_std:.4f} → {current_std:.4f})")
        
        print(f"\n✓ Results saved to: {output_path}")
        print("✓ Run plot_analysis.py to generate visualization")
    else:
        print("\n✗ ERROR: Not enough results to compare")


if __name__ == '__main__':
    main()