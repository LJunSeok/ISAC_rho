"""
Train models with different lambda values for Pareto comparison.
"""

import subprocess
from pathlib import Path


def train_lambda_model(lambda_val: float, seed: int = 42):
    """Train a model with specific lambda value."""
    print(f"\n{'='*60}")
    print(f"Training model for λ={lambda_val}")
    print(f"{'='*60}")
    
    result = subprocess.run([
        "python", "-m", "src.train_dynamic_rho",
        "--lambda", str(lambda_val),
        "--alpha", "0.05",
        "--epochs", "20",
        "--seed", str(seed)
    ])
    
    return result.returncode == 0


def main():
    print("="*60)
    print("Training Lambda Comparison Models")
    print("="*60)
    
    seed = 42
    lambda_values = [0.01, 0.99]  # 0.5 already trained
    
    for lambda_val in lambda_values:
        if train_lambda_model(lambda_val, seed):
            print(f"✓ λ={lambda_val} complete")
        else:
            print(f"✗ λ={lambda_val} failed")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("Next: python -m src.plot_analysis")


if __name__ == '__main__':
    main()