"""
Train models for system parameter sweeps.
"""

import subprocess
from pathlib import Path
from src.configs import SystemConfig, ModelConfig, TrainingConfig
from src.utils.io import get_checkpoint_name
import json
import dataclasses


def save_sys_config(sys_config: SystemConfig, filepath: Path):
    """Save system config to JSON, excluding computed fields."""
    # Get init fields only (exclude computed fields like wavelength)
    config_dict = {}
    for field in dataclasses.fields(sys_config):
        if field.init:  # Only include fields that are __init__ parameters
            value = getattr(sys_config, field.name)
            # Handle tuples (convert to list for JSON)
            if isinstance(value, tuple):
                value = list(value)
            config_dict[field.name] = value
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)


def train_for_M_value(M: int, seed: int = 42):
    """Train a model with specific M value."""
    print(f"\n{'='*60}")
    print(f"Training model for M={M}")
    print(f"{'='*60}")
    
    # Modify system config
    sys_config = SystemConfig()
    sys_config.M = M
    
    # Save modified config (excluding computed fields)
    config_dir = Path("results") / "sweep_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = config_dir / f"sys_config_M{M}.json"
    save_sys_config(sys_config, config_path)
    
    print(f"Saved config to: {config_path}")
    print(f"  M={M}, K={sys_config.K}, Nt={sys_config.Nt}, Nr={sys_config.Nr}")
    
    # Train with modified config
    result = subprocess.run([
        "python", "-m", "src.train_dynamic_rho",
        "--sys-config", str(config_path),
        "--alpha", "0.05",
        "--lambda", "0.5",
        "--epochs", "20",
        "--seed", str(seed),
        "--suffix", f"_M{M}"
    ])
    
    return result.returncode == 0


def train_for_K_value(K: int, seed: int = 42):
    """Train a model with specific K value."""
    print(f"\n{'='*60}")
    print(f"Training model for K={K}")
    print(f"{'='*60}")
    
    # Modify system config
    sys_config = SystemConfig()
    sys_config.K = K
    
    # Save modified config (excluding computed fields)
    config_dir = Path("results") / "sweep_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = config_dir / f"sys_config_K{K}.json"
    save_sys_config(sys_config, config_path)
    
    print(f"Saved config to: {config_path}")
    print(f"  M={sys_config.M}, K={K}, Nt={sys_config.Nt}, Nr={sys_config.Nr}")
    
    # Train with modified config
    result = subprocess.run([
        "python", "-m", "src.train_dynamic_rho",
        "--sys-config", str(config_path),
        "--alpha", "0.05",
        "--lambda", "0.5",
        "--epochs", "20",
        "--seed", str(seed),
        "--suffix", f"_K{K}"
    ])
    
    return result.returncode == 0


def main():
    print("="*60)
    print("Training Models for System Parameter Sweeps")
    print("="*60)
    print("This will train multiple models (takes ~2-3 hours)")
    print("="*60)
    
    seed = 42
    
    # Train for M sweep
    print("\n[1/2] Training models for M sweep...")
    M_values = [4, 8, 12, 20, 24]
    M_success = []
    
    for M in M_values:
        if train_for_M_value(M, seed):
            M_success.append(M)
            print(f"✓ M={M} complete")
        else:
            print(f"✗ M={M} failed")
    
    # Train for K sweep
    print("\n[2/2] Training models for K sweep...")
    K_values = [2, 6, 8, 10]
    K_success = []
    
    for K in K_values:
        if train_for_K_value(K, seed):
            K_success.append(K)
            print(f"✓ K={K} complete")
        else:
            print(f"✗ K={K} failed")
    
    # Summary
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    print(f"M sweep: {len(M_success)}/{len(M_values)} models trained")
    print(f"  Success: {M_success}")
    print(f"K sweep: {len(K_success)}/{len(K_values)} models trained")
    print(f"  Success: {K_success}")
    print("="*60)
    
    if len(M_success) == len(M_values) and len(K_success) == len(K_values):
        print("✓ All models trained successfully!")
        print("  Next: python -m scripts.sweep_system_params")
    else:
        print("✗ Some models failed. Check errors above.")


if __name__ == '__main__':
    main()