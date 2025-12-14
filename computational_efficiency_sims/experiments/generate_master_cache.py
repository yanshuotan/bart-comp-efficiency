"""
Generate Master CSV Cache from Pickle Files

This script scans all experiment artifact directories, loads pickle files,
and creates a master CSV containing all experiment results for easier sharing
and faster plot generation.

Usage:
    python experiments/generate_master_cache.py
    
The master CSV will be saved to: experiments/master_results_cache.csv
"""

import os
import pickle
import re
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from omegaconf import OmegaConf
from typing import Dict, List

# Setup logger
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define all experiment configurations
EXPERIMENT_CONFIGS = {
    "Initialization": "consolidated_configs/initialization.yaml",
    "Dirichlet": "consolidated_configs/dirichlet.yaml",
    "Thresholds": "consolidated_configs/thresholds.yaml",
    "Marginalized": "consolidated_configs/marginalized.yaml",
    "Temperature": "consolidated_configs/temperature.yaml",
    "Schedule": "consolidated_configs/schedule.yaml",
    "Trees": "consolidated_configs/trees.yaml",
    "Burn-in": "consolidated_configs/burnin.yaml",
    "ProposalMoves": "consolidated_configs/proposal_moves.yaml",
}



def load_pickle_file(pkl_path: Path) -> Dict:
    """Load a single pickle file and return its contents."""
    try:
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        LOGGER.error(f"Error loading {pkl_path}: {e}")
        return None


def extract_metadata_from_path(pkl_path: Path) -> Dict:
    """
    Extract metadata from pickle file path.
    Expected pattern: .../dgp/seed_X/ntrain_Y/results_variation.pkl
    """
    path_pattern = re.compile(
        r".*/(.*?)/seed_(\d+)/ntrain_(\d+)/(.*?)\.pkl"
    )
    
    match = path_pattern.match(str(pkl_path))
    if not match:
        LOGGER.warning(f"Could not parse path: {pkl_path}")
        return None
    
    dgp, seed, n_train, result_key = match.groups()
    variation = result_key.replace("results_", "")
    
    return {
        "dgp": dgp,
        "seed": int(seed),
        "n_train": int(n_train),
        "variation": variation,
    }


def scan_artifact_directory(artifact_dir: Path, experiment_name: str) -> List[Dict]:
    """
    Scan an artifact directory for all pickle files and extract results.
    
    Args:
        artifact_dir: Path to the artifact directory
        experiment_name: Name of the experiment (e.g., "Temperature")
    
    Returns:
        List of dictionaries containing experiment results
    """
    results = []
    
    if not artifact_dir.exists():
        LOGGER.warning(f"Artifact directory does not exist: {artifact_dir}")
        return results
    
    LOGGER.info(f"Scanning {experiment_name} from {artifact_dir}")
    
    for pkl_file in artifact_dir.glob("**/*.pkl"):
        # Extract metadata from path
        metadata = extract_metadata_from_path(pkl_file)
        if metadata is None:
            continue
        
        # Load pickle data
        data = load_pickle_file(pkl_file)
        if data is None:
            continue
        
        # Combine metadata with results
        result_row = {
            "experiment_name": experiment_name,
            **metadata,
            # Credible interval metrics
            "rmse_credible": data.get("rmse_credible", np.nan),
            "coverage_credible": data.get("coverage_credible", np.nan),
            "gr_rmse_credible": data.get("gr_rmse_credible", np.nan),
            "ess_rmse_credible": data.get("ess_rmse_credible", np.nan),
            # Predictive interval metrics
            "rmse_predictive": data.get("rmse_predictive", np.nan),
            "coverage_predictive": data.get("coverage_predictive", np.nan),
            "gr_rmse_predictive": data.get("gr_rmse_predictive", np.nan),
            "ess_rmse_predictive": data.get("ess_rmse_predictive", np.nan),
            # Pred vs true metrics
            "rmse_pred_vs_true": data.get("rmse_pred_vs_true", np.nan),
            "coverage_pred_vs_true": data.get("coverage_pred_vs_true", np.nan),
            "gr_rmse_pred_vs_true": data.get("gr_rmse_pred_vs_true", np.nan),
            "ess_rmse_pred_vs_true": data.get("ess_rmse_pred_vs_true", np.nan),
        }
        
        results.append(result_row)
    
    LOGGER.info(f"  Found {len(results)} results for {experiment_name}")
    return results


def generate_master_cache(output_path: str = None) -> pd.DataFrame:
    """
    Generate master CSV cache from all experiment pickle files.
    
    Args:
        output_path: Path where to save the CSV. If None, saves to experiments/master_results_cache.csv
    
    Returns:
        DataFrame containing all experiment results
    """
    script_dir = Path(__file__).parent
    
    if output_path is None:
        output_path = script_dir / "master_results_cache.csv"
    else:
        output_path = Path(output_path)
    
    all_results = []
    
    LOGGER.info("=" * 80)
    LOGGER.info("Starting Master Cache Generation")
    LOGGER.info("=" * 80)
    
    # Scan each experiment (main configs)
    for exp_name, config_rel_path in EXPERIMENT_CONFIGS.items():
        config_path = script_dir / config_rel_path
        
        if not config_path.exists():
            LOGGER.warning(f"Config not found for {exp_name}: {config_path}")
            continue
        
        # Load config to get artifact directory
        try:
            exp_cfg = OmegaConf.load(config_path)
            artifact_dir = Path(exp_cfg.artifacts_dir)
            
            # Scan and collect results
            exp_results = scan_artifact_directory(artifact_dir, exp_name)
            all_results.extend(exp_results)
            
        except Exception as e:
            LOGGER.error(f"Error processing {exp_name}: {e}")
            continue
    
    # Create DataFrame
    LOGGER.info("=" * 80)
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Sort for consistency
        df = df.sort_values(['experiment_name', 'dgp', 'n_train', 'variation', 'seed']).reset_index(drop=True)
        
        # Save to CSV
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        LOGGER.info(f"Master cache saved to: {output_path}")
        LOGGER.info(f"Total rows: {len(df)}")
        LOGGER.info(f"Experiments: {df['experiment_name'].nunique()}")
        LOGGER.info(f"DGPs: {df['dgp'].nunique()}")
        LOGGER.info(f"Unique variations: {df['variation'].nunique()}")
        LOGGER.info("=" * 80)
        
        # Print summary by experiment
        print("\nSummary by Experiment:")
        print(df.groupby('experiment_name').size().sort_index())
        
        return df
    else:
        LOGGER.warning("No results found!")
        return pd.DataFrame()


def update_cache_for_experiment(experiment_name: str, output_path: str = None):
    """
    Update the master cache by re-scanning a specific experiment.
    
    This is useful when you've added new results for one experiment
    and want to update the cache without re-scanning everything.
    
    Args:
        experiment_name: Name of experiment to update (e.g., "Temperature")
        output_path: Path to the master CSV cache
    """
    script_dir = Path(__file__).parent
    
    if output_path is None:
        output_path = script_dir / "master_results_cache.csv"
    else:
        output_path = Path(output_path)
    
    # Load existing cache
    if output_path.exists():
        df_existing = pd.read_csv(output_path)
        LOGGER.info(f"Loaded existing cache with {len(df_existing)} rows")
        
        # Remove old results for this experiment
        df_existing = df_existing[df_existing['experiment_name'] != experiment_name]
        LOGGER.info(f"Removed old {experiment_name} results, {len(df_existing)} rows remaining")
    else:
        df_existing = pd.DataFrame()
        LOGGER.info("No existing cache found, creating new one")
    
    # Get new results for this experiment
    config_path = script_dir / EXPERIMENT_CONFIGS.get(experiment_name)
    if not config_path.exists():
        LOGGER.error(f"Config not found for {experiment_name}: {config_path}")
        return
    
    exp_cfg = OmegaConf.load(config_path)
    artifact_dir = Path(exp_cfg.artifacts_dir)
    exp_results = scan_artifact_directory(artifact_dir, experiment_name)
    
    if exp_results:
        df_new = pd.DataFrame(exp_results)
        
        # Combine with existing
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined = df_combined.sort_values(['experiment_name', 'dgp', 'n_train', 'variation', 'seed']).reset_index(drop=True)
        
        # Save
        df_combined.to_csv(output_path, index=False)
        LOGGER.info(f"Updated cache saved to: {output_path}")
        LOGGER.info(f"Total rows: {len(df_combined)}")
        LOGGER.info(f"New {experiment_name} rows: {len(df_new)}")
    else:
        LOGGER.warning(f"No results found for {experiment_name}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Update specific experiment
        exp_name = sys.argv[1]
        if exp_name in EXPERIMENT_CONFIGS:
            LOGGER.info(f"Updating cache for {exp_name} only")
            update_cache_for_experiment(exp_name)
        else:
            LOGGER.error(f"Unknown experiment: {exp_name}")
            LOGGER.info(f"Available experiments: {list(EXPERIMENT_CONFIGS.keys())}")
    else:
        # Generate full cache
        generate_master_cache()


