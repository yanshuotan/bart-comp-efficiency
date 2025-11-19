import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import hashlib
import yaml
import pandas as pd
import wandb
import pmlb
from pmlb import classification_dataset_names, regression_dataset_names
from sklearn.utils import resample
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from bart_playground import DefaultBART, DefaultPreprocessor, DataGenerator
from bart_playground.samplers import default_proposal_probs, TemperatureSchedule

# Combine PMLB dataset names
pmlb_dataset_names = classification_dataset_names + regression_dataset_names

# Define the allowed real-world datasets
ALLOWED_REAL_DATASETS = [
    "1201_BNG_breastTumor",     # Breast tumor (Romano et al., 2020) - PMLB
    "1199_BNG_echoMonths",      # Echo months (Romano et al., 2020) - PMLB
    "294_satellite_image",      # Satellite image (PMLB)
    "california_housing"        # California housing (Pace & Barry, 1997) - Sklearn
]

# Setup logger
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Mixing Diagnostic Functions ---
def gelman_rubin(chains):
    if chains.ndim == 1 or chains.shape[0] < 2 or chains.shape[1] == 0:
        return np.nan
    m, n = chains.shape
    chain_means = np.mean(chains, axis=1)
    overall_mean = np.mean(chain_means)
    B = n / (m - 1) * np.sum((chain_means - overall_mean) ** 2)
    W = np.mean(np.var(chains, axis=1, ddof=1))
    if W == 0: return np.nan
    V_hat = ((n - 1) / n) * W + B / n
    return np.sqrt(V_hat / W)

def autocorrelation(chain, lag):
    n = len(chain)
    if lag >= n or n < 2 or np.all(chain == chain[0]):
        return np.nan
    return np.corrcoef(chain[:-lag], chain[lag:])[0, 1]

def effective_sample_size(chains, step=1):
    if chains.ndim == 1: chains = chains.reshape(1, -1)
    m, n = chains.shape
    if n == 0: return np.nan
    total_ess = 0.0
    for i in range(m):
        chain = chains[i, :]
        if np.all(chain == chain[0]):
            total_ess += np.nan
            continue
        ac_sum = 0.0
        for lag in range(1, n, step):
            ac = autocorrelation(chain, lag)
            if np.isnan(ac) or ac < 0: break
            ac_sum += step * ac
        ess_denominator = 1 + 2 * ac_sum
        total_ess += n / ess_denominator if ess_denominator > 0 else np.nan
    return total_ess if not np.isnan(total_ess) else np.nan

# --- W&B Logging ---
def log_wandb_artifacts(cfg: DictConfig, run_type: str, experiment_name: str, run_results: dict,
                        credible_preds_chains: np.ndarray, predictive_preds_chains: np.ndarray,
                        y_true_credible: np.ndarray, y_true_predictive: np.ndarray,
                        run_specific_artifact_dir: str, current_n_train: int):
    """Logs experiment artifacts to Weights & Biases."""

    dgp_name_safe = "".join(c if c.isalnum() else "_" for c in cfg.dgp)

    full_exp_name = (f"{run_type}_{dgp_name_safe}_{experiment_name}_ntrain_{current_n_train}_"
                     f"seed_{cfg.experiment_params.main_seed}")

    wandb.init(
        project="bart-playground",
        entity="bart_playground",
        name=full_exp_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=[run_type, cfg.dgp, experiment_name, f"seed_{cfg.experiment_params.main_seed}"],
        reinit=True
    )

    wandb_dir = os.path.join(run_specific_artifact_dir, f"wandb_artifacts_{experiment_name}")
    os.makedirs(wandb_dir, exist_ok=True)

    # Log true values
    y_true_credible_df = pd.DataFrame({"y_true": y_true_credible})
    y_true_credible_file = os.path.join(wandb_dir, "y_true_credible.csv")
    y_true_credible_df.to_csv(y_true_credible_file, index=False)

    y_true_predictive_df = pd.DataFrame({"y_true_noisy": y_true_predictive})
    y_true_predictive_file = os.path.join(wandb_dir, "y_true_predictive.csv")
    y_true_predictive_df.to_csv(y_true_predictive_file, index=False)

    # Log chains if enabled
    if cfg.experiment_params.get("save_full_chains", False):
        def save_chains(preds_chains, suffix):
            chain_files = []
            n_samples, n_chains, n_post = preds_chains.shape
            for chain_idx in range(n_chains):
                chain_preds = preds_chains[:, chain_idx, :]
                chain_df = pd.DataFrame(chain_preds, columns=[f"posterior_sample_{i}" for i in range(n_post)])
                chain_df["sample_index"] = np.arange(n_samples)
                chain_file = os.path.join(wandb_dir, f"predictions_{suffix}_chain_{chain_idx}.csv")
                chain_df.to_csv(chain_file, index=False)
                chain_files.append(chain_file)
            return chain_files

        credible_chain_files = save_chains(credible_preds_chains, "credible")
        predictive_chain_files = save_chains(predictive_preds_chains, "predictive")

    wandb.log({k: v for k, v in run_results.items() if 'samples' not in k and v is not None and not (isinstance(v, float) and np.isnan(v))})


    artifact = wandb.Artifact(
        name=f"{run_type}_results_{dgp_name_safe}_{experiment_name}_seed_{cfg.experiment_params.main_seed}_ntrain_{current_n_train}",
        type="experiment_results",
        description=f"BART {run_type} experiment for {cfg.dgp} with setting {experiment_name}"
    )

    config_file = os.path.join(wandb_dir, "config.yaml")
    with open(config_file, 'w') as f:
        yaml.dump(OmegaConf.to_container(cfg, resolve=True), f, default_flow_style=False)
    artifact.add_file(config_file, name="config.yaml")
    artifact.add_file(y_true_credible_file, name="y_true_credible.csv")
    artifact.add_file(y_true_predictive_file, name="y_true_predictive.csv")

    if cfg.experiment_params.get("save_full_chains", False):
        for chain_idx, chain_file in enumerate(credible_chain_files):
            artifact.add_file(chain_file, name=f"predictions_credible_chain_{chain_idx}.csv")
        for chain_idx, chain_file in enumerate(predictive_chain_files):
            artifact.add_file(chain_file, name=f"predictions_predictive_chain_{chain_idx}.csv")

    wandb.log_artifact(artifact)
    wandb.finish()
    LOGGER.info(f"Finished logging to W&B for {experiment_name}.")

# --- Plotting ---
def plot_results(all_results, cfg: DictConfig, plot_dir_for_run: str, n_train_samples: int):
    os.makedirs(plot_dir_for_run, exist_ok=True)

    metrics_to_plot = [
        'rmse_credible', 'coverage_credible', 'gr_rmse_credible', 'ess_rmse_credible',
        'rmse_predictive', 'coverage_predictive', 'gr_rmse_predictive', 'ess_rmse_predictive',
        'rmse_pred_vs_true', 'coverage_pred_vs_true', 'gr_rmse_pred_vs_true', 'ess_rmse_pred_vs_true',
        'runtime'
    ]
    labels = list(all_results.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))

    for metric in metrics_to_plot:
        if all(res.get(metric) is None for res in all_results.values()):
            continue
            
        values = [res.get(metric, np.nan) for res in all_results.values()]
        
        plt.figure(figsize=(8, 6))
        plt.bar(labels, values, color=colors)
        plt.title(f'{metric.replace("_", " ").title()} Comparison for DGP: {cfg.dgp} (N_train={n_train_samples})')
        plt.ylabel(metric.replace("_", " ").title())
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir_for_run, f"{cfg.dgp}_ntrain{n_train_samples}_{metric}_comparison.png"))
        plt.close()

# --- Data Handling ---
def load_and_prepare_data(cfg: DictConfig, data_variances):
    exp_params = cfg.experiment_params
    main_seed = exp_params.main_seed
    
    seed_string_test = f"{main_seed}_dgp_test"
    seed_hash_test = hashlib.sha256(seed_string_test.encode('utf-8')).hexdigest()
    test_seed = int(seed_hash_test, 16) % (2 ** 32)

    is_real_dataset = cfg.dgp in ALLOWED_REAL_DATASETS
    X_full_train, y_full_train = None, None
    X_test_run, y_test_run_true = None, None

    if is_real_dataset:
        if cfg.dgp == "california_housing":
            X, y = fetch_california_housing(return_X_y=True)
        else:
            X, y = pmlb.fetch_data(cfg.dgp, return_X_y=True)
        
        X_full_train, X_test_run, y_full_train, y_test_run_true = train_test_split(
            X, y, test_size=exp_params.test_set_fraction, random_state=test_seed
        )
    else:
        if cfg.dgp in pmlb_dataset_names:
            LOGGER.error(f"Dataset '{cfg.dgp}' is a PMLB dataset but is not in the allowed list: {ALLOWED_REAL_DATASETS}")
            raise ValueError(f"Unsupported PMLB dataset: {cfg.dgp}")
            
        dgp_params_test = OmegaConf.to_container(cfg.dgp_params, resolve=True)
        dgp_params_test['n_samples'] = exp_params.n_test_samples
        dgp_params_test['snr'] = np.inf
        dgp_params_test['random_seed'] = test_seed
        generator_test = DataGenerator(**dgp_params_test)
        X_test_run, y_test_run_true = generator_test.generate(scenario=cfg.dgp)

    # Generate a noisy version of the test y for evaluating predictive intervals
    y_test_run_noisy = None
    if not is_real_dataset:
        # Set SNR based on dataset: SNR=1 for piecewise_linear_kunzel and low_lei_candes, SNR=3 for others
        snr_divisor = 1 if cfg.dgp in ['piecewise_linear_kunzel'] else 3
        noise_std_dev = np.sqrt(data_variances[cfg.dgp]['required_noise_for_snr_1'] / snr_divisor)
        rng = np.random.default_rng(test_seed) # Use same seed for consistency
        y_test_run_noisy = y_test_run_true + rng.normal(0, noise_std_dev, size=y_test_run_true.shape)
    else:
        # For real data, the held-out y is the noisy observation
        y_test_run_noisy = y_test_run_true

    return X_full_train, y_full_train, X_test_run, y_test_run_true, y_test_run_noisy


def prepare_train_data(cfg: DictConfig, current_n_train: int, data_variances, X_full_train=None, y_full_train=None):
    main_seed = cfg.experiment_params.main_seed
    is_real_dataset = cfg.dgp in ALLOWED_REAL_DATASETS

    seed_string_train = f"{main_seed}_dgp_train_n{current_n_train}"
    seed_hash_train = hashlib.sha256(seed_string_train.encode('utf-8')).hexdigest()
    dgp_seed_train = int(seed_hash_train, 16) % (2 ** 32)

    if is_real_dataset:
        if X_full_train is None or y_full_train is None:
             raise ValueError("Full training data must be provided for real datasets.")
        if current_n_train > len(X_full_train):
            LOGGER.warning(f"Requested n_train={current_n_train} > available {len(X_full_train)}. Using all data.")
            X_train_run, y_train_run = X_full_train, y_full_train
        else:
            X_train_run, y_train_run = resample(
                X_full_train, y_full_train, n_samples=current_n_train, random_state=dgp_seed_train, replace=False
            )
    else:
        dgp_params_train_dict = OmegaConf.to_container(cfg.dgp_params, resolve=True)
        dgp_params_train_dict['n_samples'] = current_n_train
        # Set SNR based on dataset: SNR=1 for piecewise_linear_kunzel and low_lei_candes, SNR=3 for others
        snr_divisor = 1 if cfg.dgp in ['piecewise_linear_kunzel'] else 3
        dgp_params_train_dict['noise'] = data_variances[cfg.dgp]['required_noise_for_snr_1'] / snr_divisor
        dgp_params_train_dict['random_seed'] = dgp_seed_train
        generator_train = DataGenerator(**dgp_params_train_dict)
        X_train_run, y_train_run = generator_train.generate(scenario=cfg.dgp)
    
    return X_train_run, y_train_run


def create_schedule(schedule_cfg, total_iters):
    schedule_type = schedule_cfg.type
    params = OmegaConf.to_container(schedule_cfg.params, resolve=True)
    if schedule_type == "constant":
        return TemperatureSchedule(lambda t: params['temp'])
    elif schedule_type == "cosine":
        t_max, t_min = params['t_max'], params['t_min']
        return TemperatureSchedule(lambda t: t_min + 0.5 * (t_max - t_min) * (1 + np.cos(np.pi * t / total_iters)))
    elif schedule_type == "linear":
        t_max, t_min = params['t_max'], params['t_min']
        return TemperatureSchedule(lambda t: t_max - (t_max - t_min) * (t / total_iters))
    elif schedule_type == "exponential":
        t_max, gamma = params['t_max'], params['gamma']
        return TemperatureSchedule(lambda t: t_max * (gamma ** t))
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}") 
