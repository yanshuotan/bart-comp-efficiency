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
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error

from bart_playground import DefaultBART, DefaultPreprocessor, DataGenerator
from bart_playground.samplers import default_proposal_probs

from .utils import (
    LOGGER,
    gelman_rubin, 
    effective_sample_size, 
    log_wandb_artifacts, 
    plot_results,
    load_and_prepare_data,
    prepare_train_data,
    create_schedule
)

# --- Chain Execution ---
def run_chain_bart(X_train_raw, X_test_raw, y_train_raw, y_test_true, y_test_noisy, cfg: DictConfig, seed: int):
    bart_cfg = cfg.bart_params
    
    # Determine max_bins: use from config, or default to number of training samples
    max_bins_to_use = bart_cfg.get('max_bins', X_train_raw.shape[0])
    preprocessor = DefaultPreprocessor(max_bins=max_bins_to_use)
    train_data = preprocessor.fit_transform(X_train_raw, y_train_raw)

    total_mcmc_iters = bart_cfg.ndpost + bart_cfg.nskip
    temp_schedule = None
    temperature_val = bart_cfg.get('temperature', 1.0)
    
    if 'schedule' in cfg and cfg.schedule is not None:
        temp_schedule = create_schedule(cfg.schedule, total_mcmc_iters)

    model = DefaultBART(
        ndpost=bart_cfg.ndpost,
        nskip=bart_cfg.nskip,
        n_trees=bart_cfg.n_trees,
        random_state=seed,
        proposal_probs=bart_cfg.get('proposal_probs', default_proposal_probs),
        temperature=temp_schedule if temp_schedule is not None else temperature_val,
        dirichlet_prior=bart_cfg.get('dirichlet_prior', False)
    )
    
    if bart_cfg.get('marginalize', False):
        model.sampler.marginalize = True

    model.preprocessor = preprocessor
    model.data = train_data
    model.sampler.add_data(train_data)
    model.sampler.add_thresholds(preprocessor.thresholds)
    model.is_fitted = True

    if bart_cfg.get('init_from_xgb', False):
        xgb_tuning_cfg = cfg.get('xgb_tuning_params', {'enabled': False})
        if xgb_tuning_cfg.get('enabled', False):
            LOGGER.info("--- Running RandomizedSearchCV for XGBoost hyperparameters ---")
            param_distributions = {
                'n_estimators': [50, 100, 150, cfg.bart_params.n_trees],
                'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                'max_depth': [3, 4, 5, 6, 7, 8],
                'min_child_weight': [1, 3, 5, 7],
                'gamma': [0.0, 0.1, 0.2, 0.3, 0.4],
                'reg_alpha': [0, 0.005, 0.01, 0.05, 0.1],
                'reg_lambda': [0.5, 1.0, 1.5, 2.0, 5.0],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            }
            xgb_reg = xgb.XGBRegressor(random_state=seed, tree_method="exact", grow_policy="depthwise", base_score=0.0)
            random_search = RandomizedSearchCV(
                estimator=xgb_reg, param_distributions=param_distributions,
                n_iter=xgb_tuning_cfg.get('n_iter', 10), scoring='neg_mean_squared_error',
                n_jobs=-1, cv=3, verbose=1, random_state=seed
            )
            random_search.fit(train_data.X, train_data.y)
            LOGGER.info(f"Best XGBoost parameters found: {random_search.best_params_}")
            xgb_model_instance = random_search.best_estimator_
        else:
            xgb_cfg = cfg.get('xgb_params', OmegaConf.create({'max_depth': 4, 'learning_rate': 0.2}))
            xgb_model_instance = xgb.XGBRegressor(
                n_estimators=bart_cfg.n_trees, max_depth=xgb_cfg.max_depth,
                learning_rate=xgb_cfg.learning_rate, random_state=seed, tree_method="exact",
                grow_policy="depthwise", base_score=0.0
            )
            xgb_model_instance.fit(train_data.X, train_data.y)
        model.init_from_xgboost(xgb_model_instance, train_data.X, train_data.y, debug=cfg.get('debug_xgb_init', False))


    start_time = time.perf_counter()
    trace = model.sampler.run(
        n_iter=bart_cfg.ndpost + bart_cfg.nskip,
        quietly=True,
        n_skip=bart_cfg.nskip
    )
    runtime = time.perf_counter() - start_time
    model.trace = trace

    # --- Credible Interval Calculations (for f(x)) ---
    credible_samples = model.posterior_f(X_test_raw)
    if credible_samples.shape[0] != X_test_raw.shape[0] and credible_samples.shape[1] == X_test_raw.shape[0]:
        credible_samples = credible_samples.T

    preds_mean_credible = np.mean(credible_samples, axis=1)
    rmse_credible = np.sqrt(mean_squared_error(y_test_true, preds_mean_credible))

    lower_credible = np.percentile(credible_samples, 2.5, axis=1)
    upper_credible = np.percentile(credible_samples, 97.5, axis=1)
    coverage_credible = np.mean((y_test_true >= lower_credible) & (y_test_true <= upper_credible))

    # --- Posterior Predictive Interval Calculations (for y_new) ---
    predictive_samples = model.posterior_predict(X_test_raw)
    if predictive_samples.shape[0] != X_test_raw.shape[0] and predictive_samples.shape[1] == X_test_raw.shape[0]:
        predictive_samples = predictive_samples.T

    preds_mean_predictive = np.mean(predictive_samples, axis=1)
    rmse_predictive = np.sqrt(mean_squared_error(y_test_noisy, preds_mean_predictive))

    lower_predictive = np.percentile(predictive_samples, 2.5, axis=1)
    upper_predictive = np.percentile(predictive_samples, 97.5, axis=1)
    coverage_predictive = np.mean((y_test_noisy >= lower_predictive) & (y_test_noisy <= upper_predictive))

    # --- Predictive Samples vs. True Mean (f(x)) Calculations ---
    rmse_pred_vs_true = np.sqrt(mean_squared_error(y_test_true, preds_mean_predictive))
    coverage_pred_vs_true = np.mean((y_test_true >= lower_predictive) & (y_test_true <= upper_predictive))

    return {
        'rmse_credible': rmse_credible,
        'coverage_credible': coverage_credible,
        'credible_samples': credible_samples,
        'rmse_predictive': rmse_predictive,
        'coverage_predictive': coverage_predictive,
        'predictive_samples': predictive_samples,
        'rmse_pred_vs_true': rmse_pred_vs_true,
        'coverage_pred_vs_true': coverage_pred_vs_true,
        'runtime': runtime,
        'sigma2_samples': np.array([p.global_params['eps_sigma2'] for p in trace]),
    }


def run_full_experiment(X_train_specific, y_train_specific, X_test_specific, y_test_true, y_test_noisy, cfg: DictConfig, current_n_train: int):
    exp_cfg = cfg.experiment_params
    all_results = []

    for i in range(exp_cfg.n_chains):
        seed_string = f"{exp_cfg.main_seed}_chain_{i}_ntrain_{current_n_train}"
        seed_hash = hashlib.sha256(seed_string.encode('utf-8')).hexdigest()
        chain_specific_seed = int(seed_hash, 16) % (2 ** 32)
        LOGGER.info(f"Running chain {i + 1}/{exp_cfg.n_chains} with chain_specific_seed {chain_specific_seed}")
        chain_res = run_chain_bart(X_train_specific, X_test_specific, y_train_specific, y_test_true, y_test_noisy,
                                   cfg, chain_specific_seed)
        all_results.append(chain_res)

    # Aggregate credible results
    credible_preds_chains = np.stack([r['credible_samples'] for r in all_results], axis=1)
    gr_rmse_credible, ess_rmse_credible = np.nan, np.nan
    if credible_preds_chains.ndim == 3 and credible_preds_chains.shape[0] > 0:
        squared_errors_credible = (credible_preds_chains - y_test_true[:, None, None]) ** 2
        mse_chains_credible = np.mean(squared_errors_credible, axis=0)
        rmse_chains_credible = np.sqrt(mse_chains_credible)
        gr_rmse_credible = gelman_rubin(rmse_chains_credible)
        ess_rmse_credible = effective_sample_size(rmse_chains_credible)

    # Aggregate predictive results
    predictive_preds_chains = np.stack([r['predictive_samples'] for r in all_results], axis=1)
    gr_rmse_predictive, ess_rmse_predictive = np.nan, np.nan
    if predictive_preds_chains.ndim == 3 and predictive_preds_chains.shape[0] > 0:
        squared_errors_predictive = (predictive_preds_chains - y_test_noisy[:, None, None]) ** 2
        mse_chains_predictive = np.mean(squared_errors_predictive, axis=0)
        rmse_chains_predictive = np.sqrt(mse_chains_predictive)
        gr_rmse_predictive = gelman_rubin(rmse_chains_predictive)
        ess_rmse_predictive = effective_sample_size(rmse_chains_predictive)

    # Aggregate predictive vs true results
    gr_rmse_pred_vs_true, ess_rmse_pred_vs_true = np.nan, np.nan
    if predictive_preds_chains.ndim == 3 and predictive_preds_chains.shape[0] > 0:
        squared_errors_pred_vs_true = (predictive_preds_chains - y_test_true[:, None, None]) ** 2
        mse_chains_pred_vs_true = np.mean(squared_errors_pred_vs_true, axis=0)
        rmse_chains_pred_vs_true = np.sqrt(mse_chains_pred_vs_true)
        gr_rmse_pred_vs_true = gelman_rubin(rmse_chains_pred_vs_true)
        ess_rmse_pred_vs_true = effective_sample_size(rmse_chains_pred_vs_true)


    return {
        'rmse_credible': np.mean([r['rmse_credible'] for r in all_results]),
        'coverage_credible': np.mean([r['coverage_credible'] for r in all_results]),
        'gr_rmse_credible': gr_rmse_credible,
        'ess_rmse_credible': ess_rmse_credible,
        
        'rmse_predictive': np.mean([r['rmse_predictive'] for r in all_results]),
        'coverage_predictive': np.mean([r['coverage_predictive'] for r in all_results]),
        'gr_rmse_predictive': gr_rmse_predictive,
        'ess_rmse_predictive': ess_rmse_predictive,

        'rmse_pred_vs_true': np.mean([r['rmse_pred_vs_true'] for r in all_results]),
        'coverage_pred_vs_true': np.mean([r['coverage_pred_vs_true'] for r in all_results]),
        'gr_rmse_pred_vs_true': gr_rmse_pred_vs_true,
        'ess_rmse_pred_vs_true': ess_rmse_pred_vs_true,

        'runtime': np.mean([r['runtime'] for r in all_results]),
        'sigma2_samples': np.array([r['sigma2_samples'] for r in all_results if r['sigma2_samples'].size > 0]),
    }, credible_preds_chains, predictive_preds_chains


# --- Main Analysis Function ---
def run_and_analyze(cfg: DictConfig):
    exp_params = cfg.experiment_params
    main_seed = exp_params.main_seed
    exp_type = exp_params.get("type", "grid") # grid or pairs

    with open("experiments/data_variances.yaml", 'r') as f:
        data_variances = yaml.safe_load(f)

    dgp_name_safe = "".join(c if c.isalnum() else "_" for c in cfg.dgp)
    dgp_and_seed_base_artifact_dir = os.path.join(cfg.artifacts_dir, dgp_name_safe, f"seed_{main_seed}")
    os.makedirs(dgp_and_seed_base_artifact_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(dgp_and_seed_base_artifact_dir, "config_main_seed.yaml"))

    X_full_train, y_full_train, X_test_run, y_test_run_true, y_test_run_noisy = load_and_prepare_data(cfg, data_variances)

    for idx, current_n_train in enumerate(exp_params.n_train_samples_list):
        if cfg.dgp == "294_satellite_image" and current_n_train > 5000:
            LOGGER.info(f"Skipping N_train={current_n_train} for '294_satellite_image' as it exceeds 5k limit.")
            continue
            
        LOGGER.info(f"--- Starting run for DGP: {cfg.dgp} with N_train = {current_n_train} ---")
        
        X_train_run, y_train_run = prepare_train_data(cfg, current_n_train, data_variances, X_full_train, y_full_train)
            
        run_specific_artifact_dir = os.path.join(dgp_and_seed_base_artifact_dir, f"ntrain_{current_n_train}")
        os.makedirs(run_specific_artifact_dir, exist_ok=True)

        all_results = {}
        
        variations = []
        if exp_type == "grid":
            param_to_vary = exp_params.param_to_vary
            param_values = exp_params.param_values
            for value in param_values:
                variations.append({
                    "name": f"{param_to_vary.split('.')[-1]}_{value}",
                    "params": {param_to_vary: value}
                })
        elif exp_type == "pairs":
            variations = exp_params.pairs
        elif exp_type == "schedule":
            variations = [{"name": s.name, "params": {}, "schedule_cfg": s} for s in exp_params.schedules]

        for variation in variations:
            result_key = variation["name"]
            results_path = os.path.join(run_specific_artifact_dir, f"results_{result_key}.pkl")
            if os.path.exists(results_path):
                LOGGER.info(f"Results file already exists, skipping: {results_path}")
                with open(results_path, "rb") as f:
                    all_results[result_key] = pickle.load(f)
                continue

            LOGGER.info(f"--- Running Experiment: {result_key} ---")
            
            run_cfg = cfg.copy()
            OmegaConf.set_struct(run_cfg, False)
            for param, value in variation.get("params", {}).items():
                 OmegaConf.update(run_cfg, f"{param}", value)

            if exp_type == "schedule":
                 run_cfg.schedule = variation["schedule_cfg"]

            OmegaConf.set_struct(run_cfg, True)

            results, credible_chains, predictive_chains = run_full_experiment(
                X_train_run, y_train_run, X_test_run, y_test_run_true, y_test_run_noisy, 
                run_cfg, current_n_train=current_n_train
            )
            
            if not exp_params.get("save_full_chains", False):
                if 'sigma2_samples' in results:
                    del results['sigma2_samples']
            
            all_results[result_key] = results

            LOGGER.info(f"Results ({result_key}) for DGP {cfg.dgp}, N_train={current_n_train}:")
            for k, v in results.items():
                if k not in ['sigma2_samples']:
                    LOGGER.info(f"  {k}: {v}")

            with open(results_path, "wb") as f:
                pickle.dump(results, f)
            
            # Save full chains if requested (compressed format with float32 to save space)
            if exp_params.get("save_full_chains", False):
                credible_chains_path = os.path.join(run_specific_artifact_dir, f"credible_chains_{result_key}.npz")
                np.savez_compressed(credible_chains_path, credible_chains=credible_chains.astype(np.float32))
                LOGGER.info(f"Saved compressed credible chains to {credible_chains_path}")

            if cfg.experiment_params.get("log_to_wandb", False):
                log_wandb_artifacts(
                    cfg, exp_params.name, result_key, results, 
                    credible_chains, predictive_chains, 
                    y_test_run_true, y_test_run_noisy, 
                    run_specific_artifact_dir, current_n_train
                )

        if exp_params.get("plot_results", True) and all_results:
            plot_results(all_results, cfg, run_specific_artifact_dir, current_n_train)

        LOGGER.info(f"Experiment for N_train={current_n_train} complete.")

@hydra.main(config_path="consolidated_configs", config_name="temperature", version_base=None)
def main(cfg: DictConfig):
    run_and_analyze(cfg)

if __name__ == "__main__":
    main() 