# BART Computational Efficiency Experiments 

This directory contains all experiment scripts and configurations needed to reproduce
the results from our paper: https://arxiv.org/abs/2406.19958.

## Prerequisites

Before running these experiments, you must install the `bart_playground` package:

```bash
# Option 1: From PyPI
pip install bart-playground

# Option 2: From source
git clone https://github.com/yanshuotan/bart-playground
cd bart-playground
pip install -e .
```

## Quick Start

### Run a basic experiment
```bash
python -m experiments.runner \
  --config-name temperature \
  dgp=linear_additive \
  experiment_params.n_train_samples_list=[100,500]
```

### Run credible predictions for GR analysis
```bash
python experiments/runner.py \
  --config-name credible_predictions_gr_analysis \
  dgp=piecewise_linear_kunzel
```

### Generate plots
```bash
python experiments/plotting_consolidated.py experiment_name=Temperature
```

## Documentation

## Available Experiments

- **temperature.yaml**: Test different posterior temperatures
- **schedule.yaml**: Test temperature schedules (annealing)
- **trees.yaml**: Test different ensemble sizes
- **burnin.yaml**: Test different burn-in lengths
- **proposal_moves.yaml**: Test different proposal move types
- **thresholds.yaml**: Test split point discretization
- **dirichlet.yaml**: Test prior specifications
- **initialization.yaml**: Test XGBoost vs random initialization
- **marginalized.yaml**: Test sampling strategies
- **credible_predictions_gr_analysis.yaml**: Save chains for convergence analysis

## Datasets

### Synthetic (auto-generated)
- linear_additive
- piecewise_linear_kunzel
- low_lei_candes
- high_lei_candes

### Real-world (auto-downloaded)
- california_housing (sklearn)
- 1199_BNG_echoMonths (PMLB)
- 1201_BNG_breastTumor (PMLB)
- 294_satellite_image (PMLB)

## Citation

