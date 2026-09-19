# Current numerical-experiment release

This folder contains the Matlab code, final figures, and table data for the numerical experiments.

## Main experiments

1. `main_experiments/exp1_regular_stationary_gaussian/` — regular stationary Gaussian.
2. `main_experiments/exp2_ill_conditioned_stationary_gaussian/` — ill-conditioned stationary Gaussian.
3. `main_experiments/exp3_time_varying_volatility/` — fixed-C time-varying volatility with long-run initial marginal variance.

## Sensitivity experiments

1. `sensitivity_experiments/sensitivity_horizon/` — horizon sensitivity.
2. `sensitivity_experiments/sensitivity_training_size/` — training-sample-size sensitivity.
3. `sensitivity_experiments/sensitivity_asset_dimension/` — asset-dimension sensitivity.

Each experiment folder contains `code/`, `figures/`, `tables/`, and `docs/`. Its README records the  key parameters, main outputs.

The six unified main-experiment panels in each main experiment's `figures/` directory are the current main-paper figure materials. Sensitivity figures and paper-facing tables are online-appendix materials.
