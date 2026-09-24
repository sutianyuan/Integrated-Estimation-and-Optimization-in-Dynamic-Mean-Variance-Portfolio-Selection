# Horizon sensitivity experiment

## Appendix experiment parameters

- Environments: `Regular stationary Gaussian` and `Severely ill-conditioned stationary Gaussian`.
- Horizon grid: `T = 10:5:200`.
- Number of assets: `n = 10`.
- Training sample size: `Ntr = 60` at every stage.
- Evaluation sample size: `Nev = 1000`.
- Monte Carlo replications: `S = 2000`.
- Initial wealth: `x0 = 1`.
- Regularization parameter: `lambda = 1e-4`.
- Paired-bootstrap resamples: `5000`.
- Base random seed: `42`.
- Replication-seed offset: `620000`.
- Bootstrap seed: `271828 + 100*panel_index + horizon_index`.
- Reported difference: `Delta_R = R0_ETO_pop_mean - R0_IEO_pop_mean`.

## MATLAB entry

Run from this module's `code` directory:

```matlab
run_horizon_environment_sensitivity('dry')
run_horizon_environment_sensitivity('full')
run_horizon_environment_sensitivity('plot_only')
```

## Files

Code:

- `code/run_horizon_environment_sensitivity.m`
- `code/horizon_environment_sensitivity/run_horizon_environment_sensitivity_core.m`
- `code/horizon_environment_sensitivity/plot_horizon_regret_full.m`
- `code/dependencies/calculate_oracle.m`
- `code/dependencies/run_eto.m`
- `code/dependencies/run_ieo.m`

Appendix tables:

- `tables/results/horizon_mainDGP_aligned_full.csv`
- `tables/results/summary.csv`

Appendix figures:

- `figures/horizon_mainDGP_aligned_deltaR.pdf`
- `figures/horizon_mainDGP_aligned_deltaR.png`
- `figures/horizon_mainDGP_aligned_deltaR_regular.pdf`
- `figures/horizon_mainDGP_aligned_deltaR_regular.png`
- `figures/horizon_mainDGP_aligned_deltaR_ill_conditioned.pdf`
- `figures/horizon_mainDGP_aligned_deltaR_ill_conditioned.png`
