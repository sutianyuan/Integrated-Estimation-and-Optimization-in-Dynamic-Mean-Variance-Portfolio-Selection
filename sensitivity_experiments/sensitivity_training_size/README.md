# Training-sample-size sensitivity experiment

## Appendix experiment parameters

- Environments: `Regular stationary Gaussian` and `Severely ill-conditioned stationary Gaussian`.
- Training-size grid: `Ntr = {40,50,60,80,100,120,160,200,240,320,480}`.
- Number of stages: `T = 50`.
- Number of assets: `n = 10`.
- Evaluation sample size: `Nev = 1000`.
- Monte Carlo replications: `S = 2000`.
- Initial wealth: `x0 = 1`.
- Regularization parameter: `lambda = 1e-4`.
- Paired-bootstrap resamples: `5000`.
- Base random seed: `42`.
- Replication-seed offset: `720000`.
- Bootstrap seed: `314159 + 1000*panel_index + grid_index`.
- Reported difference: `Delta_R = R0_ETO_pop_mean - R0_IEO_pop_mean`.

## MATLAB entry

Run from this module's `code` directory:

```matlab
run_training_sample_sensitivity('dry')
run_training_sample_sensitivity('full')
run_training_sample_sensitivity('plot_only')
```

## Files

Code:

- `code/run_training_sample_sensitivity.m`
- `code/training_sample_sensitivity/run_training_sample_sensitivity_core.m`
- `code/training_sample_sensitivity/plot_training_sample_full.m`
- `code/dependencies/calculate_oracle.m`
- `code/dependencies/run_eto.m`
- `code/dependencies/run_ieo.m`

Appendix tables:

- `tables/results/training_mainDGP_aligned_full.csv`
- `tables/results/summary.csv`

Appendix figures:

- `figures/training_mainDGP_aligned_deltaR.pdf`
- `figures/training_mainDGP_aligned_deltaR.png`
- `figures/training_mainDGP_aligned_deltaR_regular.pdf`
- `figures/training_mainDGP_aligned_deltaR_regular.png`
- `figures/training_mainDGP_aligned_deltaR_ill_conditioned.pdf`
- `figures/training_mainDGP_aligned_deltaR_ill_conditioned.png`
