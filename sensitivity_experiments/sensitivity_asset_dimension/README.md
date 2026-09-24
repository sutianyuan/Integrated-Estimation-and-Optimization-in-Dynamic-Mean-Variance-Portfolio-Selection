# Asset-dimension sensitivity experiment

## Appendix experiment parameters

- Data-generating process: growing-block equicorrelated Gaussian.
- Asset dimensions: `n = {5,10,15,20,25,30,35,40,45,50}`.
- Number of stages: `T = 50`.
- Training sample size: `Ntr = 300` at every stage.
- Evaluation sample size: `Nev = 1000`.
- Monte Carlo replications: `S = 2000`.
- Initial wealth: `x0 = 1`.
- Regularization parameter: `lambda = 1e-4`.
- Training paths per stage: `J = 300`.
- Factor count: `K(n) = n/5`.
- Within-block population condition number: `20`.
- Stage-0 population-oracle L1 norm: `1.15`.
- Paired-bootstrap resamples: `5000`.
- Base random seed: `42`.
- Full-run replication-seed offset: `500000`.
- Bootstrap seed at grid position `i`: `271828 + 11000 + i`.
- Reported difference: `Delta_R = R0_ETO_star - R0_IEO_star`.

## MATLAB entry

Run from this module's `code` directory:

```matlab
run_asset_dimension_sensitivity('dry')
run_asset_dimension_sensitivity('full')
```

## Files

Code:

- `code/run_asset_dimension_sensitivity.m`
- `code/run_block_gaussian_j300_variable_k_mc2000_wealth.m`
- `code/build_block_gaussian_population.m`
- `code/build_policy_space_bridge.m`
- `code/evaluate_recursive_wealth.m`
- `code/run_structured_ieo_bridge.m`
- `code/dependencies/calculate_oracle.m`
- `code/dependencies/run_eto_factor_cov.m`

Appendix tables:

- `tables/asset_dimension_growing_block_table.csv`
- `tables/paired_bootstrap_ci_by_n.csv`
- `tables/stage0_regret_summary.csv`
