# Experiment 2: Ill-conditioned stationary Gaussian environment

## Supplement-disclosed parameters

The supplement refers the common Monte Carlo design to the main text and
directly discloses the experiment-specific population parameters below.

- Return model: `R_t ~ N_10(mu, Sigma)`, stationary for `t = 0,...,T-1`.
- Common experiment settings: `T=50`, `n=10`, `x0=1`, `Ntr=60`,
  `Nev=1000`, `S=2000`, and `lambda=1e-4`.
- The appendix reports the following population values to six decimal places:

```text
mu = [1.000222, 1.000301, 1.000260, 1.000329, 1.000566,
      1.000462, 1.000511, 1.000276, 1.000171, 1.000421]'

Sigma = 1e-4 * [
  1.444892, 0.608758, 0.223521, 0.669920, 1.045445, 0.629189, 0.759646, 0.492059, 0.517480, 0.640077;
  0.608758, 1.182215, 0.234938, 0.474066, 0.305164, 0.273266, 0.979190, 0.316150, 0.475427, 0.618758;
  0.223521, 0.234938, 0.443832, 0.432770, 0.455728, 0.275172, 0.442594, 0.323052, 0.189194, 0.267404;
  0.669920, 0.474066, 0.432770, 0.873362, 0.564917, 0.286847, 0.534643, 0.569970, 0.242264, 0.712152;
  1.045445, 0.305164, 0.455728, 0.564917, 1.333514, 0.639652, 0.987443, 0.692849, 0.425877, 0.534645;
  0.629189, 0.273266, 0.275172, 0.286847, 0.639652, 0.705442, 0.471510, 0.236669, 0.394900, 0.495580;
  0.759646, 0.979190, 0.442594, 0.534643, 0.987443, 0.471510, 1.487281, 0.541378, 0.346224, 0.692036;
  0.492059, 0.316150, 0.323052, 0.569970, 0.692849, 0.236669, 0.541378, 0.910612, 0.543481, 0.626032;
  0.517480, 0.475427, 0.189194, 0.242264, 0.425877, 0.394900, 0.346224, 0.543481, 0.765938, 0.338907;
  0.640077, 0.618758, 0.267404, 0.712152, 0.534645, 0.495580, 0.692036, 0.626032, 0.338907, 1.244230]
```

## Reproduction entry

Run from this experiment's `code/` directory:

```matlab
run_exp2_main_experiment('full')
```

### Root file (1)

- `README.md`

### Code files (5)

- `code/generate_data.m`
- `code/numpy_pcg64_bootstrap_mean_ci.m`
- `code/run_eto.m`
- `code/run_exp2_main_experiment.m`
- `code/run_ieo.m`

### Figure files (12)

- `figures/exp2_intercept_angle.pdf`
- `figures/exp2_intercept_angle.png`
- `figures/exp2_intercept_gap.pdf`
- `figures/exp2_intercept_gap.png`
- `figures/exp2_population_regret.pdf`
- `figures/exp2_population_regret.png`
- `figures/exp2_slope_angle.pdf`
- `figures/exp2_slope_angle.png`
- `figures/exp2_slope_gap.pdf`
- `figures/exp2_slope_gap.png`
- `figures/exp2_wealth.pdf`
- `figures/exp2_wealth.png`

### Table files (5)

- `tables/exp2_scale0p60_ridge3em09_confirmation_Sigma.csv`
- `tables/exp2_scale0p60_ridge3em09_confirmation_coefficient_diagnostics.csv`
- `tables/exp2_scale0p60_ridge3em09_confirmation_mu.csv`
- `tables/exp2_scale0p60_ridge3em09_confirmation_stagewise_population_regret.csv`
- `tables/exp2_scale0p60_ridge3em09_confirmation_wealth.csv`
