# Experiment 1: Regular stationary Gaussian environment

## Supplement-disclosed parameters

The supplement refers the common Monte Carlo design to the main text and
directly discloses the experiment-specific population parameters below.

- Return model: `R_t ~ N_10(mu, Sigma)`, stationary for `t = 0,...,T-1`.
- Common experiment settings: `T=50`, `n=10`, `x0=1`, `Ntr=60`,
  `Nev=1000`, `S=2000`, and `lambda=1e-4`.
- The appendix reports the following population values to six decimal places:

```text
mu = [1.000404, 1.000185, 1.000180, 0.999810, 1.000117,
      1.000177, 1.000116, 1.000366, 1.000401, 1.000614]'

Sigma = 1e-4 * [
  0.524751,  0.025934, -0.360601, -0.034440,  0.355683, -0.316986, -0.135901, -0.084878, -0.046953, -0.168199;
  0.025934,  0.785543,  0.044136,  0.082410, -0.102082,  0.171996,  0.113952, -0.451855, -0.040720,  0.057669;
 -0.360601,  0.044136,  0.731875, -0.113506, -0.283706, -0.018387, -0.143125,  0.033220,  0.030842,  0.182392;
 -0.034440,  0.082410, -0.113506,  0.537838, -0.183596,  0.151577,  0.458496,  0.085172, -0.072742,  0.092638;
  0.355683, -0.102082, -0.283706, -0.183596,  0.847116, -0.405821, -0.282447, -0.364300, -0.277688, -0.202246;
 -0.316986,  0.171996, -0.018387,  0.151577, -0.405821,  1.343005,  0.577217, -0.251881,  0.237220,  0.085883;
 -0.135901,  0.113952, -0.143125,  0.458496, -0.282447,  0.577217,  1.030729, -0.015730,  0.079969,  0.177818;
 -0.084878, -0.451855,  0.033220,  0.085172, -0.364300, -0.251881, -0.015730,  1.218117,  0.049380, -0.081225;
 -0.046953, -0.040720,  0.030842, -0.072742, -0.277688,  0.237220,  0.079969,  0.049380,  0.578830, -0.054054;
 -0.168199,  0.057669,  0.182392,  0.092638, -0.202246,  0.085883,  0.177818, -0.081225, -0.054054,  0.505532]
```

## Reproduction entry

Run from this experiment's `code/` directory:

```matlab
run_regular_gaussian_main_experiment('full')
```

### Root file (1)

- `README.md`

### Code files (6)

- `code/calculate_oracle.m`
- `code/generate_data.m`
- `code/numpy_pcg64_bootstrap_mean_ci.m`
- `code/run_eto.m`
- `code/run_ieo.m`
- `code/run_regular_gaussian_main_experiment.m`

### Figure files (12)

- `figures/exp1_intercept_angle.pdf`
- `figures/exp1_intercept_angle.png`
- `figures/exp1_intercept_gap.pdf`
- `figures/exp1_intercept_gap.png`
- `figures/exp1_population_regret.pdf`
- `figures/exp1_population_regret.png`
- `figures/exp1_slope_angle.pdf`
- `figures/exp1_slope_angle.png`
- `figures/exp1_slope_gap.pdf`
- `figures/exp1_slope_gap.png`
- `figures/exp1_wealth.pdf`
- `figures/exp1_wealth.png`

### Table files (3)

- `tables/population_regret_stagewise.csv`
- `tables/stagewise_summary.csv`
- `tables/wealth_summary.csv`
