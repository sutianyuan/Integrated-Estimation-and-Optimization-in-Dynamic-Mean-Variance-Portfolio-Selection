# Experiment 3: Time-varying volatility environment

## Supplement-disclosed parameters

The supplement refers the common Monte Carlo design to the main text and
directly discloses the experiment-specific volatility, correlation, and mean
process below.

- Common experiment settings: `T=50`, `n=10`, `x0=1`, `Ntr=60`,
  `Nev=1000`, `S=2000`, and `lambda=5e-4`.
- Initial marginal volatility is the GARCH long-run level:
  `sigma_i,0 = sqrt(omega/(1-alpha-beta))`.
- With `omega=1e-6`, `alpha=0.05`, and `beta=0.90`, the initial marginal
  variance is `2e-5` and the initial volatility is approximately
  `0.00447213595`.
- For subsequent stages,
  `sigma_i,t+1 = max(0.005, sqrt(omega + alpha*(sigma_i,t*z_i,t)^2 + beta*sigma_i,t^2))`,
  where `z_i,t ~ N(0,1)`.
- `Sigma_t = diag(sigma_t) * C * diag(sigma_t)`, with fixed correlation
  matrix:

```text
C = [
  1.000000, 0.421820, 0.441280, 0.451158, 0.699482, 0.519918, 0.580214, 0.426110, 0.622667, 0.325587;
  0.421820, 1.000000, 0.541822, 0.478110, 0.458755, 0.600489, 0.455362, 0.377259, 0.668164, 0.536441;
  0.441280, 0.541822, 1.000000, 0.321747, 0.695696, 0.589005, 0.382903, 0.388383, 0.477386, 0.607148;
  0.451158, 0.478110, 0.321747, 1.000000, 0.541395, 0.457211, 0.461625, 0.479509, 0.696436, 0.456498;
  0.699482, 0.458755, 0.695696, 0.541395, 1.000000, 0.657055, 0.535094, 0.396409, 0.366890, 0.373125;
  0.519918, 0.600489, 0.589005, 0.457211, 0.657055, 1.000000, 0.682907, 0.423716, 0.376124, 0.409612;
  0.580214, 0.455362, 0.382903, 0.461625, 0.535094, 0.682907, 1.000000, 0.435203, 0.329620, 0.604281;
  0.426110, 0.377259, 0.388383, 0.479509, 0.396409, 0.423716, 0.435203, 1.000000, 0.619966, 0.381383;
  0.622667, 0.668164, 0.477386, 0.696436, 0.366890, 0.376124, 0.329620, 0.619966, 1.000000, 0.609985;
  0.325587, 0.536441, 0.607148, 0.456498, 0.373125, 0.409612, 0.604281, 0.381383, 0.609985, 1.000000]
```

## Reproduction entry

Run from this experiment's `code/` directory:

```matlab
run_exp3_fixedC_longruninit_minfix('Mode','full')
```

The current paper-facing figures in this folder use the concise `exp3_*`
filenames listed below. The runner's native full diagnostic export uses
additional `exp3_fixedC_longruninit_minfix_*` filenames; those additional
files are not currently packaged in this folder. The saved result CSV files do
use the runner's full `exp3_fixedC_longruninit_minfix_*` prefix.

### Root file (1)

- `README.md`

### Code files (4)

- `code/generate_data.m`
- `code/run_eto.m`
- `code/run_exp3_fixedC_longruninit_minfix.m`
- `code/run_ieo.m`

### Figure files (12)

- `figures/exp3_intercept_angle.pdf`
- `figures/exp3_intercept_angle.png`
- `figures/exp3_intercept_gap.pdf`
- `figures/exp3_intercept_gap.png`
- `figures/exp3_population_regret.pdf`
- `figures/exp3_population_regret.png`
- `figures/exp3_slope_angle.pdf`
- `figures/exp3_slope_angle.png`
- `figures/exp3_slope_gap.pdf`
- `figures/exp3_slope_gap.png`
- `figures/exp3_wealth.pdf`
- `figures/exp3_wealth.png`

### Result files (5)

- `results/exp3_fixedC_longruninit_minfix_bootstrap_summary.csv`
- `results/exp3_fixedC_longruninit_minfix_coefficient_diagnostics.csv`
- `results/exp3_fixedC_longruninit_minfix_extreme_points.csv`
- `results/exp3_fixedC_longruninit_minfix_stagewise_population_regret.csv`
- `results/exp3_fixedC_longruninit_minfix_wealth.csv`

