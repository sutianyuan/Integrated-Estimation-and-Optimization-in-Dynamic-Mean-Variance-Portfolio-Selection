# exp3_time_varying_volatility

## Key DGP/experiment parameters

T=50; n=10; x0=1; Ntr=60; Nev=1000; S=2000; lambda=5e-4; fixed C0; omega=1e-6; alpha=0.05; beta=0.90; sigma0^2=2e-5; later-stage volatility floor=0.005.

## Main outputs

Six current unified main panels, population-regret/wealth/coefficient summaries, DGP LaTeX, and promotion/run audits.

Main-experiment figures are paper main-panel materials. Sensitivity figures and tables are online-appendix materials. Validation and audit tables are included for traceability.

## Code files (5)

- `code/generate_data.m`
- `code/run_eto.m`
- `code/run_exp3_fixedC_longruninit_minfix.m`
- `code/run_ieo.m`
- `code/run_nonstationary_simulation.m`

##  Figure files (12)

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

## Table files (5)

- `tables/exp3_fixedC_longruninit_minfix_bootstrap_summary.csv`
- `tables/exp3_fixedC_longruninit_minfix_coefficient_diagnostics.csv`
- `tables/exp3_fixedC_longruninit_minfix_extreme_points.csv`
- `tables/exp3_fixedC_longruninit_minfix_stagewise_population_regret.csv`
- `tables/exp3_fixedC_longruninit_minfix_wealth.csv`
