# Empirical experiment replication package

This folder contains only the finalized main-text empirical illustration. It
corresponds to Section 5 and Figure A.4 of the paper.

No experiment was rerun, no figure was redrawn, and no original source file
was modified during packaging.

## Final empirical design

- Data: daily value-weighted Fama--French industry portfolio returns,
  2017-01-03 through 2025-11-28.
- Assets: Food Products, Healthcare, Construction, Automobiles, Utilities,
  Services, Paper, Wholesale, Finance, and Electrical Equipment.
- Horizon: `T = 30` trading days.
- Lookback: 500 days, with 501 observations in each inclusive history.
- Rolling step: 10 trading days.
- Rolling origins: `K = 171`.
- Training paths: `N_A = 100` 30-day blocks sampled with replacement.
- Sampling pool: 472 admissible starting positions in each rolling history.
- Initial wealth: `x0 = 1`.
- Risk-aversion parameter in the available empirical code: `lambda = 0.001`.
- Value-function evaluation: 201-point, stage-specific empirical wealth grid.
- Methods: ETO and IEO.

The number 472 is the size of the admissible block-start pool. It is not the
number of bootstrap training paths in the finalized design.

## Contents

- `code/`: current data-preparation and ETO/IEO supporting code;
- `data/`: the processed 20-industry input and finalized ten-industry mapping;
- `figures/`: the two PNG components directly referenced as Figure A.4;
- `tables/`: status of the unavailable exact final numeric arrays;
- `docs/`: source map, design, data provenance, manuscript excerpt, and
  reproducibility note.

## Reproduction status

No exact final runner was located. The packaged figures are the approved files
referenced by the latest manuscript, but the available runner/result pairs do
not implement all final requirements simultaneously. The exact gap is
documented in `docs/REPRODUCIBILITY_NOTE.md` and `PACKAGING_AUDIT.md`.

The supporting code can be inspected from `code/`, and the expected input is
`data/Fama-French_20行业指数_日度_2017-2025.csv`. Exact regeneration of Figure
A.4 additionally requires a runner that combines the ten-industry selection,
`N_A = 100` block resampling from the 472-start pool, and the 201-point
wealth-grid discrepancy.

## Exclusions

Monte Carlo main experiments, sensitivity analyses, the old 20-asset
empirical version, the 472-full-overlap empirical version, PAA extensions, and
other exploratory or obsolete outputs are excluded.
