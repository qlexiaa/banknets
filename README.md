# Bank Networks and Credit Growth Spillovers

This project replicates and extends Favara & Imbs (2015) with a focus on whether interstate banking deregulation propagates through shared bank-network exposure across counties. The main dependent variable is county-level credit growth, `Dl_nloans_b`.

The core comparison is between:

- `W_geo`: queen contiguity between counties.
- `W_bank`: cosine similarity in bank holding-company presence, averaged over 1994-2005.

House-price growth is kept as a companion reduced-form exercise, but the current thesis pipeline is organized around the credit-growth DV.

## Repository Layout

```text
analysis/
  run_all.py                         Current active analysis pipeline
  utils.py                           Shared data and weights helpers
  diagnostics.py                     Baseline Moran's I diagnostics
  bank_moran_diagnostics.py          Moran's I under W_bank variants
  bank_geo_overlap.py                Geographic overlap in bank links
  lm_diagnostics_credit.py           LM error/lag diagnostics for credit growth
  panel_fe_credit.py                 Main Panel_FE_Error SEM: W_geo vs W_bank
  panel_fe_hpi.py                    HPI companion Panel_FE_Error check
  rq1_composite_w_hpi.py             HPI-DV composite W(alpha) sweep
  rq1_four_w_comparison_credit.py    W_geo, W_bank_bin, W_bank_count, W_bank_nonGeo
  bank_nongeo_credit.py              W_bank after removing geographic-neighbor links
  knn_crossover_credit.py            KNN sweep, k=1 to k=20
  rq1_jtest.py                       Credit-DV Davidson-MacKinnon J-tests
  rq1_composite_w.py                 Credit-DV composite W(alpha) sweep
  spatial_multiplier_decomposition.py Spatial multiplier and distance decomposition
  deprecated/                        Older HPI-first and superseded scripts

pipeline/
  01_download_fdic.py
  02_merge_panel_geo.py
  03_build_geo_weights.py
  04_build_bank_weights.py
  05_build_panel.py

data/
  estimation_panel.csv
  W_geo_queen.gal
  W_bank_avg.npz
  W_bank_count_avg.npz
  W_bank_nonGeo.npz
  county_order_Wgeo.csv
  county_order_Wbank.csv

output/
  Current results, figures, and diagnostics
  deprecated/                        Older outputs archived during cleanup
```

`Replication/` contains the original Favara-Imbs replication package and is not modified by the current pipeline.

## Run

Build the data once:

```bash
python pipeline/01_download_fdic.py
python pipeline/02_merge_panel_geo.py
python pipeline/03_build_geo_weights.py
python pipeline/04_build_bank_weights.py
python pipeline/05_build_panel.py
```

Run the current analysis:

```bash
python analysis/run_all.py
```

The active pipeline runs, in order: baseline diagnostics, bank Moran diagnostics, bank-geography overlap, LM diagnostics, credit SEM, HPI companion SEM, HPI composite W sweep, four-W comparison, non-geographic bank links, KNN sweep, J-tests, credit composite W sweep, and spatial multiplier decomposition.

## Main Credit Results

Panel fixed-effects SEM estimates show much stronger spatial error dependence under the bank network than under geographic contiguity:

| Sample | W_geo lambda | W_bank lambda | Gap |
|---|---:|---:|---:|
| Full | 0.180 | 0.699 | 0.519 |
| Non-border | 0.170 | 0.671 | 0.500 |

The four-W comparison gives the same pattern. In the full sample, lambda is 0.180 for `W_geo`, 0.686 for binary `W_bank`, 0.708 for count-weighted `W_bank`, and 0.485 for `W_bank_nonGeo`. In the non-border sample, lambda is 0.170 for `W_geo`, 0.652 for binary `W_bank`, 0.646 for count-weighted `W_bank`, and 0.495 for `W_bank_nonGeo`.

The non-geographic bank result matters: after removing all bank links between geographic neighbors, `W_bank_nonGeo` still produces lambda of 0.485 in the full sample and 0.495 in the non-border sample. The bank channel is therefore not just disguised geographic adjacency.

## Robustness And Diagnostics

The KNN sweep now runs from `k=1` through `k=20` and reports density/sparsity for each truncated matrix. The bank-network lambda exceeds the geographic benchmark starting at `k=2`, and rises to 0.477 by `k=20` in the full sample.

LM diagnostics for the credit-growth within residuals give mixed but informative specification guidance:

| Sample | W | Decision |
|---|---|---|
| Full | W_geo | Indeterminate |
| Full | W_bank | SAR |
| Non-border | W_geo | SEM |
| Non-border | W_bank | SAR |

Composite weights improve fit relative to either pure matrix. The optimal credit-growth composites put most weight on the bank network while retaining a meaningful geography component:

| Pair | Sample | alpha* | Log-likelihood gain vs W_geo |
|---|---|---:|---:|
| W_geo + W_bank_bin | Full | 0.20 | 60.47 |
| W_geo + W_bank_bin | Non-border | 0.20 | 40.47 |
| W_geo + W_bank_count | Full | 0.20 | 60.77 |
| W_geo + W_bank_count | Non-border | 0.20 | 41.20 |
| W_geo + W_bank_nonGeo | Full | 0.30 | 41.22 |
| W_geo + W_bank_nonGeo | Non-border | 0.25 | 40.28 |

Here `alpha` is the weight on `W_geo` in `W(alpha) = alpha * W_geo + (1 - alpha) * W_bank_variant`.

## HPI Companion Check

The HPI reduced-form Panel_FE_Error comparison is unweighted because `Panel_FE_Error` does not natively support the inverse-counties-per-state weights used in Favara & Imbs. Under that unweighted specification, lambda is 0.733 under `W_geo` and 0.977 under `W_bank` in the full sample; in the non-border sample it is 0.693 under `W_geo` and 0.972 under `W_bank`.

The HPI composite W result is estimated separately with `Dl_hpi` as the dependent variable and the Favara-Imbs reduced-form regressors `Linter_bra` and `Linter_ela`. This replaces the old deprecated demand-shifter composite result where alpha* was about 0.85.

| Sample | alpha* | lambda at alpha* | Log-likelihood gain vs best pure W |
|---|---:|---:|---:|
| Full | 0.75 | 0.917 | 89.02 |
| Non-border | 0.70 | 0.913 | 73.59 |

Here `alpha` is the weight on `W_geo`, so the HPI composite is geography-heavy but still includes a bank-network component.

## Spatial Multipliers

For credit growth, `W_bank` produces a much larger indirect component and a much longer spatial reach than `W_geo`:

| Outcome | W | Total multiplier | Indirect share | Median reach |
|---|---|---:|---:|---:|
| Credit | W_geo | 1.220 | 17.1% | 40 km |
| Credit | W_bank | 3.323 | 69.6% | 926 km |
| HPI | W_geo | 3.751 | 63.5% | 54 km |
| HPI | W_bank | 43.053 | 97.5% | 1224 km |

## Current Outputs

Key current files in `output/`:

- `panel_fe_credit_results.csv`
- `panel_fe_hpi_results.csv`
- `composite_w_hpi_results.csv`
- `composite_w_hpi_optima.csv`
- `composite_w_hpi_profile.png`
- `four_w_comparison_credit.csv`
- `bank_nongeo_credit_results.csv`
- `bank_nongeo_matrix_stats.csv`
- `knn_sweep_credit_results.csv`
- `lm_diagnostics_credit.csv`
- `jtest_credit_results.csv`
- `composite_w_credit_results.csv`
- `composite_w_credit_optima.csv`
- `composite_w_credit_profiles.png`
- `geo_bank_overlap_stats.csv`
- `geo_bank_overlap_distribution.csv`
- `overlap_histogram.png`
- `spatial_multiplier_decomposition.csv`
- `spatial_multiplier_decay.csv`

Diagnostic outputs are in `output/diagnostics/`:

- `moran_i_results.csv`
- `moran_i_by_year.png`
- `moran_i_wbank_results.csv`
- `moran_i_wbank_summary.csv`
- `moran_i_wbank_by_year.png`
- `moran_i_composite_credit_results.csv`
- `moran_i_composite_credit_summary.csv`
- `moran_i_composite_credit_by_year.png`

Older outputs from superseded scripts are archived in `output/deprecated/`.

## References

- Favara, G. and Imbs, J. (2015). Credit Supply and the Price of Housing. American Economic Review, 105(3), 958-992.
- Rice, T. and Strahan, P. (2010). Does Credit Competition Affect Small-Firm Finance? Journal of Finance, 65(3), 861-889.
- Anselin, L. (1996). The Moran scatterplot as an ESDA tool to assess local instability in spatial association.
