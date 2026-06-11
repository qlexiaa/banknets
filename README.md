# Bank Networks and Credit Growth Spillovers

This project replicates and spatially extends Favara & Imbs (2015), focusing
on whether interstate banking deregulation propagates through shared bank
holding-company exposure across counties. The main dependent variable is
county-level credit growth, `Dl_nloans_b`.

The core comparison is between seven spatial weight specifications:

- `W_geo`: queen contiguity between counties (geographic benchmark)
- `W_bank`: cosine similarity in bank HHC presence, time-averaged 1994–2005
- `W_bank_count`: branch-count-weighted HHC cosine similarity
- `W_bank_binary`: binarised bank overlap (nonzero → 1)
- `W_bank_knn4`: top-4 bank network neighbours (k-NN truncation)
- `W_bank_nonGeo`: bank network with geographic-neighbour links zeroed out
- `W_bank_interstate`: cross-state bank links only
- `W_bank_intrastate`: within-state bank links only

## Repository Layout

```text
analysis/
  run_all.py                         Active 18-step analysis pipeline
  utils.py                           Weight-matrix helpers and spreg compatibility patch
  panel_data.py                      Shared credit-panel loading, get_samples(), placebo loader
  generate_report.py                 Builds output/results_report.md from CSV outputs
  w_variants.py                      Centralised W matrix catalogue (all 7 bank variants)

  diagnostics/
    moran_baseline.py                Baseline Moran's I under W_geo, credit
    moran_bank_variants.py           Moran's I under W_bank variants and KNN truncations
    network_overlap.py               Bank-geography top-link overlap diagnostics

  sem/
    sem_credit.py                    Main Panel_FE_Error credit model, W_geo and 7 bank variants
    sem_w_variants.py                Credit SEM across all 7 bank weight matrices
    sem_link_restrictions.py         nonGeo, interstate, and intrastate bank-link restrictions
    sem_knn_sweep.py                 KNN sweep, k = 1 to 20

  sar/
    sar_credit.py                    Panel_FE_Lag robustness check, all W variants
    sar_iv_credit.py                 IV-SAR following Kelejian & Prucha (1998)
    multiplier_decomposition.py      SEM direct/indirect effects and distance decay

  model_selection/
    lm_tests.py                      LM error/lag diagnostics for all W variants
    jtest.py                         Davidson-MacKinnon J-tests, W_geo vs 7 bank variants
    composite_w.py                   Composite W(alpha) profile-likelihood sweep, 7 pairs

  inference/
    conley_se.py                     Conley (1999) and two-way HAC standard-error comparison
    fgls.py                          FGLS vs OLS point-estimator comparison

  extensions/
    slx_exposure.py                  Bank-network SLX exposure augmentation of F&I first stage
    multiplier_counterfactual.py     SAR multiplier decomposition and naive vs network counterfactual

  deprecated/                        Archived superseded analyses

pipeline/                            Data-build scripts
data/                                Estimation panel and spatial weight matrices
output/                              Current results, figures, and diagnostics
Replication/                         Favara-Imbs replication package (unmodified)
```

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

The active pipeline runs, in order:

1. Baseline Moran's I under `W_geo`, credit
2. Moran's I under `W_bank` variants
3. Bank-geography link overlap
4. LM error/lag diagnostics for credit growth
5. Main credit Panel_FE_Error, `W_geo` vs all 7 bank variants
6. Credit SEM across all 7 bank weight matrices
7. nonGeo, interstate, and intrastate bank-link restrictions
8. KNN sweep, `k = 1..20`
9. Davidson-MacKinnon J-tests, W_geo vs 7 bank variants
10. Composite `W(alpha)` profile-likelihood sweep, 7 pairs (Pairs A–G)
11. Panel_FE_Lag robustness, all W variants
12. IV-SAR following Kelejian & Prucha (1998)
13. Direct/indirect effects and distance decay
14. Standard-error estimator comparison
15. FGLS vs OLS point-estimator comparison
16. Bank-network SLX exposure augmentation
17. SAR multiplier decomposition and counterfactual

## Samples

Three samples are estimated throughout:

- **Full**: all counties in the estimation panel (~1,015 counties)
- **Border**: counties in MSAs that straddle state borders (`border == 1`, 273 counties
  in 43 MSAs), following Favara & Imbs (2015) border-discontinuity identification
- **NonBorder**: counties NOT in border-straddling MSAs (`border == 0`); serves as a
  complementary comparison group to the Border sample

## Main Credit Results

Panel fixed-effects SEM estimates show much stronger spatial error dependence
under the bank network than under geographic contiguity:

| Sample | W_geo λ | W_bank λ | Gap |
|---|---:|---:|---:|
| Full   | 0.180 | 0.699 | 0.519 |
| Border | 0.170 | 0.671 | 0.500 |

The SAR robustness check gives the same qualitative pattern:

| Sample | W_geo ρ | W_bank ρ | Gap |
|---|---:|---:|---:|
| Full   | 0.179 | 0.685 | 0.506 |
| Border | 0.167 | 0.648 | 0.481 |

## Companion Checks

**Non-geographic bank channel:** After removing all bank links between geographic
neighbours, `W_bank_nonGeo` still produces λ = 0.485 (full) and 0.495 (border).
The bank channel is not disguised geographic adjacency.

**Interstate bank links:** `W_bank_interstate` (cross-state links only) produces a
positive and significant λ, consistent with post-IBBEA cross-state branch expansion
as the transmission mechanism.

**Intrastate placebo:** `W_bank_intrastate` (within-state links only) produces a
smaller λ than the full bank network, supporting the cross-state channel interpretation.

**SLX exposure:** The interstate bank-network exposure variable
`E_{c,t} = sum_{c'} w^{interstate}_{cc'} * D_{s(c'),t}` is positive and significant
in the augmented credit regression, and near-zero for the non-bank placebo DV
(`Dl_nloans_pl`), consistent with the exclusion restriction.

## Current Outputs

Key files in `output/`:

- `panel_fe_credit_results.csv`
- `four_w_comparison_credit.csv`
- `bank_nongeo_credit_results.csv`
- `bank_interstate_credit_results.csv`
- `bank_intrastate_credit_results.csv`
- `knn_sweep_credit_results.csv`
- `lm_diagnostics_credit.csv`
- `jtest_credit_results.csv`
- `composite_w_credit_results.csv`
- `composite_w_credit_optima.csv`
- `sar_robustness_credit.csv`
- `sar_iv_results.csv`
- `spatial_multiplier_decomposition.csv`
- `conley_se_comparison.csv`
- `fgls_comparison.csv`
- `geo_bank_overlap_stats.csv`
- `slx_exposure_results.csv`
- `multiplier_counterfactual.csv`

Diagnostic outputs are in `output/diagnostics/`.

## References

- Favara, G. and Imbs, J. (2015). *American Economic Review*, 105(3), 958–992.
- Kelejian, H. H. and Prucha, I. R. (1998). A generalized spatial two-stage least squares procedure for estimating a spatial autoregressive model with autoregressive disturbances. *Journal of Real Estate Finance and Economics*, 17(1), 99–121.
- LeSage, J. and Pace, R. K. (2009). *Introduction to Spatial Econometrics*. CRC Press.
- Conley, T. G. (1999). GMM estimation with cross sectional dependence. *Journal of Econometrics*, 92(1), 1–45.
- Colella, F., Lalive, R., Sakalli, S. O., and Thoenig, M. (2019). Inference with arbitrary clustering. *IZA Discussion Paper No. 12584*.
- Davidson, R. and MacKinnon, J. G. (1981). Several tests for model specification in the presence of alternative hypotheses. *Econometrica*, 49(3), 781–793.
