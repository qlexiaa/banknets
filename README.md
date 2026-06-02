# Bank Networks and Spatial Spillovers in Housing Markets

A spatial econometric study of banking deregulation and housing markets, building on Favara & Imbs (2015) *"Credit Supply and the Price of Housing"* (AER). This project tests whether banking deregulation propagates through the geographic bank network — and whether bank-network spatial structure explains more co-movement in house prices than simple geographic contiguity.

**Core question**: Does spatial autocorrelation in house-price growth reflect geographic proximity (W_geo) or shared bank-holding-company exposure (W_bank)?

---

## Directory structure

```
.
├── pipeline/                   # Data construction — run scripts in numbered order
│   ├── 01_download_fdic.py         Download FDIC Summary of Deposits 1994-2005
│   ├── 02_merge_panel_geo.py       Attach TIGER county geometries to the panel
│   ├── 03_build_geo_weights.py     Build queen-contiguity W_geo from geometries
│   ├── 04_build_bank_weights.py    Build W_bank_avg (binary + count) from FDIC
│   └── 05_build_panel.py           Filter and sort the estimation-ready panel
│
├── analysis/                   # All estimation and robustness scripts
│   ├── utils.py                    Shared helpers + spreg numpy-2.0 compatibility patch
│   ├── run_all.py                  Master runner — executes all analysis, saves to output/
│   │
│   ├── diagnostics.py              Moran's I on OLS residuals (credit and HPI outcomes)
│   ├── sem_pooled.py               Pooled GM spatial error model: W_geo vs W_bank
│   ├── panel_fe_error.py           Panel_FE_Error ML, two-way FE: W_geo vs W_bank_avg
│   ├── spatial_hpi_models.py       OLS + SEM + SAR for the full HPI reduced form (W_geo)
│   │
│   ├── panel_fe_knn.py             Density robustness: W_bank_knn (k=5) vs W_geo
│   ├── knn_crossover.py            k sweep (k=5..10,20): crossover where lambda_knn > lambda_geo
│   ├── bank_geo_overlap.py         Overlap between top-5 bank connections and geo neighbours
│   ├── bank_nongeo.py              Residual W_bank after removing geo-overlapping links
│   │
│   ├── lambda_time.py              Year-by-year ML_Error: lambda over time with W_bank_avg
│   ├── lambda_time_yearspecific.py Year-by-year ML_Error: lambda with year-specific W_bank_t
│   │
│   ├── rq1_inference.py            LR test, lambda gap significance, OLS vs SEM beta comparison
│   ├── rq1_lm_tests.py             Panel LM tests (error, lag, robust) for W_geo and W_bank
│   ├── rq1_four_w_comparison.py    Formal comparison of all four W matrices with LR tests
│   ├── rq1_jtest.py                Davidson-MacKinnon J-test for competing spatial weights
│   └── rq1_composite_w.py          Profile log-likelihood over W(alpha) = alpha*W_geo + (1-alpha)*W_bank
│
├── data/                       # All data files
│   ├── estimation_panel.csv        1,023 counties x 11 years (1995-2005), 35 variables
│   ├── W_geo_queen.gal             Queen contiguity weights (GAL format)
│   ├── W_bank_avg.npz              Bank-network weights, binary cosine similarity, 1994-2005 avg
│   ├── W_bank_count_avg.npz        Count-weighted variant (branch volumes as matrix entries)
│   ├── W_bank_nonGeo.npz           W_bank_avg with all geo-neighbour pairs zeroed out
│   ├── W_bank_yearly/              Year-specific matrices W_bank_1995.npz ... W_bank_2005.npz
│   ├── county_order_Wgeo.csv       Row order for all W matrices (1,023 FIPS codes)
│   ├── county_order_Wbank.csv      Alias of county_order_Wgeo.csv (same order)
│   ├── fdic_deposits_1994_2005.csv [gitignored — ~64 MB, built by 01_download_fdic.py]
│   └── merged_panel.gpkg           [gitignored — ~571 MB, built by 02_merge_panel_geo.py]
│
├── output/                     # All results written here by run_all.py (CSV + PNG)
├── Replication/                # Original Favara & Imbs Stata replication package
├── docs/                       # PDF references [gitignored]
├── requirements.txt
└── README.md
```

---

## Spatial weights

| Matrix | Definition | Sparsity | Avg neighbours |
|---|---|---|---|
| **W_geo** | Queen contiguity — counties share a border or corner | ~99.7% | 3.4 |
| **W_bank_avg** | Binary HC-presence cosine similarity, averaged 1994-2005 | ~60% | 412 |
| **W_bank_t** | Same as W_bank_avg but built from a single year's FDIC data | varies | 127–381 |
| **W_bank_knn** | W_bank_avg truncated to top-k connections per row, re-standardised | ~99.5% (k=5) | k |
| **W_bank_nonGeo** | W_bank_avg with all geo-neighbour pairs zeroed out | ~60% | 408 |

Cosine similarity construction: `w_cc' = B_cc' / sqrt(B_cc * B_c'c')` where `B = M @ M.T` and `M[c,h] = 1` if holding company `h` has any branch in county `c`. All matrices are diagonal-zeroed and row-standardised.

---

## Reproducing results

### Prerequisites

```bash
pip install -r requirements.txt
```

### Step 1 — Build the data (one-time, requires internet for scripts 01 and 02)

```bash
python pipeline/01_download_fdic.py        # ~20 min — downloads from FDIC API
python pipeline/02_merge_panel_geo.py      # ~5  min — downloads TIGER shapefile
python pipeline/03_build_geo_weights.py    # < 1 min
python pipeline/04_build_bank_weights.py   # ~3  min
python pipeline/05_build_panel.py          # < 1 min
```

### Step 2 — Run all analysis

```bash
python analysis/run_all.py
```

This runs every analysis module in sequence with no console output. All results are saved to `output/` as CSV files and PNG plots. Individual scripts can still be run standalone (they print results to stdout).

---

## Analysis modules

| Script | What it does |
|---|---|
| `panel_fe_error.py` | Panel_FE_Error ML estimator with county FE + year dummies; W_geo and W_bank_avg for full and non-border samples |
| `spatial_hpi_models.py` | OLS (state-clustered SE) + SEM (GM) + SAR (ML) for the Favara-Imbs HPI reduced form using W_geo |
| `panel_fe_knn.py` | Density robustness: truncates W_bank to top-k=5 connections per county, tests whether high lambda survives |
| `knn_crossover.py` | Sweeps k=5..10,20 to find the crossover k where lambda_bank_knn first exceeds lambda_geo |
| `bank_geo_overlap.py` | Per-county overlap fraction between top-5 bank connections and geographic neighbours |
| `bank_nongeo.py` | Builds W_bank_nonGeo by zeroing geo-overlapping entries; tests whether purely non-geographic bank links transmit shocks |
| `lambda_time.py` | Cross-sectional ML_Error year by year (1997-2005) using the fixed W_bank_avg; tests for time trend in lambda |
| `lambda_time_yearspecific.py` | Same but uses year-specific W_bank_t matrices; properly tests the IBBEA consolidation hypothesis |
| `rq1_inference.py` | LR test (W_bank vs W_geo), lambda gap z-test, and OLS vs SEM beta comparison table |
| `rq1_lm_tests.py` | Panel LM and robust LM tests (error, lag) for all W/sample combinations; determines SEM vs SAR |
| `rq1_four_w_comparison.py` | Panel_FE_Error for all four W matrices (W_geo, W_bank_bin, W_bank_count, W_bank_nonGeo); reports lambda gap and LR test vs W_geo |
| `rq1_jtest.py` | Davidson-MacKinnon J-test in both directions for each W pair; classifies W_geo vs W_bank as one preferred, other preferred, or indeterminate |
| `rq1_composite_w.py` | Profiles Panel_FE_Error log-likelihood over W(alpha) = alpha*W_geo + (1-alpha)*W_bank; finds ML mixing weight alpha* |

---

## Key results

### Primary: Panel_FE_Error ML, two-way FE

| Sample | W_geo lambda | W_bank_avg lambda | Gap | z-stat |
|---|---|---|---|---|
| Full | 0.827 | **0.988** | +0.161 | 37.6 |
| Non-border | 0.832 | **0.987** | +0.156 | 34.3 |

W_bank_avg yields lambda ~0.16 higher than W_geo in every specification (p ≈ 0 one-sided). The deregulation coefficient beta(Linter_bra) ≈ 2.31 under W_geo and ≈ 2.40 under W_bank (both highly significant), indicating that controlling for bank-network spatial correlation raises the estimated deregulation effect.

### Four-W comparison (rq1_four_w_comparison)

| Sample | W | lambda | beta | Gap vs W_geo | LR vs W_geo |
|---|---|---|---|---|---|
| Full | W_geo | 0.827 | 2.309 | — | — |
| Full | W_bank_bin | 0.986 | 2.415 | +0.159 (z=35.4) | −8,977 |
| Full | W_bank_count | 0.986 | 2.411 | +0.159 (z=37.0) | −8,707 |
| Full | W_bank_nonGeo | 0.979 | 2.425 | +0.152 (z=27.9) | −8,809 |
| Non-border | W_geo | 0.832 | 2.144 | — | — |
| Non-border | W_bank_bin | 0.985 | 2.499 | +0.154 (z=32.1) | −6,261 |
| Non-border | W_bank_count | 0.985 | 2.497 | +0.153 (z=33.6) | −5,972 |
| Non-border | W_bank_nonGeo | 0.958 | 2.514 | +0.126 (z=14.3) | −6,290 |

W_bank has higher lambda in all specifications but a lower log-likelihood than W_geo. The two matrices capture different dimensions of spatial structure: W_bank induces stronger error correlation but W_geo provides a better overall fit.

### Composite W profile (rq1_composite_w)

| Pair | Sample | alpha* | lambda at alpha* | logll gain vs W_geo |
|---|---|---|---|---|
| W_geo + W_bank_bin | Full | 0.85 | 0.937 | +108.5 |
| W_geo + W_bank_bin | Non-border | 0.80 | 0.976 | +116.6 |
| W_geo + W_bank_count | Full | 0.85 | 0.934 | +125.2 |
| W_geo + W_bank_count | Non-border | 0.80 | 0.968 | +135.7 |

The maximum-likelihood mixing weight alpha* is 0.85 (full) / 0.80 (non-border), meaning the optimal W gives ~85% weight to geographic contiguity and ~15–20% to bank-network overlap. This composite W yields higher log-likelihood than either pure matrix, confirming that both channels are independently informative.

### J-test for competing W matrices (rq1_jtest)

All six W-pair / sample combinations yield **Indeterminate (both)**: both directions of the J-test reject (J-coefficient z > 8.5, p < 10^-16 in every case). Neither W_geo nor W_bank is the uniquely correct specification; each contains spatial prediction power not captured by the other, consistent with the composite W result that both matrices contribute.

### LM diagnostic tests (rq1_lm_tests)

For all W / sample combinations, the robust LM-error statistic exceeds the robust LM-lag statistic (rLM-error >> rLM-lag, both p ≈ 0), pointing to **SEM** as the preferred specification over SAR. This holds for both W_geo and W_bank, and for full and non-border samples.

### Density robustness (panel_fe_knn, knn_crossover)

| k | Density | lambda_full | Gap vs W_geo |
|---|---|---|---|
| 5 | 0.48% | 0.797 | −0.029 |
| 7 | 0.66% | 0.828 | **+0.001** ← crossover |
| 10 | 0.94% | 0.861 | +0.034 |
| 20 | 1.84% | 0.895 | +0.068 |
| avg (full) | 40.3% | 0.988 | +0.161 |

The lambda gap over W_geo turns positive at k=7 (full sample). The high lambda under W_bank_avg is not purely a density artefact — it reflects genuine network transmission, but only once enough bank connections are included per county.

### Bank-geography overlap

The top-5 bank connections of a typical county overlap with ~35% of its geographic neighbours (mean 0.35, median 0.40). Bank and geographic networks are related but largely distinct: 65% of the strongest bank links cross to non-contiguous counties.

### Non-geographic bank transmission (bank_nongeo)

After removing all W_bank entries between geographic neighbours (0.77% of pairs), lambda under W_bank_nonGeo remains 0.979 (full) and 0.958 (non-border) — both well above W_geo's 0.827. The high lambda under W_bank is driven by the 99% of connections that are non-geographic, confirming a genuine bank-network transmission channel.

### IBBEA consolidation over time (lambda_time_yearspecific)

Year-specific W_bank_t matrices (built from each year's FDIC data) show lambda_bank rising at +0.016/year (p=0.029) vs lambda_geo at +0.008/year (p=0.047) over 1997-2005, with the gap widening at +0.008/year (p=0.042). The bank-network W matrix also densifies from 178K county pairs in 1997 to 390K in 2005 (3x growth), consistent with the post-IBBEA consolidation wave.

---

## Output files

All saved to `output/` by `run_all.py`:

| File | Contents |
|---|---|
| `panel_fe_results.csv` | Panel_FE_Error estimates: beta, SE, lambda, SE(lambda), p-values |
| `spatial_hpi_main_results.csv` | OLS / SEM / SAR estimates for the full HPI equation |
| `spatial_hpi_morans_i.csv` | Moran's I on OLS residuals by year |
| `four_w_comparison.csv` | Panel_FE_Error for all four W matrices; lambda gap and LR vs W_geo |
| `lr_test_results.csv` | Likelihood ratio test: W_bank vs W_geo |
| `lambda_inference_results.csv` | Lambda gap: point estimate, SE, z-stat, p-value (one-sided) |
| `beta_comparison_table.csv` | Deregulation beta: OLS (state-clustered), SEM W_geo, SEM W_bank |
| `lm_test_results.csv` | Panel LM and robust LM error/lag statistics; SEM vs SAR decision |
| `jtest_results.csv` | J-test coefficients, z-stats, and verdicts for all W pairs |
| `composite_w_optima.csv` | alpha*, lambda and logll at alpha*, logll improvement vs W_geo |
| `composite_w_results.csv` | Full grid of logll / lambda / beta over alpha in [0,1] |
| `composite_w_profiles.png` | Log-likelihood profile curves over alpha for all W pairs |
| `knn_density_model_results.csv` | Panel_FE_Error estimates for W_geo and W_bank_knn(k=5) |
| `knn_density_comparison.csv` | Lambda comparison with verdict (gap GONE / HALVED / HOLDS / REVERSED) |
| `knn_sweep_results.csv` | Lambda and gap for k = 5, 6, 7, 8, 9, 10, 20 |
| `knn_crossover_summary.csv` | Crossover k (first k where lambda_knn > lambda_geo) by sample |
| `geo_bank_overlap_stats.csv` | Summary stats of per-county bank-geo overlap fraction |
| `geo_bank_overlap_distribution.csv` | County counts at each overlap level (0, 0.2, 0.4, 0.6, 0.8, 1.0) |
| `bank_nongeo_results.csv` | Panel_FE_Error estimates for W_bank_nonGeo |
| `bank_nongeo_matrix_stats.csv` | Sparsity comparison: W_geo / W_bank_avg / W_bank_nonGeo |
| `lambda_time_avg_results.csv` | Year-by-year lambda (1997-2005), W_bank_avg fixed matrix |
| `lambda_time_avg_trends.csv` | OLS trend slopes and p-values for lambda over time |
| `lambda_time_yearspecific_results.csv` | Year-by-year lambda with year-specific W_bank_t |
| `lambda_time_yearspecific_network.csv` | Annual W_bank_t density: HCs, pairs, sparsity |
| `lambda_time_yearspecific_trends.csv` | OLS trend slopes for year-specific specification |
| `overlap_histogram.png` | Histogram of per-county bank-geo overlap fraction |
| `lambda_time_variation.png` | Lambda over time: W_geo vs W_bank_avg with 95% CI |
| `lambda_time_variation_yearspecific.png` | Lambda over time: W_geo vs W_bank_t with gap panel |

---

## References

- Favara, G. & Imbs, J. (2015). Credit Supply and the Price of Housing. *AER* 105(3), 958–992.
- Rice, T. & Strahan, P. (2010). Does Credit Competition Affect Small-Firm Finance? *JF* 65(3), 861–889.
- Elhorst, J.P. (2003). Specification and Estimation of Spatial Panel Data Models. *Geographical Analysis* 35(4), 301–328.
- Kelejian, H. & Prucha, I. (1998). A Generalised Spatial Two-Stage Least Squares Procedure. *JRSS-B* 60(2), 509–524.
- Anselin, L. (1988). *Spatial Econometrics: Methods and Models.* Kluwer Academic Publishers.
- Davidson, R. & MacKinnon, J.G. (1981). Several Tests for Model Specification in the Presence of Alternative Hypotheses. *Econometrica* 49(3), 781–793.
- Conley, T.G. & Topa, G. (2002). Socio-Economic Distance and Spatial Patterns in Unemployment. *Journal of Applied Econometrics* 17(4), 303–327.
- Riegle-Neal Interstate Banking and Branching Efficiency Act (1994). Pub.L. 103-328.
