# Bank Networks and Spatial Spillovers in Housing Markets

A spatial econometric extension of Favara & Imbs (2015) *"Credit Supply and the Price of Housing"* (AER). This project tests whether banking deregulation propagates through the geographic bank network — and whether bank-network spatial structure explains more co-movement in house prices than simple geographic contiguity.

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
│   └── lambda_time_yearspecific.py Year-by-year ML_Error: lambda with year-specific W_bank_t
│
├── data/                       # All data files
│   ├── estimation_panel.csv        1,023 counties x 11 years (1995-2005), 35 variables
│   ├── W_geo_queen.gal             Queen contiguity weights (GAL format)
│   ├── W_bank_avg.npz              Bank-network weights, binary cosine similarity, 1994-2005 avg
│   ├── W_bank_count_avg.npz        Count-weighted variant (branch volumes as matrix entries)
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
| `knn_crossover.py` | Sweeps k=5..10,20 to find the crossover k where lambda\_bank\_knn first exceeds lambda\_geo |
| `bank_geo_overlap.py` | Per-county overlap fraction between top-5 bank connections and geographic neighbours |
| `bank_nongeo.py` | Builds W\_bank\_nonGeo by zeroing geo-overlapping entries; tests whether purely non-geographic bank links transmit shocks |
| `lambda_time.py` | Cross-sectional ML\_Error year by year (1997-2005) using the fixed W\_bank\_avg; tests for time trend in lambda |
| `lambda_time_yearspecific.py` | Same but uses year-specific W\_bank\_t matrices; properly tests the IBBEA consolidation hypothesis |

---

## Key results

### Primary: Panel_FE_Error ML, two-way FE

| Sample | W_geo lambda | W_bank_avg lambda |
|---|---|---|
| Full | 0.827 | **0.988** |
| Non-border | 0.832 | **0.987** |

W\_bank\_avg yields lambda ~0.16 higher than W\_geo in every specification, indicating that shared bank holding company exposure transmits spatial error correlation significantly more strongly than geographic adjacency alone.

### Density robustness (panel_fe_knn, knn_crossover)

| k | Density | lambda\_full | Gap vs W\_geo |
|---|---|---|---|
| 5 | 0.48% | 0.797 | −0.029 |
| 7 | 0.66% | 0.828 | **+0.001** ← crossover |
| 10 | 0.94% | 0.861 | +0.034 |
| 20 | 1.84% | 0.895 | +0.068 |
| avg (full) | 40.3% | 0.988 | +0.161 |

The lambda gap over W\_geo turns positive at k=7 (full sample). The high lambda under W\_bank\_avg is not purely a density artefact — it reflects genuine network transmission, but only once enough bank connections are included per county.

### Bank-geography overlap

The top-5 bank connections of a typical county overlap with ~35% of its geographic neighbours (mean 0.35, median 0.40). Bank and geographic networks are related but largely distinct: 65% of the strongest bank links cross to non-contiguous counties.

### Non-geographic bank transmission (bank_nongeo)

After removing all W\_bank entries between geographic neighbours (0.77% of pairs), lambda under W\_bank\_nonGeo remains 0.979 (full) and 0.958 (non-border) — both well above W\_geo's 0.827. The high lambda under W\_bank is driven by the 99% of connections that are non-geographic, confirming a genuine bank-network transmission channel.

### IBBEA consolidation over time (lambda_time_yearspecific)

Year-specific W\_bank\_t matrices (built from each year's FDIC data) show lambda\_bank rising at +0.016/year (p=0.029) vs lambda\_geo at +0.008/year (p=0.047) over 1997-2005, with the gap widening at +0.008/year (p=0.042). The bank-network W matrix also densifies from 178K county pairs in 1997 to 390K in 2005 (3x growth), consistent with the post-IBBEA consolidation wave.

---

## Output files

All saved to `output/` by `run_all.py`:

| File | Contents |
|---|---|
| `panel_fe_results.csv` | Panel_FE_Error estimates: beta, SE, lambda, SE(lambda), p-values |
| `spatial_hpi_main_results.csv` | OLS / SEM / SAR estimates for the full HPI equation |
| `spatial_hpi_morans_i.csv` | Moran's I on OLS residuals by year |
| `knn_density_model_results.csv` | Panel_FE_Error estimates for W\_geo and W\_bank\_knn(k=5) |
| `knn_density_comparison.csv` | Lambda comparison with verdict (gap GONE / HALVED / HOLDS / REVERSED) |
| `knn_sweep_results.csv` | Lambda and gap for k = 5, 6, 7, 8, 9, 10, 20 |
| `knn_crossover_summary.csv` | Crossover k (first k where lambda\_knn > lambda\_geo) by sample |
| `geo_bank_overlap_stats.csv` | Summary stats of per-county bank-geo overlap fraction |
| `geo_bank_overlap_distribution.csv` | County counts at each overlap level (0, 0.2, 0.4, 0.6, 0.8, 1.0) |
| `bank_nongeo_results.csv` | Panel_FE_Error estimates for W\_bank\_nonGeo |
| `bank_nongeo_matrix_stats.csv` | Sparsity comparison: W\_geo / W\_bank\_avg / W\_bank\_nonGeo |
| `lambda_time_avg_results.csv` | Year-by-year lambda (1997-2005), W\_bank\_avg fixed matrix |
| `lambda_time_avg_trends.csv` | OLS trend slopes and p-values for lambda over time |
| `lambda_time_yearspecific_results.csv` | Year-by-year lambda with year-specific W\_bank\_t |
| `lambda_time_yearspecific_network.csv` | Annual W\_bank\_t density: HCs, pairs, sparsity |
| `lambda_time_yearspecific_trends.csv` | OLS trend slopes for year-specific specification |
| `overlap_histogram.png` | Histogram of per-county bank-geo overlap fraction |
| `lambda_time_variation.png` | Lambda over time: W\_geo vs W\_bank\_avg with 95% CI |
| `lambda_time_variation_yearspecific.png` | Lambda over time: W\_geo vs W\_bank\_t with gap panel |

---

## References

- Favara, G. & Imbs, J. (2015). Credit Supply and the Price of Housing. *AER* 105(3), 958–992.
- Rice, T. & Strahan, P. (2010). Does Credit Competition Affect Small-Firm Finance? *JF* 65(3), 861–889.
- Elhorst, J.P. (2003). Specification and Estimation of Spatial Panel Data Models. *Geographical Analysis* 35(4), 301–328.
- Kelejian, H. & Prucha, I. (1998). A Generalised Spatial Two-Stage Least Squares Procedure. *JRSS-B* 60(2), 509–524.
- Anselin, L. (1988). *Spatial Econometrics: Methods and Models.* Kluwer Academic Publishers.
- Riegle-Neal Interstate Banking and Branching Efficiency Act (1994). Pub.L. 103-328.
