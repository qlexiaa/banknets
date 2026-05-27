# Bank Networks and Spatial Spillovers in Housing Markets

A spatial econometric extension of Favara & Imbs (2015) *"Credit Supply and the Price of Housing"* (AER). This project tests whether banking deregulation propagates through the geographic bank network — and whether the bank-network spatial structure explains more co-movement in house prices than simple geographic contiguity.

**Core question**: Does the spatial autocorrelation in house-price growth reflect geographic proximity (W_geo) or shared bank exposure (W_bank)?

---

## Directory structure

```
.
├── pipeline/               # Data construction pipeline — run scripts in numbered order
│   ├── 01_download_fdic.py         Download FDIC Summary of Deposits 1994-2005
│   ├── 02_merge_panel_geo.py       Attach TIGER county geometries to the panel
│   ├── 03_build_geo_weights.py     Build queen-contiguity W_geo from geometries
│   ├── 04_build_bank_weights.py    Build W_bank (binary) and W_bank_count from FDIC
│   └── 05_build_panel.py           Filter and sort the estimation-ready panel
│
├── analysis/               # Estimation and diagnostics
│   ├── utils.py                    Shared helpers (GAL parsing, row-standardisation)
│   ├── diagnostics.py              Moran's I on OLS residuals (credit and HPI outcomes)
│   ├── sem_pooled.py               Pooled GM spatial error model: W_geo vs W_bank
│   ├── panel_fe_error.py           Panel_FE_Error ML estimator, two-way FE
│   └── spatial_hpi_models.py       OLS + SEM + SAR for the full HPI reduced form
│
├── data/                   # All data files
│   ├── estimation_panel.csv        1,023 counties x 11 years (1995-2005), 35 cols
│   ├── W_geo_queen.gal             Queen contiguity weights (GAL format)
│   ├── W_bank_avg.npz              Bank-network weights, binary (scipy sparse CSR)
│   ├── W_bank_count_avg.npz        Bank-network weights, count-weighted
│   ├── county_order_Wgeo.csv       Row order for W_geo (1,023 FIPS codes)
│   ├── county_order_Wbank.csv      Row order for W_bank (same as W_geo)
│   ├── fdic_deposits_1994_2005.csv [gitignored — 64 MB, built by 01_download_fdic.py]
│   └── merged_panel.gpkg           [gitignored — 571 MB, built by 02_merge_panel_geo.py]
│
├── output/                 # Figures and result CSVs written by analysis scripts
├── Replication/            # Original Favara & Imbs Stata replication package
├── docs/                   # PDF references and thesis documents [gitignored]
├── requirements.txt
└── README.md
```

---

## Spatial weights

| Matrix | Definition | Sparsity |
|---|---|---|
| **W_geo** | Queen contiguity — counties share a border or corner point | ~99.4% |
| **W_bank** | Binary cosine similarity of HC-presence vectors, time-averaged 1994-2005 | ~5% |
| **W_bank_count** | Count-weighted variant (branch volumes as matrix entries) | ~5% |

W_bank cosine similarity: `w_cc' = B_cc' / sqrt(B_cc * B_c'c')` where `B = M @ M.T`. All matrices are diagonal-zeroed and row-standardised (each row sums to 1).

---

## Reproducing results

### Prerequisites

```bash
pip install -r requirements.txt
```

### Step 1 — Build the data (one-time)

Run from the project root in order:

```bash
python pipeline/01_download_fdic.py        # ~20 min — downloads from FDIC API
python pipeline/02_merge_panel_geo.py      # ~5 min  — downloads TIGER shapefile
python pipeline/03_build_geo_weights.py    # < 1 min
python pipeline/04_build_bank_weights.py   # ~3 min
python pipeline/05_build_panel.py          # < 1 min
```

Scripts 01 and 02 require an internet connection. All outputs land in `data/`.

### Step 2 — Run analysis

```bash
python analysis/diagnostics.py         # Moran's I; saves to output/diagnostics/
python analysis/sem_pooled.py          # Pooled GM-SEM, W_geo vs W_bank (binary + count)
python analysis/panel_fe_error.py      # Panel_FE_Error ML, two-way FE
python analysis/spatial_hpi_models.py  # Full HPI model: OLS + SEM + SAR
```

Results are printed to stdout. Run from the project root or from within `analysis/`.

---

## Key results

| Estimator | Sample | W_geo lambda | W_bank lambda |
|---|---|---|---|
| Pooled GM-SEM (two-way demeaned) | Full | 0.8058 | 0.9900* |
| Pooled GM-SEM (two-way demeaned) | Non-border | 0.8036 | 0.9900* |
| Panel_FE_Error ML | Full | 0.8268 | 0.9879 |
| Panel_FE_Error ML | Non-border | 0.8270 | 0.9869 |

*GM estimator clips lambda at ±0.99; ML gives interior estimates.

W_bank consistently yields a higher spatial error parameter than W_geo across all specifications, supporting the hypothesis that banking networks — not just geographic proximity — drive co-movement in house prices.

---

## References

- Favara, G. & Imbs, J. (2015). Credit Supply and the Price of Housing. *AER* 105(3), 958–992.
- Rice, T. & Strahan, P. (2010). Does Credit Competition Affect Small-Firm Finance? *JF* 65(3).
- Elhorst, J.P. (2003). Specification and Estimation of Spatial Panel Data Models. *Geographical Analysis* 35(4).
- Kelejian, H. & Prucha, I. (1998). A Generalised Spatial Two-Stage Least Squares Procedure. *JRSS-B* 60(2).
- Anselin, L. (1988). *Spatial Econometrics: Methods and Models.* Kluwer Academic.
