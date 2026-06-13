"""
w_density_summary.py
====================
Summarises sparsity / density statistics for every W matrix used in the
analysis pipeline.

For each matrix reports:
  N                    -- number of counties (rows)
  nnz                  -- non-zero off-diagonal entries
  density_pct          -- 100 * nnz / (N*(N-1))
  mean_nbrs            -- average number of non-zero off-diagonal links per county
  median_nbrs          -- median
  pct_isolated         -- % of rows with zero off-diagonal weight
  mean_nonzero_weight  -- mean value of non-zero off-diagonal entries
  median_nonzero_weight-- median value of non-zero off-diagonal entries

Matrices covered:
  W_geo, W_bank, W_bank_count, W_bank_bin, W_bank_knn4,
  W_bank_nonGeo, W_bank_interstate, W_bank_intrastate,
  W_bank_knn_{k}  (k = 1, 2, 4, 8, 15, 20)

Two additional rows are appended that describe the UN-row-standardised
time-averaged cosine W_bank split by pair type:
  "W_bank same-state entries"  -- (i,j) pairs where state(i) = state(j)
  "W_bank cross-state entries" -- (i,j) pairs where state(i) != state(j)

These rows use the raw cosine similarity values (before row-standardisation),
making mean_nonzero_weight / median_nonzero_weight directly interpretable as
the degree of bank portfolio overlap between county pairs of each type.

Output
------
  output/w_density_summary.csv
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse

sys.path.insert(0, str(Path(__file__).parents[1]))
from utils import row_standardize
from w_variants import load_w_geo, load_bank_variants

ROOT        = Path(__file__).parents[2]
COUNTY_PATH = ROOT / "data" / "county_order_Wgeo.csv"
FDIC_PATH   = ROOT / "data" / "fdic_deposits_1994_2005.csv"

KNN_SELECTED = [1, 2, 4, 8, 15, 20]


# ── Per-matrix statistics ─────────────────────────────────────────────────────

def _row_stats(W_sp, label):
    """Compute density and weight statistics for a sparse weight matrix.

    Parameters
    ----------
    W_sp : scipy sparse matrix
        Weight matrix (any format).  The diagonal is zeroed before analysis.
    label : str
        Row label for the summary table.

    Returns
    -------
    dict with keys: matrix, N, nnz, density_pct, mean_nbrs, median_nbrs,
                    pct_isolated, mean_nonzero_weight, median_nonzero_weight.
    """
    W = W_sp.copy().astype(np.float64)
    W = scipy.sparse.csr_matrix(W)
    W.setdiag(0)
    W.eliminate_zeros()

    N   = W.shape[0]
    nnz = W.nnz
    tot = N * (N - 1)

    nbrs_per_row = np.array((W > 0).sum(axis=1)).flatten().astype(float)

    nz_vals = W.data  # all stored nonzero values (diagonal already zeroed)
    mean_w   = float(nz_vals.mean())   if len(nz_vals) > 0 else np.nan
    median_w = float(np.median(nz_vals)) if len(nz_vals) > 0 else np.nan

    return {
        "matrix":               label,
        "N":                    N,
        "nnz":                  nnz,
        "density_pct":          100.0 * nnz / tot if tot > 0 else 0.0,
        "mean_nbrs":            float(nbrs_per_row.mean()),
        "median_nbrs":          float(np.median(nbrs_per_row)),
        "pct_isolated":         100.0 * (nbrs_per_row == 0).sum() / N,
        "mean_nonzero_weight":  mean_w,
        "median_nonzero_weight": median_w,
    }


# ── KNN builder ───────────────────────────────────────────────────────────────

def _build_knn(W_sp, k):
    """Keep top-k weights per row, row-standardise.  Returns scipy CSR."""
    W = W_sp.toarray().astype(np.float64)
    np.fill_diagonal(W, 0.0)
    N     = W.shape[0]
    W_knn = np.zeros_like(W)
    for i in range(N):
        row = W[i]
        nz  = np.count_nonzero(row)
        if nz == 0:
            continue
        if nz <= k:
            W_knn[i] = row
        else:
            top_k = np.argpartition(row, -k)[-k:]
            W_knn[i, top_k] = row[top_k]
    np.fill_diagonal(W_knn, 0.0)
    rs = W_knn.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        W_knn = np.where(rs > 0, W_knn / rs, 0.0)
    return scipy.sparse.csr_matrix(W_knn)


# ── Raw cosine matrix (un-row-standardised, time-averaged) ────────────────────

def _build_raw_avg_cosine(county_order):
    """Reconstruct the time-averaged raw cosine matrix for W_bank.

    data/W_bank_avg.npz stores the row-standardised cosine matrix.
    This helper rebuilds the underlying raw cosine similarity values
    (before row-standardisation) by re-processing FDIC branch data
    year-by-year, mirroring pipeline/04_build_bank_weights.py but
    omitting the row-standardisation step inside each year.

    Returns
    -------
    W_raw : ndarray (N, N), dtype float64
        Time-averaged raw cosine similarities, zero diagonal.
        Entry (i, j) = mean_t[ cos_t(i,j) ] where cos_t(i,j) is the binary
        BHC portfolio cosine similarity between counties i and j in year t.
    years : list of int
        Calendar years averaged over.
    """
    N          = len(county_order)
    county_idx = {fips: k for k, fips in enumerate(county_order)}
    county_set = set(county_order)

    print("  Loading FDIC data for raw cosine construction ...", flush=True)
    fdic = pd.read_csv(FDIC_PATH, dtype={"STCNTYBR": str})
    fdic["fips5"] = fdic["STCNTYBR"].str.zfill(5)
    fdic = (fdic[fdic["fips5"].isin(county_set) &
                 fdic["RSSDHCR"].notna() &
                 (fdic["RSSDHCR"] != 0)]
            .copy())

    years  = sorted(fdic["year"].dropna().unique())
    W_sum  = np.zeros((N, N), dtype=np.float64)

    for yr in years:
        df_yr = fdic[fdic["year"] == yr]
        pairs = df_yr[["fips5", "RSSDHCR"]].drop_duplicates().reset_index(drop=True)
        hcs   = pairs["RSSDHCR"].unique()
        n_hcs = len(hcs)
        if n_hcs == 0:
            continue
        hc_map   = {hc: i for i, hc in enumerate(hcs)}
        rows_idx = pairs["fips5"].map(county_idx).values
        cols_idx = pairs["RSSDHCR"].map(hc_map).values
        M = scipy.sparse.csr_matrix(
            (np.ones(len(pairs), dtype=np.float64), (rows_idx, cols_idx)),
            shape=(N, n_hcs),
        )
        B     = (M @ M.T).toarray().astype(np.float64)
        d     = B.diagonal().copy()
        denom = np.sqrt(np.outer(d, d))
        with np.errstate(divide="ignore", invalid="ignore"):
            W_yr = np.where(denom > 0, B / denom, 0.0)
        np.fill_diagonal(W_yr, 0.0)
        W_sum += W_yr

    W_raw = W_sum / len(years) if years else W_sum
    np.fill_diagonal(W_raw, 0.0)
    print(f"  Raw cosine averaged over {len(years)} years "
          f"({min(years)}-{max(years)}).", flush=True)
    return W_raw, [int(y) for y in years]


# ── State-split weight statistics ─────────────────────────────────────────────

def _state_split_rows(W_raw, county_order):
    """Compute density + weight stats for same-state and cross-state subsets.

    Uses the UN-row-standardised raw cosine matrix W_raw.

    Parameters
    ----------
    W_raw : ndarray (N, N)
        Raw cosine matrix, zero diagonal.
    county_order : list[str]
        FIPS5 codes; first two characters identify the state.

    Returns
    -------
    list of two dicts (same-state row, cross-state row) with all columns
    expected by the summary table.
    """
    N      = len(county_order)
    states = np.array([c[:2] for c in county_order])

    same_state_mask  = (states[:, None] == states[None, :])   # (N, N) bool
    np.fill_diagonal(same_state_mask, False)
    cross_state_mask = ~same_state_mask
    np.fill_diagonal(cross_state_mask, False)

    # Total eligible pairs for each category
    n_same_pairs  = int(same_state_mask.sum())
    n_cross_pairs = int(cross_state_mask.sum())

    def _subset_stats(label, mask, n_pairs_total):
        vals = W_raw[mask]                          # all entries in subset
        nz   = vals > 0
        nnz  = int(nz.sum())
        nz_vals = vals[nz]

        mean_w   = float(nz_vals.mean())     if nnz > 0 else np.nan
        median_w = float(np.median(nz_vals)) if nnz > 0 else np.nan

        # Per-county neighbour counts within this subset
        # nbrs_same[i] = # of counties in same state with nonzero cosine link
        W_masked = W_raw * mask.astype(np.float64)
        nbrs_per_row = (W_masked > 0).sum(axis=1).astype(float)

        return {
            "matrix":               label,
            "N":                    N,
            "nnz":                  nnz,
            "density_pct":          100.0 * nnz / n_pairs_total if n_pairs_total > 0 else 0.0,
            "mean_nbrs":            float(nbrs_per_row.mean()),
            "median_nbrs":          float(np.median(nbrs_per_row)),
            "pct_isolated":         100.0 * (nbrs_per_row == 0).sum() / N,
            "mean_nonzero_weight":  mean_w,
            "median_nonzero_weight": median_w,
        }

    return [
        _subset_stats("W_bank same-state entries",  same_state_mask,  n_same_pairs),
        _subset_stats("W_bank cross-state entries", cross_state_mask, n_cross_pairs),
    ]


# ── Public entry point ────────────────────────────────────────────────────────

def run(output_dir=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    co_df        = pd.read_csv(COUNTY_PATH, dtype={"fips5": str})
    county_order = co_df["fips5"].str.zfill(5).tolist()

    W_geo_all, _ = load_w_geo(county_order)
    bank_vars    = load_bank_variants(county_order, W_geo_all=W_geo_all)

    rows = []

    # ── Named standard variants ──────────────────────────────────────────────
    named = [
        ("W_geo",             W_geo_all),
        ("W_bank",            bank_vars["W_bank"]),
        ("W_bank_count",      bank_vars["W_bank_count"]),
        ("W_bank_bin",        bank_vars["W_bank_binary"]),
        ("W_bank_knn4",       bank_vars["W_bank_knn4"]),
        ("W_bank_nonGeo",     bank_vars["W_bank_nonGeo"]),
        ("W_bank_interstate", bank_vars["W_bank_interstate"]),
        ("W_bank_intrastate", bank_vars["W_bank_intrastate"]),
    ]
    for label, W in named:
        r = _row_stats(W, label)
        rows.append(r)
        print(f"  {label:<28}: density={r['density_pct']:.4f}%  "
              f"mean_nbrs={r['mean_nbrs']:.2f}  "
              f"mean_w={r['mean_nonzero_weight']:.4f}  "
              f"pct_isolated={r['pct_isolated']:.1f}%")

    # ── KNN variants at selected k values ────────────────────────────────────
    W_bank_rs = bank_vars["W_bank"]
    for k in KNN_SELECTED:
        W_knn = _build_knn(W_bank_rs, k)
        label = f"W_bank_knn_{k}"
        r = _row_stats(W_knn, label)
        rows.append(r)
        print(f"  {label:<28}: density={r['density_pct']:.4f}%  "
              f"mean_nbrs={r['mean_nbrs']:.2f}  "
              f"mean_w={r['mean_nonzero_weight']:.4f}  "
              f"pct_isolated={r['pct_isolated']:.1f}%")

    # ── Raw cosine: same-state vs cross-state rows ───────────────────────────
    print()
    print("  Computing raw cosine same-state / cross-state weight statistics ...")
    try:
        W_raw, avg_years = _build_raw_avg_cosine(county_order)
        split_rows = _state_split_rows(W_raw, county_order)
        for r in split_rows:
            rows.append(r)
            print(f"  {r['matrix']:<28}: density={r['density_pct']:.4f}%  "
                  f"nnz={r['nnz']:,}  "
                  f"mean_w={r['mean_nonzero_weight']:.4f}  "
                  f"median_w={r['median_nonzero_weight']:.4f}")
    except Exception as exc:
        print(f"  [WARN] Could not build raw cosine: {exc}")
        # Append placeholder rows so the CSV schema stays consistent
        for label in ("W_bank same-state entries", "W_bank cross-state entries"):
            rows.append({
                "matrix": label, "N": len(county_order),
                "nnz": np.nan, "density_pct": np.nan,
                "mean_nbrs": np.nan, "median_nbrs": np.nan,
                "pct_isolated": np.nan,
                "mean_nonzero_weight": np.nan,
                "median_nonzero_weight": np.nan,
            })

    # ── Print summary table ───────────────────────────────────────────────────
    W_TBL = 108
    print()
    print("=" * W_TBL)
    print("W Matrix Density Summary (including raw-cosine same/cross-state rows)")
    print("=" * W_TBL)
    print(f"{'Matrix':<32} {'N':>6}  {'density %':>10} {'mean nbrs':>10} "
          f"{'median nbrs':>12} {'% isolated':>10} "
          f"{'mean w':>8} {'med w':>8}")
    print("-" * W_TBL)
    for r in rows:
        mw  = f"{r['mean_nonzero_weight']:>8.4f}" if not pd.isna(r.get("mean_nonzero_weight", np.nan)) else f"{'--':>8}"
        mdw = f"{r['median_nonzero_weight']:>8.4f}" if not pd.isna(r.get("median_nonzero_weight", np.nan)) else f"{'--':>8}"
        print(f"{r['matrix']:<32} {r['N']:>6}  {r['density_pct']:>10.4f} "
              f"{r['mean_nbrs']:>10.2f} {r['median_nbrs']:>12.2f} "
              f"{r['pct_isolated']:>10.1f} {mw} {mdw}")
    print("=" * W_TBL)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if output_dir is not None:
        cols = [
            "matrix", "N", "nnz", "density_pct",
            "mean_nbrs", "median_nbrs", "pct_isolated",
            "mean_nonzero_weight", "median_nonzero_weight",
        ]
        pd.DataFrame(rows)[cols].to_csv(
            output_dir / "w_density_summary.csv", index=False)
        print(f"\nSaved w_density_summary.csv to {output_dir}")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    run(ROOT / "output")
