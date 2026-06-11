"""
w_density_summary.py
====================
Summarises sparsity / density statistics for every W matrix used in the
analysis pipeline.

For each matrix reports:
  N              -- number of counties (rows)
  density_pct    -- 100 * nnz(off-diag) / (N*(N-1))
  mean_nbrs      -- average off-diagonal links per county
  median_nbrs    -- median
  pct_isolated   -- % of rows with zero off-diagonal weight

Matrices covered:
  W_geo, W_bank, W_bank_count, W_bank_bin, W_bank_knn4,
  W_bank_nonGeo, W_bank_interstate, W_bank_intrastate,
  W_bank_knn_ (k = 1, 2, 4, 8, 15, 20)

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
WBANK_PATH  = ROOT / "data" / "W_bank.npz"

KNN_SELECTED = [1, 2, 4, 8, 15, 20]


def _row_stats(W_sp, label):
    """Compute density statistics for a sparse weight matrix."""
    W = W_sp.copy()
    W.setdiag(0)
    W.eliminate_zeros()

    N    = W.shape[0]
    nnz  = W.nnz
    tot  = N * (N - 1)

    row_sums = np.array(W.sum(axis=1)).flatten()
    nbrs_per_row = np.array((W > 0).sum(axis=1)).flatten().astype(float)

    return {
        "matrix":        label,
        "N":             N,
        "nnz":           nnz,
        "density_pct":   100.0 * nnz / tot if tot > 0 else 0.0,
        "mean_nbrs":     float(nbrs_per_row.mean()),
        "median_nbrs":   float(np.median(nbrs_per_row)),
        "pct_isolated":  100.0 * (nbrs_per_row == 0).sum() / N,
    }


def _build_knn(W_sp, k):
    """Keep top-k weights per row, row-standardise. Returns scipy CSR."""
    W = W_sp.toarray().astype(np.float64)
    np.fill_diagonal(W, 0.0)
    N = W.shape[0]
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


def run(output_dir=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    co_df        = pd.read_csv(COUNTY_PATH, dtype={"fips5": str})
    county_order = co_df["fips5"].str.zfill(5).tolist()

    W_geo_all, _ = load_w_geo(county_order)
    bank_vars    = load_bank_variants(county_order, W_geo_all=W_geo_all)

    rows = []

    # Named variants
    named = [
        ("W_geo",              W_geo_all),
        ("W_bank",             bank_vars["W_bank"]),
        ("W_bank_count",       bank_vars["W_bank_count"]),
        ("W_bank_bin",         bank_vars["W_bank_binary"]),
        ("W_bank_knn4",        bank_vars["W_bank_knn4"]),
        ("W_bank_nonGeo",      bank_vars["W_bank_nonGeo"]),
        ("W_bank_interstate",  bank_vars["W_bank_interstate"]),
        ("W_bank_intrastate",  bank_vars["W_bank_intrastate"]),
    ]
    for label, W in named:
        rows.append(_row_stats(W, label))
        print(f"  {label:<24}: density={rows[-1]['density_pct']:.4f}%  "
              f"mean_nbrs={rows[-1]['mean_nbrs']:.2f}  "
              f"pct_isolated={rows[-1]['pct_isolated']:.1f}%")

    # KNN variants at selected k values
    W_bank_raw = bank_vars["W_bank"]
    for k in KNN_SELECTED:
        W_knn = _build_knn(W_bank_raw, k)
        label = f"W_bank_knn_{k}"
        rows.append(_row_stats(W_knn, label))
        print(f"  {label:<24}: density={rows[-1]['density_pct']:.4f}%  "
              f"mean_nbrs={rows[-1]['mean_nbrs']:.2f}  "
              f"pct_isolated={rows[-1]['pct_isolated']:.1f}%")

    df = pd.DataFrame(rows)

    W_TBL = 90
    print()
    print("=" * W_TBL)
    print("W Matrix Density Summary")
    print("=" * W_TBL)
    print(f"{'Matrix':<28} {'N':>6}  {'density %':>10} {'mean nbrs':>10} "
          f"{'median nbrs':>12} {'% isolated':>10}")
    print("-" * W_TBL)
    for r in rows:
        print(f"{r['matrix']:<28} {r['N']:>6}  {r['density_pct']:>10.4f} "
              f"{r['mean_nbrs']:>10.2f} {r['median_nbrs']:>12.2f} "
              f"{r['pct_isolated']:>10.1f}")
    print("=" * W_TBL)

    if output_dir is not None:
        cols = ["matrix", "N", "nnz", "density_pct", "mean_nbrs",
                "median_nbrs", "pct_isolated"]
        df[cols].to_csv(output_dir / "w_density_summary.csv", index=False)
        print(f"\nSaved w_density_summary.csv to {output_dir}")

    return df


if __name__ == "__main__":
    run(ROOT / "output")
