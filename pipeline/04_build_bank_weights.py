"""
Build bank-network spatial weights matrices from FDIC branch data.

Two variants, both time-averaged over 1994-2005 and row-standardised:

  W_bank (binary)
    M[c,h] = 1 if holding company h has any branch in county c
    w_cc'  = (M @ M.T)[c,c'] / sqrt(diag[c] * diag[c'])   (cosine similarity)

  W_bank_count (count-weighted)
    M[c,h] = number of branches of HC h in county c
    w_cc'  = (M @ M.T)[c,c'] / sqrt(total_branches[c] * total_branches[c'])

Reads  : data/fdic_deposits_1994_2005.csv
         data/county_order_Wgeo.csv       (defines the N x N matrix dimensions)
Writes : data/W_bank_avg.npz             binary variant
         data/W_bank_count_avg.npz        count variant
         data/county_order_Wbank.csv      row order (matches W_geo order)
"""
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse

ROOT       = Path(__file__).parent.parent
FDIC_PATH  = ROOT / "data" / "fdic_deposits_1994_2005.csv"
ORDER_PATH = ROOT / "data" / "county_order_Wgeo.csv"
OUT_BIN    = ROOT / "data" / "W_bank_avg.npz"
OUT_CNT    = ROOT / "data" / "W_bank_count_avg.npz"
OUT_ORDER  = ROOT / "data" / "county_order_Wbank.csv"


def build_binary_year(df_year, N, county_idx):
    """Binary cosine-similarity W from HC presence indicators."""
    pairs  = df_year[["fips5", "RSSDHCR"]].drop_duplicates().reset_index(drop=True)
    hcs    = pairs["RSSDHCR"].unique()
    hc_map = {hc: i for i, hc in enumerate(hcs)}
    M = csr_matrix(
        (np.ones(len(pairs), dtype=np.float32),
         (pairs["fips5"].map(county_idx).values,
          pairs["RSSDHCR"].map(hc_map).values)),
        shape=(N, len(hcs))
    )
    B     = (M @ M.T).toarray()
    d     = B.diagonal().copy()
    denom = np.sqrt(np.outer(d, d))
    with np.errstate(divide="ignore", invalid="ignore"):
        W = np.where(denom > 0, B / denom, 0.0)
    np.fill_diagonal(W, 0.0)
    rs = W.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        W = np.where(rs > 0, W / rs, 0.0)
    return W, len(hcs)


def build_count_year(df_year, N, county_idx):
    """Count-weighted cosine-style W from HC branch counts."""
    counts = (df_year.groupby(["fips5", "RSSDHCR"])
              .size().reset_index(name="n_branches"))
    hcs    = counts["RSSDHCR"].unique()
    hc_map = {hc: i for i, hc in enumerate(hcs)}
    M = csr_matrix(
        (counts["n_branches"].values.astype(np.float32),
         (counts["fips5"].map(county_idx).values,
          counts["RSSDHCR"].map(hc_map).values)),
        shape=(N, len(hcs))
    )
    B             = (M @ M.T).toarray()
    branch_totals = np.array(M.sum(axis=1)).flatten()
    denom         = np.sqrt(np.outer(branch_totals, branch_totals))
    with np.errstate(divide="ignore", invalid="ignore"):
        W = np.where(denom > 0, B / denom, 0.0)
    np.fill_diagonal(W, 0.0)
    rs = W.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        W = np.where(rs > 0, W / rs, 0.0)
    return W, len(hcs)


def finalise_avg(W_sum, n_years):
    """Average over years, zero diagonal, re-standardise rows."""
    W = W_sum / n_years
    np.fill_diagonal(W, 0.0)
    rs = W.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        W = np.where(rs > 0, W / rs, 0.0)
    return W


if __name__ == "__main__":
    co           = pd.read_csv(ORDER_PATH, dtype={"fips5": str})
    county_order = co["fips5"].str.zfill(5).tolist()
    N            = len(county_order)
    county_idx   = {fips: i for i, fips in enumerate(county_order)}
    county_set   = set(county_order)

    print("Loading FDIC data ...")
    fdic          = pd.read_csv(FDIC_PATH, dtype={"STCNTYBR": str})
    fdic["fips5"] = fdic["STCNTYBR"].str.zfill(5)
    fdic          = fdic[fdic["fips5"].isin(county_set)].copy()
    fdic          = fdic[fdic["RSSDHCR"].notna() & (fdic["RSSDHCR"] != 0)].copy()
    print(f"Filtered: {len(fdic):,} branch-year rows  |  {N} panel counties")

    YEARS     = sorted(fdic["year"].unique())
    W_bin_sum = np.zeros((N, N), dtype=np.float64)
    W_cnt_sum = np.zeros((N, N), dtype=np.float64)

    print(f"\n{'Year':>6}  {'Binary nz':>12}  {'Count nz':>12}  {'HCs':>8}")
    print("-" * 48)
    for yr in YEARS:
        df_yr         = fdic[fdic["year"] == yr]
        W_bin, n_hcs  = build_binary_year(df_yr, N, county_idx)
        W_cnt, _      = build_count_year(df_yr, N, county_idx)
        W_bin_sum    += W_bin
        W_cnt_sum    += W_cnt
        print(f"  {yr}:  binary nz={(W_bin > 0).sum():>9,}  "
              f"count nz={(W_cnt > 0).sum():>9,}  HCs={n_hcs:,}")

    W_bin_avg = finalise_avg(W_bin_sum, len(YEARS))
    W_cnt_avg = finalise_avg(W_cnt_sum, len(YEARS))

    nz_bin = (W_bin_avg > 0).sum()
    nz_cnt = (W_cnt_avg > 0).sum()
    print(f"\nTime-averaged binary : nz={nz_bin:,}  "
          f"avg w={W_bin_avg[W_bin_avg > 0].mean():.4f}  "
          f"sparsity={1 - nz_bin / (N * (N - 1)):.4f}")
    print(f"Time-averaged count  : nz={nz_cnt:,}  "
          f"avg w={W_cnt_avg[W_cnt_avg > 0].mean():.4f}  "
          f"sparsity={1 - nz_cnt / (N * (N - 1)):.4f}")

    scipy.sparse.save_npz(OUT_BIN, scipy.sparse.csr_matrix(W_bin_avg))
    scipy.sparse.save_npz(OUT_CNT, scipy.sparse.csr_matrix(W_cnt_avg))
    pd.DataFrame({"row_index": range(N), "fips5": county_order}).to_csv(OUT_ORDER, index=False)

    assert county_order == co["fips5"].str.zfill(5).tolist()
    print(f"\nSaved {OUT_BIN.name}")
    print(f"Saved {OUT_CNT.name}")
    print(f"Saved {OUT_ORDER.name}")
