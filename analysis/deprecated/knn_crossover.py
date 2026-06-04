"""
KNN crossover analysis: k sweep (k=1..20) to find crossover point where
lambda_knn > lambda_geo, for full sample and non-border sample.

Part A: k sweep
  For each k in 1, 2, ..., 20:
    Build W_bank_knn by keeping top-k weights per row (zero rest, row-standardise).
    Run Panel_FE_Error (county FE + year dummies 1996-2005).
    Record lambda, density, sparsity, and average neighbors for full and non-border.

Part B: crossover identification
  Crossover k = first k where lambda_knn > lambda_geo for each sample.
  Reference lambdas: LAM_GEO_FULL=0.8268, LAM_GEO_NB=0.8316.
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import scipy.sparse
import pandas as pd
import spreg

import utils  # noqa: applies spreg Panel_FE_Error compatibility patch
from utils import gal_to_W, row_standardize, sparse_to_pysal_w

ROOT        = Path(__file__).parent.parent
PANEL_PATH  = ROOT / "data" / "estimation_panel.csv"
GAL_PATH    = ROOT / "data" / "W_geo_queen.gal"
COUNTY_PATH = ROOT / "data" / "county_order_Wgeo.csv"
WBANK_PATH  = ROOT / "data" / "W_bank_avg.npz"

K_VALUES = list(range(1, 21))

# Reference lambdas from Panel_FE_Error with W_geo (from panel_fe_error.py)
LAM_GEO_FULL = 0.8268
LAM_GEO_NB   = 0.8316


def build_wbank_knn(W_sparse, k):
    """
    For each row keep only the k largest weights; zero the rest.
    Diagonal is zeroed; rows are re-standardised to sum to 1.
    Rows with fewer than k non-zero entries are kept as-is.
    Returns a scipy sparse CSR matrix.
    """
    W = W_sparse.toarray().astype(np.float64)
    np.fill_diagonal(W, 0.0)
    N = W.shape[0]
    W_knn = np.zeros_like(W)
    for i in range(N):
        row = W[i]; nz = np.count_nonzero(row)
        if nz == 0: continue
        if nz <= k: W_knn[i] = row
        else:
            top_k = np.argpartition(row, -k)[-k:]
            W_knn[i, top_k] = row[top_k]
    np.fill_diagonal(W_knn, 0.0)
    rs = W_knn.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        W_knn = np.where(rs > 0, W_knn / rs, 0.0)
    return scipy.sparse.csr_matrix(W_knn)


def build_arrays(panel_sub, county_order, W_geo_all, W_knn_all,
                 YEARS, year_pos, N_ALL, sample_label):
    """Long-format (N*T, 1) y and x for Panel_FE_Error; time-major sort."""
    T = len(YEARS)
    na_all = panel_sub.groupby("fips5")["Linter_ela"].apply(lambda s: s.isna().all())
    sub_co = set(panel_sub["fips5"].unique())
    usable = [c for c in county_order if c in sub_co and not na_all.get(c, True)]
    N      = len(usable)
    u_pos  = {c: i for i, c in enumerate(usable)}

    df = (panel_sub[panel_sub["fips5"].isin(set(usable))]
          .assign(t_idx=lambda d: d["year"].map(year_pos),
                  c_idx=lambda d: d["fips5"].map(u_pos))
          .sort_values(["t_idx", "c_idx"]))

    y_long = df["Linter_ela"].values.reshape(-1, 1)
    dummy_years  = YEARS[1:]
    t_idx_vec    = df["t_idx"].values
    year_dummies = np.column_stack([
        (t_idx_vec == year_pos[yr]).astype(np.float64)
        for yr in dummy_years
    ])
    x_long = np.hstack([df["Linter_bra"].values.reshape(-1, 1), year_dummies])

    assert y_long.shape == (N * T, 1),  f"y wrong shape: {y_long.shape}"
    assert x_long.shape == (N * T, 11), f"x wrong shape: {x_long.shape}"

    idx       = np.array([county_order.index(c) for c in usable])
    W_knn_sub = row_standardize(W_knn_all[idx, :][:, idx])

    return y_long, x_long, sparse_to_pysal_w(W_knn_sub), N


def run_fe(y, x, w, w_label, ds_label, YEARS):
    nx = ["Linter_bra"] + [f"yr{yr}" for yr in YEARS[1:]]
    return spreg.Panel_FE_Error(
        y, x, w,
        name_y="Linter_ela", name_x=nx,
        name_w=w_label, name_ds=ds_label
    )


def extract(res, N, T):
    return dict(
        beta  = float(res.betas[0, 0]),
        se_b  = float(res.std_err[0]),
        lam   = float(res.lam),
        se_l  = float(res.std_err[-1]),
        n_co  = N,
        n_obs = N * T,
    )


def run(output_dir=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load inputs ───────────────────────────────────────────────────────────
    co_df        = pd.read_csv(COUNTY_PATH, dtype={"fips5": str})
    county_order = co_df["fips5"].str.zfill(5).tolist()
    N_ALL        = len(county_order)

    W_geo_all, gal_order = gal_to_W(GAL_PATH, county_order)
    assert gal_order == county_order, "GAL order != county_order_Wgeo.csv"

    W_bank_raw = scipy.sparse.load_npz(WBANK_PATH)
    W_bank_all = row_standardize(W_bank_raw)

    panel = pd.read_csv(PANEL_PATH)
    panel["fips5"] = panel["fips5"].astype(str).str.zfill(5)
    YEARS    = sorted(panel["year"].unique())
    T        = len(YEARS)
    year_pos = {yr: i for i, yr in enumerate(YEARS)}

    panel_nb = panel[panel["border"] == 0].copy()

    # ── Part A: k sweep ───────────────────────────────────────────────────────
    sweep_rows = []
    for k in K_VALUES:
        W_knn_all = build_wbank_knn(W_bank_all, k)

        # Matrix density/sparsity for the full cross-sectional W.
        nz  = W_knn_all.nnz - (W_knn_all.diagonal() != 0).sum()
        tot = N_ALL * (N_ALL - 1)
        density   = nz / tot
        sparsity  = 1.0 - density
        avg_nbrs  = nz / N_ALL

        # Full sample
        y_f, x_f, w_knn_f, N_f = build_arrays(
            panel, county_order, W_geo_all, W_knn_all,
            YEARS, year_pos, N_ALL, "Full")
        res_f = extract(run_fe(y_f, x_f, w_knn_f, f"W_bank_k{k}", "Full", YEARS),
                        N_f, T)

        # Non-border sample
        y_nb, x_nb, w_knn_nb, N_nb = build_arrays(
            panel_nb, county_order, W_geo_all, W_knn_all,
            YEARS, year_pos, N_ALL, "NB")
        res_nb = extract(run_fe(y_nb, x_nb, w_knn_nb, f"W_bank_k{k}", "NB", YEARS),
                         N_nb, T)

        gap_full = res_f["lam"]  - LAM_GEO_FULL
        gap_nb   = res_nb["lam"] - LAM_GEO_NB

        sweep_rows.append(dict(
            k=k,
            n_links=nz,
            possible_links=tot,
            density=density,
            sparsity=sparsity,
            avg_nbrs=avg_nbrs,
            lam_full=res_f["lam"],
            gap_full=gap_full,
            lam_nb=res_nb["lam"],
            gap_nb=gap_nb,
        ))

    df_sweep = pd.DataFrame(sweep_rows)

    # ── Part B: crossover identification ──────────────────────────────────────
    crossover_rows = []
    for sample, lam_col, gap_col, lam_geo_ref in [
        ("full",       "lam_full", "gap_full", LAM_GEO_FULL),
        ("non-border", "lam_nb",   "gap_nb",   LAM_GEO_NB),
    ]:
        crossover_k   = None
        crossover_den = None
        crossover_avg = None
        crossover_gap = None

        for _, row in df_sweep.iterrows():
            if row[gap_col] > 0:
                crossover_k   = int(row["k"])
                crossover_den = row["density"]
                crossover_avg = row["avg_nbrs"]
                crossover_gap = row[gap_col]
                break

        crossover_rows.append(dict(
            sample=sample,
            crossover_k=crossover_k if crossover_k is not None else -1,
            density=crossover_den,
            avg_nbrs=crossover_avg,
            gap=crossover_gap,
        ))

    df_crossover = pd.DataFrame(crossover_rows)

    # ── Save outputs ──────────────────────────────────────────────────────────
    if output_dir is not None:
        df_sweep.to_csv(output_dir / "knn_sweep_results.csv", index=False)
        df_crossover.to_csv(output_dir / "knn_crossover_summary.csv", index=False)

    return dict(df_sweep=df_sweep, df_crossover=df_crossover)


if __name__ == '__main__':
    run(ROOT / 'output')
