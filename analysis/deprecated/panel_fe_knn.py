"""
Density robustness check: KNN-truncated W_bank vs W_geo

W_bank_avg is ~95% non-sparse (dense), while W_geo is ~0.6% non-sparse.
This density gap could mechanically inflate lambda under W_bank because a
denser W means each county's spatial lag draws on more neighbours, giving
the error process more "room" to be spatially structured.

Fix: for each county row in W_bank, retain only the k=5 largest weights
and zero the rest, then re-row-standardise.  This makes W_bank_knn as
sparse as W_geo (~99.5% sparsity) while keeping the strongest bank links.

If the high lambda under W_bank_avg survives truncation to W_bank_knn,
the result is attributable to network structure, not to matrix density.

Models: Panel_FE_Error ML, two-way FE (county FE + year dummies 1996-2005)
  W_geo        -- full sample and non-border
  W_bank_knn   -- full sample and non-border

Comparison table: W_geo lambda | W_bank_knn lambda | W_bank_avg lambda (reference)
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
import scipy.sparse
import spreg

import utils  # noqa: applies spreg Panel_FE_Error compatibility patch
from utils import gal_to_W, row_standardize, sparse_to_pysal_w


ROOT        = Path(__file__).parent.parent
PANEL_PATH  = ROOT / "data" / "estimation_panel.csv"
GAL_PATH    = ROOT / "data" / "W_geo_queen.gal"
COUNTY_PATH = ROOT / "data" / "county_order_Wgeo.csv"
WBANK_PATH  = ROOT / "data" / "W_bank_avg.npz"

K = 5   # neighbours to retain per row -- matches avg degree of W_geo

# ── Previously estimated W_bank_avg results (from panel_fe_error.py) ─────────
WBANK_AVG_RESULTS = {
    "full"      : dict(beta=2.4048, lam=0.9879),   # from panel_fe_error.py
    "non-border": dict(beta=2.4840, lam=0.9873),
}


# ── Build KNN-truncated W_bank ────────────────────────────────────────────────
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
        row = W[i]
        nz  = np.count_nonzero(row)
        if nz == 0:
            continue
        if nz <= k:
            W_knn[i] = row          # fewer neighbours than k -- keep all
        else:
            top_k = np.argpartition(row, -k)[-k:]
            W_knn[i, top_k] = row[top_k]

    np.fill_diagonal(W_knn, 0.0)

    # Row-standardise
    rs = W_knn.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        W_knn = np.where(rs > 0, W_knn / rs, 0.0)

    return scipy.sparse.csr_matrix(W_knn)


def sparsity(W_sparse):
    N   = W_sparse.shape[0]
    nz  = W_sparse.nnz - (W_sparse.diagonal() != 0).sum()  # off-diagonal non-zeros
    tot = N * (N - 1)
    return 1.0 - nz / tot, nz / N   # (sparsity, avg neighbours)


def build_arrays(panel_sub, county_order, W_geo_all, W_knn_all, YEARS, year_pos,
                 N_ALL, sample_label):
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

    # Subset and re-standardise W matrices for this sample
    idx       = np.array([county_order.index(c) for c in usable])
    W_geo_sub = row_standardize(W_geo_all[idx, :][:, idx])
    W_knn_sub = row_standardize(W_knn_all[idx, :][:, idx])

    return y_long, x_long, sparse_to_pysal_w(W_geo_sub), sparse_to_pysal_w(W_knn_sub), N


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
        p_b   = float(res.z_stat[0][1]),
        lam   = float(res.lam),
        se_l  = float(res.std_err[-1]),
        p_l   = float(res.z_stat[-1][1]),
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
    W_bank_all = row_standardize(W_bank_raw)   # full-density version

    panel = pd.read_csv(PANEL_PATH)
    panel["fips5"] = panel["fips5"].astype(str).str.zfill(5)
    YEARS    = sorted(panel["year"].unique())
    T        = len(YEARS)
    year_pos = {yr: i for i, yr in enumerate(YEARS)}

    # ── Sparsity diagnostics ──────────────────────────────────────────────────
    sp_geo,  avg_geo  = sparsity(W_geo_all)
    sp_bank, avg_bank = sparsity(W_bank_all)

    W_knn_all = build_wbank_knn(W_bank_all, K)
    sp_knn, avg_knn = sparsity(W_knn_all)

    # ── Estimation ────────────────────────────────────────────────────────────
    results = {}

    y_f, x_f, w_geo_f, w_knn_f, N_f = build_arrays(
        panel, county_order, W_geo_all, W_knn_all, YEARS, year_pos, N_ALL,
        "Full sample")
    results["W_geo (full)"]      = extract(
        run_fe(y_f, x_f, w_geo_f, "W_geo",     "Full", YEARS), N_f, T)
    results["W_bank_knn (full)"] = extract(
        run_fe(y_f, x_f, w_knn_f, "W_bank_knn","Full", YEARS), N_f, T)

    panel_nb = panel[panel["border"] == 0].copy()
    y_nb, x_nb, w_geo_nb, w_knn_nb, N_nb = build_arrays(
        panel_nb, county_order, W_geo_all, W_knn_all, YEARS, year_pos, N_ALL,
        "Non-border")
    results["W_geo (non-border)"]      = extract(
        run_fe(y_nb, x_nb, w_geo_nb,  "W_geo",     "NB", YEARS), N_nb, T)
    results["W_bank_knn (non-border)"] = extract(
        run_fe(y_nb, x_nb, w_knn_nb, "W_bank_knn","NB", YEARS), N_nb, T)

    # ── Save outputs ──────────────────────────────────────────────────────────
    if output_dir is not None:
        model_rows = []
        for model_name, r in results.items():
            model_rows.append(dict(
                model=model_name,
                beta=r["beta"], se_b=r["se_b"], p_b=r["p_b"],
                lam=r["lam"],   se_l=r["se_l"], p_l=r["p_l"],
                n_co=r["n_co"], n_obs=r["n_obs"],
            ))
        pd.DataFrame(model_rows).to_csv(
            output_dir / "knn_density_model_results.csv", index=False)

        comparison_rows = []
        for sample, geo_key, knn_key, avg_key in [
            ("full",       "W_geo (full)",       "W_bank_knn (full)",       "full"),
            ("non-border", "W_geo (non-border)", "W_bank_knn (non-border)", "non-border"),
        ]:
            lg      = results[geo_key]["lam"]
            lk      = results[knn_key]["lam"]
            la      = WBANK_AVG_RESULTS[avg_key]["lam"]
            gap_knn = lk - lg
            gap_avg = la - lg

            if gap_knn < -0.02:
                verdict = "REVERSED"
            elif abs(gap_knn) < 0.02:
                verdict = "gap GONE"
            elif gap_knn < gap_avg * 0.5:
                verdict = "gap HALVED"
            elif gap_knn < gap_avg * 0.8:
                verdict = "gap SHRINKS"
            else:
                verdict = "gap HOLDS"

            comparison_rows.append(dict(
                sample=sample,
                lam_geo=lg,   lam_knn=lk,   gap_knn=gap_knn,
                lam_avg=la,   gap_avg=gap_avg, verdict=verdict,
            ))
        pd.DataFrame(comparison_rows).to_csv(
            output_dir / "knn_density_comparison.csv", index=False)

    return results


if __name__ == '__main__':
    run(Path(__file__).parent.parent / 'output')
