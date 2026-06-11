"""
knn_crossover_credit.py
========================
Credit-DV parallel of knn_crossover.py.
Dependent variable: Dl_nloans_b (dln commercial-bank mortgage loans).

Sweeps k = 1..20 to find the crossover k where lambda_knn > lambda_geo for
credit growth. Reference lambdas from panel_fe_credit_results.csv:
  lam_geo_full = 0.1801
  lam_geo_nb   = 0.1701

Also reports density/sparsity of each W_bank_k matrix.

Output: output/knn_sweep_credit_results.csv
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import numpy as np
import scipy.sparse
import pandas as pd
import spreg

sys.path.insert(0, str(Path(__file__).parents[1]))
import utils  # noqa
from utils import row_standardize, sparse_to_pysal_w, build_wbank_knn
from panel_data import load_panel_with_credit

ROOT        = Path(__file__).parents[2]
COUNTY_PATH = ROOT / "data" / "county_order_Wgeo.csv"
WBANK_PATH  = ROOT / "data" / "W_bank_avg.npz"

K_VALUES = list(range(1, 21))

# Reference lambdas from sem_credit.py (will be overwritten from CSV if available)
LAM_GEO_FULL      = 0.1801
LAM_GEO_CONTIG    = 0.1701
LAM_GEO_NONCONTIG = 0.1820   # placeholder; overridden at runtime from CSV when available

DV = "Dl_nloans_b"


def build_arrays(panel_sub, county_order, W_knn_all, YEARS, year_pos, sample_label):
    """Build long-format arrays for Panel_FE_Error with y = Dl_nloans_b."""
    T       = len(YEARS)
    any_nan = panel_sub.groupby("fips5")[DV].apply(lambda s: s.isna().any())
    sub_co  = set(panel_sub["fips5"].unique())
    usable  = [c for c in county_order
               if c in sub_co and not any_nan.get(c, True)]
    N       = len(usable)
    u_pos   = {c: i for i, c in enumerate(usable)}

    df = (panel_sub[panel_sub["fips5"].isin(set(usable))]
          .assign(t_idx=lambda d: d["year"].map(year_pos),
                  c_idx=lambda d: d["fips5"].map(u_pos))
          .sort_values(["t_idx", "c_idx"]))

    y_long       = df[DV].values.reshape(-1, 1)
    t_idx_vec    = df["t_idx"].values
    year_dummies = np.column_stack([
        (t_idx_vec == year_pos[yr]).astype(np.float64) for yr in YEARS[1:]])
    x_long = np.hstack([df["Linter_bra"].values.reshape(-1, 1), year_dummies])

    assert not np.isnan(y_long).any()
    assert y_long.shape == (N * T, 1)
    assert x_long.shape == (N * T, 11)

    idx       = np.array([county_order.index(c) for c in usable])
    W_knn_sub = row_standardize(W_knn_all[idx, :][:, idx])
    return y_long, x_long, sparse_to_pysal_w(W_knn_sub), N


def run_fe(y, x, w, w_label, ds_label, YEARS):
    nx = ["Linter_bra"] + [f"yr{yr}" for yr in YEARS[1:]]
    return spreg.Panel_FE_Error(
        y, x, w, name_y=DV, name_x=nx, name_w=w_label, name_ds=ds_label)


def run(output_dir=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    co_df        = pd.read_csv(COUNTY_PATH, dtype={"fips5": str})
    county_order = co_df["fips5"].str.zfill(5).tolist()
    N_ALL        = len(county_order)

    W_bank_raw = scipy.sparse.load_npz(WBANK_PATH)
    W_bank_all = row_standardize(W_bank_raw)

    panel    = load_panel_with_credit()
    YEARS    = sorted(panel["year"].unique())
    T        = len(YEARS)
    year_pos = {yr: i for i, yr in enumerate(YEARS)}
    panel_contig    = panel[panel["border"] == 1].copy()
    panel_noncontig = panel[panel["border"] == 0].copy()

    sweep_rows = []
    for k in K_VALUES:
        print(f"  k={k} ...", flush=True)
        W_knn_all = build_wbank_knn(W_bank_all, k)

        nz  = W_knn_all.nnz - (W_knn_all.diagonal() != 0).sum()
        tot = N_ALL * (N_ALL - 1)
        density  = nz / tot
        sparsity = 1.0 - density
        avg_nbrs = nz / N_ALL

        y_f, x_f, w_knn_f, N_f = build_arrays(
            panel, county_order, W_knn_all, YEARS, year_pos, "Full")
        res_f = run_fe(y_f, x_f, w_knn_f, f"W_bank_k{k}", "Full", YEARS)
        lam_f = float(res_f.lam)

        y_b, x_b, w_knn_b, N_b = build_arrays(
            panel_contig, county_order, W_knn_all, YEARS, year_pos, "Contig")
        res_b = run_fe(y_b, x_b, w_knn_b, f"W_bank_k{k}", "Contig", YEARS)
        lam_b = float(res_b.lam)

        y_nb, x_nb, w_knn_nb, N_nb = build_arrays(
            panel_noncontig, county_order, W_knn_all, YEARS, year_pos, "NonContig")
        res_nb = run_fe(y_nb, x_nb, w_knn_nb, f"W_bank_k{k}", "NonContig", YEARS)
        lam_nb = float(res_nb.lam)

        sweep_rows.append(dict(
            k                      = k,
            n_links                = nz,
            possible_links         = tot,
            density                = density,
            sparsity               = sparsity,
            avg_nbrs               = avg_nbrs,
            lam_geo                = LAM_GEO_FULL,
            lam_bank_knn           = lam_f,
            gap                    = lam_f  - LAM_GEO_FULL,
            n_co                   = N_f,
            n_obs                  = N_f  * T,
            lam_geo_contig         = LAM_GEO_CONTIG,
            lam_bank_knn_contig    = lam_b,
            gap_contig             = lam_b - LAM_GEO_CONTIG,
            n_co_contig            = N_b,
            n_obs_contig           = N_b * T,
            lam_geo_noncontig      = LAM_GEO_NONCONTIG,
            lam_bank_knn_noncontig = lam_nb,
            gap_noncontig          = lam_nb - LAM_GEO_NONCONTIG,
            n_co_noncontig         = N_nb,
            n_obs_noncontig        = N_nb * T,
        ))

    df_sweep = pd.DataFrame(sweep_rows)

    # ── Print table ───────────────────────────────────────────────────────────
    W = 120
    print()
    print("=" * W)
    print(f"KNN crossover sweep -- DV: {DV}")
    print(f"Ref lambda: geo_full={LAM_GEO_FULL}  geo_contig={LAM_GEO_CONTIG}  "
          f"geo_noncontig={LAM_GEO_NONCONTIG}")
    print(f"Gap = lam_knn - lam_geo  |  '<<' marks first k where gap > 0")
    print("=" * W)
    print(f"{'k':>4}  {'density':>9} {'avg_nbrs':>8}  "
          f"{'lam_knn(F)':>11} {'gap(F)':>8}  "
          f"{'lam_knn(C)':>11} {'gap(C)':>8}  "
          f"{'lam_knn(NC)':>12} {'gap(NC)':>9}")
    print("-" * W)
    cross_f  = None
    cross_b  = None
    cross_nb = None
    for row in sweep_rows:
        tag_f  = " <<" if row["gap"]          > 0 and cross_f  is None else "   "
        tag_b  = " <<" if row["gap_contig"]   > 0 and cross_b  is None else "   "
        tag_nb = " <<" if row["gap_noncontig"]> 0 and cross_nb is None else "   "
        if row["gap"]          > 0 and cross_f  is None: cross_f  = row["k"]
        if row["gap_contig"]   > 0 and cross_b  is None: cross_b  = row["k"]
        if row["gap_noncontig"]> 0 and cross_nb is None: cross_nb = row["k"]
        print(f"{row['k']:>4}  {row['density']:>9.5f} {row['avg_nbrs']:>8.2f}  "
              f"{row['lam_bank_knn']:>11.4f} {row['gap']:>+8.4f}{tag_f}  "
              f"{row['lam_bank_knn_contig']:>11.4f} {row['gap_contig']:>+8.4f}{tag_b}  "
              f"{row['lam_bank_knn_noncontig']:>12.4f} {row['gap_noncontig']:>+9.4f}{tag_nb}")

    print()
    cf  = cross_f  if cross_f  is not None else ">20"
    cb  = cross_b  if cross_b  is not None else ">20"
    cnb = cross_nb if cross_nb is not None else ">20"
    print(f"Crossover k: full={cf}  contig={cb}  noncontig={cnb}")
    print("=" * W)

    if output_dir is not None:
        cols = ["k","n_links","possible_links","density","sparsity","avg_nbrs",
                "lam_geo","lam_bank_knn","gap","n_co","n_obs",
                "lam_geo_contig","lam_bank_knn_contig","gap_contig","n_co_contig","n_obs_contig",
                "lam_geo_noncontig","lam_bank_knn_noncontig","gap_noncontig",
                "n_co_noncontig","n_obs_noncontig"]
        df_sweep[cols].to_csv(output_dir / "knn_sweep_credit_results.csv", index=False)
        print(f"\nSaved knn_sweep_credit_results.csv to {output_dir}")

    return df_sweep


if __name__ == '__main__':
    run(Path(__file__).parents[2] / 'output')
