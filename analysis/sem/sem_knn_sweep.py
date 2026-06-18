"""
knn_crossover_credit.py
========================
Credit-DV parallel of knn_crossover.py.
Dependent variable: Dl_nloans_b (dln commercial-bank mortgage loans).

Sweeps k = 1..20 to find the crossover k where lambda_knn > lambda_geo for
credit growth. The W_geo reference lambda is estimated in this script using
the same dependent variable, controls, fixed effects, and sample construction.

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
from utils import row_standardize, sparse_to_pysal_w
from panel_data import CREDIT_CONTROLS, load_panel_with_credit
from w_variants import load_w_geo

ROOT        = Path(__file__).parents[2]
COUNTY_PATH = ROOT / "data" / "county_order_Wgeo.csv"
WBANK_PATH  = ROOT / "data" / "W_bank_avg.npz"

K_VALUES = list(range(1, 21))

DV = "Dl_nloans_b"
X_VARS = ["Linter_bra"] + CREDIT_CONTROLS


def build_wbank_knn(W_sparse, k):
    """Keep top-k weights per row, re-standardise. Returns scipy CSR."""
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
            W_knn[i] = row
        else:
            top_k = np.argpartition(row, -k)[-k:]
            W_knn[i, top_k] = row[top_k]
    np.fill_diagonal(W_knn, 0.0)
    rs = W_knn.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        W_knn = np.where(rs > 0, W_knn / rs, 0.0)
    return scipy.sparse.csr_matrix(W_knn)


def usable_counties(panel_sub, county_order, W_all):
    any_nan = panel_sub.groupby("fips5")[[DV] + X_VARS].apply(
        lambda g: g.isna().any().any()
    )
    sub_co = set(panel_sub["fips5"].unique())
    full_rs = np.array(W_all.sum(axis=1)).flatten()
    islands = {county_order[i] for i, r in enumerate(full_rs) if r == 0}
    return [
        c for c in county_order
        if c in sub_co and not any_nan.get(c, True) and c not in islands
    ]


def build_arrays(panel_sub, county_order, W_all, YEARS, year_pos, sample_label,
                 usable=None):
    """Build long-format arrays for Panel_FE_Error with y = Dl_nloans_b."""
    T       = len(YEARS)
    usable = list(usable) if usable is not None else usable_counties(
        panel_sub, county_order, W_all)
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
    x_long = np.hstack([df[X_VARS].values.astype(np.float64), year_dummies])

    assert not np.isnan(y_long).any()
    assert not np.isnan(x_long).any()
    assert y_long.shape == (N * T, 1)
    assert x_long.shape == (N * T, len(X_VARS) + len(YEARS) - 1)

    idx       = np.array([county_order.index(c) for c in usable])
    W_sub = row_standardize(W_all[idx, :][:, idx])
    return y_long, x_long, sparse_to_pysal_w(W_sub), N, tuple(usable)


def run_fe(y, x, w, w_label, ds_label, YEARS):
    nx = X_VARS + [f"yr{yr}" for yr in YEARS[1:]]
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
    W_geo_all, gal_order = load_w_geo(county_order)
    assert gal_order == county_order

    panel    = load_panel_with_credit()
    YEARS    = sorted(panel["year"].unique())
    T        = len(YEARS)
    year_pos = {yr: i for i, yr in enumerate(YEARS)}
    panel_contig    = panel[panel["border"] == 1].copy()
    panel_noncontig = panel[panel["border"] == 0].copy()

    samples = [
        ("Full", panel, "lam_geo", "gap", "n_co", "n_obs"),
        ("Contig", panel_contig, "lam_geo_contig", "gap_contig", "n_co_contig", "n_obs_contig"),
        ("NonContig", panel_noncontig, "lam_geo_noncontig", "gap_noncontig",
         "n_co_noncontig", "n_obs_noncontig"),
    ]

    geo_cache = {}

    def paired_geo_ref(sample_label, panel_sub, usable):
        cache_key = (sample_label, tuple(usable))
        if cache_key not in geo_cache:
            y_geo, x_geo, w_geo, N_geo, _ = build_arrays(
                panel_sub, county_order, W_geo_all, YEARS, year_pos,
                sample_label, usable=usable)
            res_geo = run_fe(y_geo, x_geo, w_geo, "W_geo", sample_label, YEARS)
            geo_cache[cache_key] = dict(
                lam=float(res_geo.lam),
                se_lam=float(res_geo.std_err[-1]),
                n_co=N_geo,
                n_obs=N_geo * T,
            )
        return geo_cache[cache_key]

    sweep_rows = []
    for k in K_VALUES:
        print(f"  k={k} ...", flush=True)
        W_knn_all = build_wbank_knn(W_bank_all, k)

        nz  = W_knn_all.nnz - (W_knn_all.diagonal() != 0).sum()
        tot = N_ALL * (N_ALL - 1)
        density  = nz / tot
        sparsity = 1.0 - density
        avg_nbrs = nz / N_ALL

        knn = {}
        geo = {}
        for sample_label, panel_sub, *_ in samples:
            y, x, w_knn, N, usable = build_arrays(
                panel_sub, county_order, W_knn_all, YEARS, year_pos, sample_label)
            res = run_fe(y, x, w_knn, f"W_bank_k{k}", sample_label, YEARS)
            knn[sample_label] = dict(lam=float(res.lam), n_co=N, n_obs=N * T)
            geo[sample_label] = paired_geo_ref(sample_label, panel_sub, usable)

        geo_f = geo["Full"]
        geo_b = geo["Contig"]
        geo_nb = geo["NonContig"]
        lam_f = knn["Full"]["lam"]
        lam_b = knn["Contig"]["lam"]
        lam_nb = knn["NonContig"]["lam"]

        sweep_rows.append(dict(
            k                      = k,
            n_links                = nz,
            possible_links         = tot,
            density                = density,
            sparsity               = sparsity,
            avg_nbrs               = avg_nbrs,
            lam_geo                = geo_f["lam"],
            se_lam_geo             = geo_f["se_lam"],
            lam_bank_knn           = lam_f,
            gap                    = lam_f - geo_f["lam"],
            n_co                   = knn["Full"]["n_co"],
            n_obs                  = knn["Full"]["n_obs"],
            lam_geo_contig         = geo_b["lam"],
            se_lam_geo_contig      = geo_b["se_lam"],
            lam_bank_knn_contig    = lam_b,
            gap_contig             = lam_b - geo_b["lam"],
            n_co_contig            = knn["Contig"]["n_co"],
            n_obs_contig           = knn["Contig"]["n_obs"],
            lam_geo_noncontig      = geo_nb["lam"],
            se_lam_geo_noncontig   = geo_nb["se_lam"],
            lam_bank_knn_noncontig = lam_nb,
            gap_noncontig          = lam_nb - geo_nb["lam"],
            n_co_noncontig         = knn["NonContig"]["n_co"],
            n_obs_noncontig        = knn["NonContig"]["n_obs"],
        ))

    df_sweep = pd.DataFrame(sweep_rows)

    # ── Print table ───────────────────────────────────────────────────────────
    W = 120
    print()
    print("=" * W)
    print(f"KNN crossover sweep -- DV: {DV}")
    print("Ref lambda: W_geo is re-estimated on each k-specific KNN county set.")
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
                "lam_geo","se_lam_geo","lam_bank_knn","gap","n_co","n_obs",
                "lam_geo_contig","se_lam_geo_contig","lam_bank_knn_contig",
                "gap_contig","n_co_contig","n_obs_contig",
                "lam_geo_noncontig","se_lam_geo_noncontig",
                "lam_bank_knn_noncontig","gap_noncontig",
                "n_co_noncontig","n_obs_noncontig"]
        df_sweep[cols].to_csv(output_dir / "knn_sweep_credit_results.csv", index=False)
        print(f"\nSaved knn_sweep_credit_results.csv to {output_dir}")

    return df_sweep


if __name__ == '__main__':
    run(Path(__file__).parents[2] / 'output')
