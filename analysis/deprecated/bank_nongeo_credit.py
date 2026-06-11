"""
bank_nongeo_credit.py
======================
Credit-DV parallel of bank_nongeo.py.
Dependent variable: Dl_nloans_b (dln commercial-bank mortgage loans).

Tests whether the purely non-geographic bank transmission channel (W_bank_nonGeo)
remains significant with credit growth as the outcome, and whether lambda exceeds
the W_geo baseline.

W matrices used:
  W_geo          -- queen contiguity benchmark
  W_bank_nonGeo  -- bank network with geo-overlapping entries zeroed out
                    (loaded from data/W_bank_nonGeo.npz, built by
                    rq1_four_w_comparison.py if not already present)

Output: output/bank_nongeo_credit_results.csv
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse
import spreg

sys.path.insert(0, str(Path(__file__).parent))
import utils  # noqa
from utils import gal_to_W, row_standardize, sparse_to_pysal_w
from panel_fe_credit import load_panel_with_credit

ROOT        = Path(__file__).parent.parent
GAL_PATH    = ROOT / "data" / "W_geo_queen.gal"
COUNTY_PATH = ROOT / "data" / "county_order_Wgeo.csv"
WBANK_PATH  = ROOT / "data" / "W_bank_avg.npz"
WBANK_NG_PATH = ROOT / "data" / "W_bank_nonGeo.npz"

DV = "Dl_nloans_b"


def build_arrays(panel_sub, county_order, W_all, YEARS, year_pos, sample_label):
    T       = len(YEARS)
    any_nan = panel_sub.groupby("fips5")[DV].apply(lambda s: s.isna().any())
    sub_co  = set(panel_sub["fips5"].unique())

    # any-NaN filter + full-matrix island exclusion
    full_rs = np.array(W_all.sum(axis=1)).flatten()
    islands = {county_order[i] for i, r in enumerate(full_rs) if r == 0}
    usable  = [c for c in county_order
               if c in sub_co and not any_nan.get(c, True) and c not in islands]
    N          = len(usable)
    usable_pos = {c: i for i, c in enumerate(usable)}

    df = (
        panel_sub[panel_sub["fips5"].isin(set(usable))]
        .assign(
            t_idx=lambda d: d["year"].map(year_pos),
            c_idx=lambda d: d["fips5"].map(usable_pos),
        )
        .sort_values(["t_idx", "c_idx"])
    )

    y_long       = df[DV].values.reshape(-1, 1)
    t_idx_vec    = df["t_idx"].values
    year_dummies = np.column_stack([
        (t_idx_vec == year_pos[yr]).astype(np.float64) for yr in YEARS[1:]
    ])
    x_long = np.hstack([df["Linter_bra"].values.reshape(-1, 1), year_dummies])

    assert not np.isnan(y_long).any()
    assert y_long.shape == (N * T, 1)
    assert x_long.shape == (N * T, 11)

    idx     = np.array([county_order.index(c) for c in usable])
    W_sub   = row_standardize(W_all[idx, :][:, idx])
    return y_long, x_long, sparse_to_pysal_w(W_sub), N


def run_fe(y, x, w, w_label, ds_label, YEARS):
    nx = ["Linter_bra"] + [f"yr{yr}" for yr in YEARS[1:]]
    return spreg.Panel_FE_Error(
        y, x, w, name_y=DV, name_x=nx, name_w=w_label, name_ds=ds_label)


def extract(res, N, T, model_name):
    return dict(
        model   = model_name,
        beta    = float(res.betas[0, 0]),
        se_beta = float(res.std_err[0]),
        z_beta  = float(res.z_stat[0][0]),
        p_beta  = float(res.z_stat[0][1]),
        lam     = float(res.lam),
        se_lam  = float(res.std_err[-1]),
        z_lam   = float(res.z_stat[-1][0]),
        p_lam   = float(res.z_stat[-1][1]),
        n_co    = N,
        n_obs   = N * T,
    )


def run(output_dir=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    co_df        = pd.read_csv(COUNTY_PATH, dtype={"fips5": str})
    county_order = co_df["fips5"].str.zfill(5).tolist()

    W_geo_all, gal_order = gal_to_W(GAL_PATH, county_order)
    assert gal_order == county_order

    if not WBANK_NG_PATH.exists():
        raise FileNotFoundError(
            f"{WBANK_NG_PATH} not found — run rq1_four_w_comparison.py first.")
    W_ng_all = row_standardize(scipy.sparse.load_npz(WBANK_NG_PATH))

    panel    = load_panel_with_credit()
    YEARS    = sorted(panel["year"].unique())
    T        = len(YEARS)
    year_pos = {yr: i for i, yr in enumerate(YEARS)}
    panel_nb = panel[panel["border"] == 0].copy()

    rows = []
    for sample_label, panel_sub in [("Full", panel), ("Non-border", panel_nb)]:
        for w_label, W_all in [("W_geo", W_geo_all), ("W_bank_nonGeo", W_ng_all)]:
            print(f"  {sample_label} x {w_label} ...", flush=True)
            y, x, w, N = build_arrays(panel_sub, county_order, W_all,
                                      YEARS, year_pos, sample_label)
            res = run_fe(y, x, w, w_label, sample_label, YEARS)
            rows.append(extract(res, N, T, f"{w_label} ({sample_label.lower()})"))

    # Print table
    def stars(p):
        return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))

    W = 80
    print()
    print("=" * W)
    print(f"W_bank_nonGeo vs W_geo -- DV: {DV} | Panel_FE_Error (two-way FE, ML)")
    print("=" * W)
    print(f"{'Model':<35} {'N':>6}  {'beta':>8} {'SE':>6}  {'lambda':>8} {'SE':>6}")
    print("-" * W)
    for r in rows:
        print(f"{r['model']:<35} {r['n_co']:>6}  "
              f"{r['beta']:>8.4f}{stars(r['p_beta'])} {r['se_beta']:>6.4f}  "
              f"{r['lam']:>8.4f}{stars(r['p_lam'])} {r['se_lam']:>6.4f}")

    # Lambda gap (nonGeo vs geo) per sample
    print()
    print("Lambda gap (W_bank_nonGeo vs W_geo):")
    for sample_label in ["Full", "Non-border"]:
        r_geo = next(r for r in rows
                     if "W_geo" in r["model"] and sample_label.lower() in r["model"])
        r_ng  = next(r for r in rows
                     if "nonGeo" in r["model"] and sample_label.lower() in r["model"])
        import scipy.stats as stats_mod
        gap    = r_ng["lam"] - r_geo["lam"]
        se_gap = np.sqrt(r_geo["se_lam"]**2 + r_ng["se_lam"]**2)
        z_gap  = gap / se_gap
        p_gap  = float(stats_mod.norm.sf(z_gap))
        print(f"  {sample_label:<14}: gap={gap:+.4f}  z={z_gap:.3f}  p={p_gap:.4f}{stars(p_gap)}")

    print()
    print("Significance: *** p<0.01  ** p<0.05  * p<0.10")

    if output_dir is not None:
        cols = ["model","beta","se_beta","z_beta","p_beta",
                "lam","se_lam","z_lam","p_lam","n_co","n_obs"]
        pd.DataFrame(rows)[cols].to_csv(
            output_dir / "bank_nongeo_credit_results.csv", index=False)
        print(f"\nSaved bank_nongeo_credit_results.csv to {output_dir}")

    return rows


if __name__ == '__main__':
    run(Path(__file__).parent.parent / 'output')
