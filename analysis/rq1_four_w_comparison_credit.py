"""
rq1_four_w_comparison_credit.py
================================
Credit-DV parallel of rq1_four_w_comparison.py.
Dependent variable: Dl_nloans_b (dln commercial-bank mortgage loans).

W matrices: W_geo | W_bank_bin | W_bank_count | W_bank_nonGeo
Samples:    Full  | Non-border

County filter: any-NaN on Dl_nloans_b (stricter than all-NaN on Linter_ela,
  required because Dl_nloans_b has partial-NaN counties that Panel_FE_Error
  cannot handle).

Output: output/four_w_comparison_credit.csv
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.stats as stats
import spreg

sys.path.insert(0, str(Path(__file__).parent))
import utils  # noqa
from utils import gal_to_W, row_standardize, sparse_to_pysal_w
from panel_fe_credit import load_panel_with_credit

ROOT           = Path(__file__).parent.parent
GAL_PATH       = ROOT / "data" / "W_geo_queen.gal"
COUNTY_PATH    = ROOT / "data" / "county_order_Wgeo.csv"
WBANK_BIN_PATH = ROOT / "data" / "W_bank_avg.npz"
WBANK_CNT_PATH = ROOT / "data" / "W_bank_count_avg.npz"
WBANK_NG_PATH  = ROOT / "data" / "W_bank_nonGeo.npz"

DV = "Dl_nloans_b"


def run_one(panel_sub, county_order, W_all, w_label, sample_label, YEARS, year_pos):
    T      = len(YEARS)
    # any-NaN filter for DV (not all-NaN — panel needs to be balanced)
    any_nan = panel_sub.groupby("fips5")[DV].apply(lambda s: s.isna().any())
    sub_co  = set(panel_sub["fips5"].unique())

    usable = [c for c in county_order if c in sub_co and not any_nan.get(c, True)]

    # Drop structural W-islands
    full_rs = np.array(W_all.sum(axis=1)).flatten()
    islands = {county_order[i] for i, r in enumerate(full_rs) if r == 0}
    usable  = [c for c in usable if c not in islands]
    N       = len(usable)

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

    assert not np.isnan(y_long).any(), f"NaN in y ({sample_label}, {w_label})"
    assert y_long.shape == (N * T, 1)
    assert x_long.shape == (N * T, 11)

    idx     = np.array([county_order.index(c) for c in usable])
    W_sub   = row_standardize(W_all[idx, :][:, idx])
    w_pysal = sparse_to_pysal_w(W_sub)

    nx  = ["Linter_bra"] + [f"yr{yr}" for yr in YEARS[1:]]
    res = spreg.Panel_FE_Error(
        y_long, x_long, w_pysal,
        name_y=DV, name_x=nx, name_w=w_label, name_ds=sample_label,
    )

    return dict(
        sample     = sample_label,
        w_matrix   = w_label,
        n_counties = N,
        n_obs      = N * T,
        beta       = float(res.betas[0, 0]),
        se_beta    = float(res.std_err[0]),
        z_beta     = float(res.z_stat[0][0]),
        p_beta     = float(res.z_stat[0][1]),
        lam        = float(res.lam),
        se_lam     = float(res.std_err[-1]),
        z_lam      = float(res.z_stat[-1][0]),
        p_lam      = float(res.z_stat[-1][1]),
        logll      = float(res.logll),
        aic        = float(res.aic),
    )


def run(output_dir=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    co_df        = pd.read_csv(COUNTY_PATH, dtype={"fips5": str})
    county_order = co_df["fips5"].str.zfill(5).tolist()

    W_geo_all, gal_order = gal_to_W(GAL_PATH, county_order)
    assert gal_order == county_order
    W_bin_all = row_standardize(scipy.sparse.load_npz(WBANK_BIN_PATH))
    W_cnt_all = row_standardize(scipy.sparse.load_npz(WBANK_CNT_PATH))

    if not WBANK_NG_PATH.exists():
        raise FileNotFoundError(f"{WBANK_NG_PATH} not found — run rq1_four_w_comparison.py first.")
    W_ng_all = row_standardize(scipy.sparse.load_npz(WBANK_NG_PATH))

    W_MATRICES = [
        ("W_geo",         W_geo_all),
        ("W_bank_bin",    W_bin_all),
        ("W_bank_count",  W_cnt_all),
        ("W_bank_nonGeo", W_ng_all),
    ]

    panel    = load_panel_with_credit()
    YEARS    = sorted(panel["year"].unique())
    year_pos = {yr: i for i, yr in enumerate(YEARS)}
    panel_nb = panel[panel["border"] == 0].copy()

    results = {}
    for sample_label, panel_sub in [("Full", panel), ("Non-border", panel_nb)]:
        for w_label, W_all in W_MATRICES:
            print(f"  Estimating: {sample_label} x {w_label} ...", flush=True)
            results[(sample_label, w_label)] = run_one(
                panel_sub, county_order, W_all, w_label, sample_label, YEARS, year_pos)

    BANK_WS = ["W_bank_bin", "W_bank_count", "W_bank_nonGeo"]

    def gap_stat(r_bank, r_geo):
        gap    = r_bank["lam"]  - r_geo["lam"]
        se_gap = np.sqrt(r_bank["se_lam"]**2 + r_geo["se_lam"]**2)
        z_gap  = gap / se_gap
        p_gap  = float(stats.norm.sf(z_gap))
        lr     = 2.0 * (r_bank["logll"] - r_geo["logll"])
        p_lr   = float(stats.chi2.sf(lr, df=1))
        return gap, se_gap, z_gap, p_gap, lr, p_lr

    def stars(p):
        if np.isnan(p): return "   "
        return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else "   "))

    W = 90
    print()
    print("=" * W)
    print("TABLE A -- Panel_FE_Error: beta and lambda | DV: Dl_nloans_b (two-way FE, ML)")
    print("=" * W)
    print(f"{'Sample':<14} {'W matrix':<18} {'N':>6}  "
          f"{'beta':>8} {'SE':>6}  {'lambda':>8} {'SE':>6}  {'logLL':>11}  {'AIC':>11}")
    print("-" * W)
    for sample_label, _ in [("Full", None), ("Non-border", None)]:
        for w_label, _ in W_MATRICES:
            r = results[(sample_label, w_label)]
            print(f"{r['sample']:<14} {r['w_matrix']:<18} {r['n_counties']:>6}  "
                  f"{r['beta']:>8.4f}{stars(r['p_beta'])} {r['se_beta']:>6.4f}  "
                  f"{r['lam']:>8.4f}{stars(r['p_lam'])} {r['se_lam']:>6.4f}  "
                  f"{r['logll']:>11.2f}  {r['aic']:>11.2f}")
        print()
    print("Significance: *** p<0.01  ** p<0.05  * p<0.10")

    print()
    print("=" * W)
    print("TABLE B -- Lambda gap vs W_geo | z_gap one-sided H1: lam_bank > lam_geo")
    print("=" * W)
    print(f"{'Sample':<14} {'W vs W_geo':<18}  "
          f"{'gap':>7} {'se_gap':>7} {'z_gap':>7} {'p(1-tail)':>10}  "
          f"{'LR stat':>9} {'p_LR':>8}")
    print("-" * W)
    for sample_label, _ in [("Full", None), ("Non-border", None)]:
        r_geo = results[(sample_label, "W_geo")]
        for w_label in BANK_WS:
            r = results[(sample_label, w_label)]
            gap, se_gap, z_gap, p_gap, lr, p_lr = gap_stat(r, r_geo)
            print(f"{sample_label:<14} {w_label:<18}  "
                  f"{gap:>7.4f} {se_gap:>7.4f} {z_gap:>7.3f} {p_gap:>10.4f}{stars(p_gap)}  "
                  f"{lr:>9.2f} {p_lr:>8.4f}{stars(p_lr)}")
        print()
    print("Significance: *** p<0.01  ** p<0.05  * p<0.10")

    if output_dir is not None:
        csv_rows = []
        for sample_label, _ in [("Full", None), ("Non-border", None)]:
            r_geo = results[(sample_label, "W_geo")]
            for w_label, _ in W_MATRICES:
                r = results[(sample_label, w_label)]
                row = {k: r[k] for k in
                       ["sample","w_matrix","n_counties","n_obs",
                        "beta","se_beta","p_beta","lam","se_lam","p_lam"]}
                if w_label == "W_geo":
                    row.update(gap_vs_geo=np.nan, se_gap=np.nan,
                               z_gap=np.nan, p_gap_onesided=np.nan)
                else:
                    gap, se_gap, z_gap, p_gap, _, _ = gap_stat(r, r_geo)
                    row.update(gap_vs_geo=gap, se_gap=se_gap,
                               z_gap=z_gap, p_gap_onesided=p_gap)
                csv_rows.append(row)
        cols = ["sample","w_matrix","n_counties","n_obs",
                "beta","se_beta","p_beta","lam","se_lam","p_lam",
                "gap_vs_geo","se_gap","z_gap","p_gap_onesided"]
        pd.DataFrame(csv_rows)[cols].to_csv(
            output_dir / "four_w_comparison_credit.csv", index=False)
        print(f"\nSaved four_w_comparison_credit.csv to {output_dir}")

    return results


if __name__ == '__main__':
    run(Path(__file__).parent.parent / 'output')
