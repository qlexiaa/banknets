"""
rq1_four_w_comparison.py -- Formal comparison of four spatial weight matrices
==============================================================================
Estimates Panel_FE_Error (ML, two-way FE) for each combination of
  W matrix  : W_geo | W_bank_bin | W_bank_count | W_bank_nonGeo
  Sample    : Full  | Non-border

W_bank_nonGeo is built on the fly by zeroing all W_bank_bin entries
at positions where W_geo is nonzero (pure non-geographic bank links only).

For each bank W, reports the gap vs W_geo and an LR test.

Output: output/four_w_comparison.csv
        Table A (beta + lambda, SEs, stars)
        Table B (gap and LR test vs W_geo baseline)
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.stats as stats
import spreg

import utils  # noqa: applies spreg compatibility patch
from utils import gal_to_W, row_standardize, sparse_to_pysal_w

ROOT           = Path(__file__).parent.parent
PANEL_PATH     = ROOT / "data" / "estimation_panel.csv"
GAL_PATH       = ROOT / "data" / "W_geo_queen.gal"
COUNTY_PATH    = ROOT / "data" / "county_order_Wgeo.csv"
WBANK_BIN_PATH = ROOT / "data" / "W_bank_avg.npz"
WBANK_CNT_PATH = ROOT / "data" / "W_bank_count_avg.npz"
WBANK_NG_PATH  = ROOT / "data" / "W_bank_nonGeo.npz"


# ── W_bank_nonGeo construction ───────────────────────────────────────────────

def build_wbank_nongeo(W_bank_sp, W_geo_sp, output_path):
    """
    Zero geo-overlapping entries from W_bank, re-row-standardise, save and return.

    For every (i,j) pair where W_geo[i,j] > 0, set W_bank[i,j] = 0.
    Then row-standardise the residual matrix.

    Parameters
    ----------
    W_bank_sp   : scipy sparse, N×N, row-standardised W_bank (binary cosine)
    W_geo_sp    : scipy sparse, N×N, row-standardised W_geo (queen contiguity)
    output_path : path to save the resulting .npz

    Returns
    -------
    W_ng_sp : scipy sparse CSR, N×N, row-standardised W_bank_nonGeo
    """
    # Dense copy so element assignment is unambiguous
    W_ng  = W_bank_sp.toarray().astype(np.float64)
    W_geo = W_geo_sp.toarray().astype(np.float64)
    np.fill_diagonal(W_ng,  0.0)
    np.fill_diagonal(W_geo, 0.0)

    # Zero entries at geo-neighbour positions
    W_ng[W_geo > 0] = 0.0
    np.fill_diagonal(W_ng, 0.0)

    W_ng_sp = row_standardize(scipy.sparse.csr_matrix(W_ng))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scipy.sparse.save_npz(str(output_path), W_ng_sp)

    return W_ng_sp


# ── Single model estimator ───────────────────────────────────────────────────

def run_one(panel_sub, county_order, W_all, w_label, sample_label, YEARS, year_pos):
    """
    Build long-format arrays and run Panel_FE_Error for one W × sample combination.

    County subsetting (same as panel_fe_error.py):
      - Exclude counties where Linter_ela is all-NaN.
      - Additionally exclude W-islands (zero-row counties in the subsetted W).

    Returns a result dict with all key statistics.
    """
    T      = len(YEARS)
    na_all = panel_sub.groupby("fips5")["Linter_ela"].apply(lambda s: s.isna().all())
    sub_co = set(panel_sub["fips5"].unique())

    # Pass 1: exclude all-NaN counties
    usable = [c for c in county_order if c in sub_co and not na_all.get(c, True)]

    # Pass 2: drop structural islands in the FULL N×N matrix
    # (same logic as bank_nongeo.py — only removes counties truly isolated in W_all,
    #  NOT counties that merely lose neighbours after sample subsetting)
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

    y_long       = df["Linter_ela"].values.reshape(-1, 1)
    t_idx_vec    = df["t_idx"].values
    year_dummies = np.column_stack([
        (t_idx_vec == year_pos[yr]).astype(np.float64)
        for yr in YEARS[1:]
    ])
    x_long = np.hstack([df["Linter_bra"].values.reshape(-1, 1), year_dummies])

    assert not np.isnan(y_long).any(), f"NaN in y ({sample_label}, {w_label})"
    assert y_long.shape == (N * T, 1),  f"y shape {y_long.shape}"
    assert x_long.shape == (N * T, 11), f"x shape {x_long.shape}"

    idx    = np.array([county_order.index(c) for c in usable])
    W_sub  = row_standardize(W_all[idx, :][:, idx])
    w_pysal = sparse_to_pysal_w(W_sub)

    nx  = ["Linter_bra"] + [f"yr{yr}" for yr in YEARS[1:]]
    res = spreg.Panel_FE_Error(
        y_long, x_long, w_pysal,
        name_y="Linter_ela",
        name_x=nx,
        name_w=w_label,
        name_ds=sample_label,
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


# ── Master run function ──────────────────────────────────────────────────────

def run(output_dir=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load shared inputs ────────────────────────────────────────────────────
    co_df        = pd.read_csv(COUNTY_PATH, dtype={"fips5": str})
    county_order = co_df["fips5"].str.zfill(5).tolist()

    W_geo_all, gal_order = gal_to_W(GAL_PATH, county_order)
    assert gal_order == county_order, "GAL order mismatch"

    W_bin_all = row_standardize(scipy.sparse.load_npz(WBANK_BIN_PATH))
    W_cnt_all = row_standardize(scipy.sparse.load_npz(WBANK_CNT_PATH))

    # Build (or reload) W_bank_nonGeo
    if WBANK_NG_PATH.exists():
        W_ng_all = row_standardize(scipy.sparse.load_npz(WBANK_NG_PATH))
        print("Loaded W_bank_nonGeo from cache.")
    else:
        print("Building W_bank_nonGeo ...", end=" ", flush=True)
        W_ng_all = build_wbank_nongeo(W_bin_all, W_geo_all, WBANK_NG_PATH)
        print("done.")

    W_MATRICES = [
        ("W_geo",          W_geo_all),
        ("W_bank_bin",     W_bin_all),
        ("W_bank_count",   W_cnt_all),
        ("W_bank_nonGeo",  W_ng_all),
    ]

    panel = pd.read_csv(PANEL_PATH)
    panel["fips5"] = panel["fips5"].astype(str).str.zfill(5)
    YEARS    = sorted(panel["year"].unique())
    year_pos = {yr: i for i, yr in enumerate(YEARS)}

    panel_nb = panel[panel["border"] == 0].copy()

    # ── Estimate all 8 models ─────────────────────────────────────────────────
    results = {}   # (sample_label, w_label) -> result dict
    for sample_label, panel_sub in [("Full", panel), ("Non-border", panel_nb)]:
        for w_label, W_all in W_MATRICES:
            print(f"  Estimating: {sample_label} x {w_label} ...", flush=True)
            r = run_one(panel_sub, county_order, W_all, w_label,
                        sample_label, YEARS, year_pos)
            results[(sample_label, w_label)] = r

    # ── Compute gap and LR statistics (bank W vs W_geo within each sample) ────
    BANK_WS = ["W_bank_bin", "W_bank_count", "W_bank_nonGeo"]

    def gap_lr(r_bank, r_geo):
        gap    = r_bank["lam"]   - r_geo["lam"]
        se_gap = np.sqrt(r_bank["se_lam"]**2 + r_geo["se_lam"]**2)
        z_gap  = gap / se_gap
        p_gap  = float(stats.norm.sf(z_gap))
        lr     = 2.0 * (r_bank["logll"] - r_geo["logll"])
        p_lr   = float(stats.chi2.sf(lr, df=1))
        return gap, se_gap, z_gap, p_gap, lr, p_lr

    # ── Build CSV rows ────────────────────────────────────────────────────────
    csv_rows = []
    for sample_label, _ in [("Full", None), ("Non-border", None)]:
        r_geo = results[(sample_label, "W_geo")]
        for w_label, _ in W_MATRICES:
            r = results[(sample_label, w_label)]
            row = {
                "sample":    r["sample"],
                "w_matrix":  r["w_matrix"],
                "n_counties": r["n_counties"],
                "n_obs":     r["n_obs"],
                "beta":      r["beta"],
                "se_beta":   r["se_beta"],
                "p_beta":    r["p_beta"],
                "lam":       r["lam"],
                "se_lam":    r["se_lam"],
                "p_lam":     r["p_lam"],
                "logll":     r["logll"],
                "aic":       r["aic"],
            }
            if w_label == "W_geo":
                row.update({
                    "gap_vs_geo":      np.nan,
                    "se_gap":          np.nan,
                    "z_gap":           np.nan,
                    "p_gap_onesided":  np.nan,
                    "LR_vs_geo":       np.nan,
                    "p_LR":            np.nan,
                })
            else:
                gap, se_gap, z_gap, p_gap, lr, p_lr = gap_lr(r, r_geo)
                row.update({
                    "gap_vs_geo":      gap,
                    "se_gap":          se_gap,
                    "z_gap":           z_gap,
                    "p_gap_onesided":  p_gap,
                    "LR_vs_geo":       lr,
                    "p_LR":            p_lr,
                })
            csv_rows.append(row)

    # ── Print Table A: estimates ──────────────────────────────────────────────
    def stars(p):
        if np.isnan(p): return "   "
        return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else "   "))

    W = 88
    print()
    print("=" * W)
    print("TABLE A — Panel_FE_Error estimates: beta and lambda (two-way FE, ML)")
    print("Dependent: Linter_ela  |  Regressor: Linter_bra  |  year FE 1996-2005")
    print("=" * W)
    print(
        f"{'Sample':<14} {'W matrix':<18} {'N':>6}  "
        f"{'beta':>8} {'SE':>6}  "
        f"{'lambda':>8} {'SE':>6}  {'logLL':>11}  {'AIC':>11}"
    )
    print("-" * W)
    for sample_label, _ in [("Full", None), ("Non-border", None)]:
        for w_label, _ in W_MATRICES:
            r = results[(sample_label, w_label)]
            sb = stars(r["p_beta"])
            sl = stars(r["p_lam"])
            print(
                f"{r['sample']:<14} {r['w_matrix']:<18} {r['n_counties']:>6}  "
                f"{r['beta']:>8.4f}{sb} {r['se_beta']:>6.4f}  "
                f"{r['lam']:>8.4f}{sl} {r['se_lam']:>6.4f}  "
                f"{r['logll']:>11.2f}  {r['aic']:>11.2f}"
            )
        print()

    print("Significance: *** p<0.01  ** p<0.05  * p<0.10")

    # ── Print Table B: gap and LR ─────────────────────────────────────────────
    print()
    print("=" * W)
    print("TABLE B — Lambda gap and LR test vs W_geo baseline")
    print("  gap = lam_bank - lam_geo  |  z_gap one-sided H1: lam_bank > lam_geo")
    print("  LR  = 2*(logLL_bank - logLL_geo) ~ chi2(df=1)  [non-nested, see note]")
    print("=" * W)
    print(
        f"{'Sample':<14} {'W vs W_geo':<18}  "
        f"{'gap':>7} {'se_gap':>7} {'z_gap':>7} {'p(1-tail)':>10}  "
        f"{'LR stat':>9} {'p_LR':>8}"
    )
    print("-" * W)
    for sample_label, _ in [("Full", None), ("Non-border", None)]:
        r_geo = results[(sample_label, "W_geo")]
        for w_label in BANK_WS:
            r = results[(sample_label, w_label)]
            gap, se_gap, z_gap, p_gap, lr, p_lr = gap_lr(r, r_geo)
            sg = stars(p_gap)
            slr = stars(p_lr)
            print(
                f"{sample_label:<14} {w_label:<18}  "
                f"{gap:>7.4f} {se_gap:>7.4f} {z_gap:>7.3f} {p_gap:>10.4f}{sg}  "
                f"{lr:>9.2f} {p_lr:>8.4f}{slr}"
            )
        print()

    print("Significance: *** p<0.01  ** p<0.05  * p<0.10")
    print("Note: LR test valid only when n_counties matches between W_geo and bank W.")
    print(f"      W_bank_nonGeo may have fewer counties due to island exclusion.")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if output_dir is not None:
        cols = [
            "sample", "w_matrix", "n_counties", "n_obs",
            "beta", "se_beta", "p_beta",
            "lam",  "se_lam",  "p_lam",
            "logll", "aic",
            "gap_vs_geo", "se_gap", "z_gap", "p_gap_onesided",
            "LR_vs_geo", "p_LR",
        ]
        pd.DataFrame(csv_rows)[cols].to_csv(
            output_dir / "four_w_comparison.csv", index=False
        )
        print(f"\nSaved four_w_comparison.csv to {output_dir}")

    return dict(results=results, csv_rows=csv_rows)


if __name__ == '__main__':
    run(Path(__file__).parent.parent / 'output')
