"""
rq1_lm_tests.py -- Panel Lagrange Multiplier diagnostic tests
==============================================================
Runs four panel LM tests from spreg for each combination of
sample (full, non-border) and W matrix (W_geo, W_bank_avg):

  spreg.panel_LMerror(y, x, w)   -- LM error [Anselin 2008]
  spreg.panel_LMlag(y, x, w)     -- LM lag [Anselin 2008]
  spreg.panel_rLMerror(y, x, w)  -- Robust LM error [Elhorst 2014]
  spreg.panel_rLMlag(y, x, w)    -- Robust LM lag [Elhorst 2014]

Data format: RAW (not within-demeaned) long-format panel arrays.
  y : (N*T, 1)  -- Linter_ela, time-major sort
  x : (N*T, 1)  -- Linter_bra only, NO constant, NO year dummies
  w : N×N PySAL W, cross-sectional only

The LM functions add the constant and run their own OLS internally.

Decision rule: Anselin, Bera, Florax & Yoon (1996).

Output: output/lm_test_results.csv
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse
import spreg

import utils  # noqa: applies spreg compatibility patch
from utils import gal_to_W, row_standardize, sparse_to_pysal_w

ROOT        = Path(__file__).parent.parent
PANEL_PATH  = ROOT / "data" / "estimation_panel.csv"
GAL_PATH    = ROOT / "data" / "W_geo_queen.gal"
COUNTY_PATH = ROOT / "data" / "county_order_Wgeo.csv"
WBANK_PATH  = ROOT / "data" / "W_bank_avg.npz"


def run_lm_tests(panel_sub, county_order, W_all, w_label, sample_label):
    """
    Build raw (non-demeaned) long-format arrays for panel LM tests,
    run all four panel LM statistics, and return a result dict.

    Parameters
    ----------
    panel_sub    : DataFrame — sample to use (full or non-border slice)
    county_order : list of fips5 strings in W row order (from county_order_Wgeo.csv)
    W_all        : scipy sparse (N_ALL × N_ALL) — W_geo or W_bank, not yet subsetted
    w_label      : string label, e.g. 'W_geo' or 'W_bank'
    sample_label : string label, e.g. 'Full' or 'Non-border'
    """
    YEARS    = sorted(panel_sub["year"].unique())
    T        = len(YEARS)
    year_pos = {yr: i for i, yr in enumerate(YEARS)}

    # County subsetting: exclude counties with ANY NaN in Linter_ela or Linter_bra
    # (stricter than panel_fe_error's all-NaN filter — LM functions need clean data)
    sub_co = set(panel_sub["fips5"].unique())
    na_ela = panel_sub.groupby("fips5")["Linter_ela"].apply(lambda s: s.isna().any())
    na_bra = panel_sub.groupby("fips5")["Linter_bra"].apply(lambda s: s.isna().any())
    usable = [
        c for c in county_order
        if c in sub_co
        and not na_ela.get(c, True)
        and not na_bra.get(c, True)
    ]
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

    # Raw arrays — no constant, no year dummies, no within-demeaning
    y_raw = df["Linter_ela"].values.reshape(-1, 1)
    x_raw = df["Linter_bra"].values.reshape(-1, 1)

    assert not np.isnan(y_raw).any(), f"NaN in y_raw ({sample_label})"
    assert not np.isnan(x_raw).any(), f"NaN in x_raw ({sample_label})"
    assert y_raw.shape == (N * T, 1), f"y_raw shape: {y_raw.shape} expected ({N*T},1)"
    assert x_raw.shape == (N * T, 1), f"x_raw shape: {x_raw.shape} expected ({N*T},1)"

    # W submatrix for usable counties, row-standardised
    idx     = np.array([county_order.index(c) for c in usable])
    W_sub   = row_standardize(W_all[idx, :][:, idx])
    w_pysal = sparse_to_pysal_w(W_sub)

    # ── Four LM tests ────────────────────────────────────────────────────────
    lme_stat,  lme_p  = spreg.panel_LMerror( y_raw, x_raw, w_pysal)
    lml_stat,  lml_p  = spreg.panel_LMlag(   y_raw, x_raw, w_pysal)
    rlme_stat, rlme_p = spreg.panel_rLMerror(y_raw, x_raw, w_pysal)
    rlml_stat, rlml_p = spreg.panel_rLMlag(  y_raw, x_raw, w_pysal)

    # ── Decision rule (Anselin, Bera, Florax & Yoon 1996) ────────────────────
    sig05    = lambda p: p < 0.05
    sig_lme  = sig05(lme_p)
    sig_lml  = sig05(lml_p)
    sig_rlme = sig05(rlme_p)
    sig_rlml = sig05(rlml_p)

    if sig_lme and not sig_lml:
        decision = "SEM preferred (LM-Error sig, LM-Lag not sig)"
    elif sig_lml and not sig_lme:
        decision = "SAR preferred (LM-Lag sig, LM-Error not sig)"
    elif sig_lme and sig_lml:
        if sig_rlme and not sig_rlml:
            decision = "SEM preferred (rLM-Error sig, rLM-Lag not sig)"
        elif sig_rlml and not sig_rlme:
            decision = "SAR preferred (rLM-Lag sig, rLM-Error not sig)"
        elif not sig_rlme and not sig_rlml:
            decision = "Inconclusive (both LM sig, neither rLM sig)"
        else:
            # Both rLM significant — prefer the larger statistic
            if rlme_stat >= rlml_stat:
                decision = "SEM preferred (rLM-Error >= rLM-Lag, both sig)"
            else:
                decision = "SAR preferred (rLM-Lag > rLM-Error, both sig)"
    else:
        decision = "No spatial autocorrelation detected"

    return dict(
        sample        = sample_label,
        w_matrix      = w_label,
        n_counties    = N,
        n_obs         = N * T,
        lm_error_stat = float(lme_stat),
        lm_error_p    = float(lme_p),
        lm_lag_stat   = float(lml_stat),
        lm_lag_p      = float(lml_p),
        rlm_error_stat = float(rlme_stat),
        rlm_error_p    = float(rlme_p),
        rlm_lag_stat   = float(rlml_stat),
        rlm_lag_p      = float(rlml_p),
        decision      = decision,
    )


def run(output_dir=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load shared inputs ────────────────────────────────────────────────────
    co_df        = pd.read_csv(COUNTY_PATH, dtype={"fips5": str})
    county_order = co_df["fips5"].str.zfill(5).tolist()

    W_geo_all, gal_order = gal_to_W(GAL_PATH, county_order)
    assert gal_order == county_order, "GAL order mismatch"

    W_bank_all = row_standardize(scipy.sparse.load_npz(WBANK_PATH))

    panel = pd.read_csv(PANEL_PATH)
    panel["fips5"] = panel["fips5"].astype(str).str.zfill(5)

    panel_nb = panel[panel["border"] == 0].copy()

    # ── Run all 2 × 2 combinations ────────────────────────────────────────────
    rows = []
    for sample_label, panel_sub in [("Full", panel), ("Non-border", panel_nb)]:
        for w_label, W_all in [("W_geo", W_geo_all), ("W_bank", W_bank_all)]:
            print(f"  LM tests: {sample_label} x {w_label} ...", flush=True)
            r = run_lm_tests(panel_sub, county_order, W_all, w_label, sample_label)
            rows.append(r)

    # ── Formatted table ───────────────────────────────────────────────────────
    def stars(p):
        return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else "   "))

    def fmt(stat, p):
        return f"{stat:>10.3f}{stars(p)}"

    W = 95
    print()
    print("=" * W)
    print("Panel LM Diagnostic Tests  (spreg panel_LM*)")
    print("Dependent: Linter_ela  |  Regressor: Linter_bra  |  LM from pooled OLS residuals")
    print("=" * W)
    print(
        f"{'Sample':<14} {'W':<9} {'N':>6}  "
        f"{'LM-Error':>13}  {'LM-Lag':>13}  "
        f"{'rLM-Error':>13}  {'rLM-Lag':>13}"
    )
    print("-" * W)

    for r in rows:
        print(
            f"{r['sample']:<14} {r['w_matrix']:<9} {r['n_counties']:>6}  "
            f"{fmt(r['lm_error_stat'],  r['lm_error_p'])}  "
            f"{fmt(r['lm_lag_stat'],    r['lm_lag_p'])}  "
            f"{fmt(r['rlm_error_stat'], r['rlm_error_p'])}  "
            f"{fmt(r['rlm_lag_stat'],   r['rlm_lag_p'])}"
        )
        print(f"  -> {r['decision']}")

    print()
    print("Significance: *** p<0.01  ** p<0.05  * p<0.10")
    print("Decision rule: Anselin, Bera, Florax & Yoon (1996)")
    print("=" * W)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if output_dir is not None:
        cols = [
            "sample", "w_matrix", "n_counties", "n_obs",
            "lm_error_stat", "lm_error_p",
            "lm_lag_stat",   "lm_lag_p",
            "rlm_error_stat", "rlm_error_p",
            "rlm_lag_stat",   "rlm_lag_p",
            "decision",
        ]
        pd.DataFrame(rows)[cols].to_csv(
            output_dir / "lm_test_results.csv", index=False
        )
        print(f"\nSaved lm_test_results.csv to {output_dir}")

    return rows


if __name__ == '__main__':
    run(Path(__file__).parent.parent / 'output')
