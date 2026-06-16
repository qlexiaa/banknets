"""
lm_diagnostics_credit.py -- LM diagnostics for credit-growth panel OLS
========================================================================
Runs spatial LM diagnostics for the credit-growth dependent variable
Dl_nloans_b under W_geo and W_bank, for full and border samples.

Mean specification matches sem_credit.py:
  Dl_nloans_b_it = beta * Linter_bra_it + gamma * X_ct
                    + county FE + year FE + u_it

Implementation note
-------------------
spreg provides panel-specific LM diagnostics:
  panel_LMerror, panel_LMlag, panel_rLMerror, panel_rLMlag.

Those functions estimate OLS internally. To make their OLS residuals match the
two-way fixed-effects mean specification, this script feeds them y and x after
two-way within transformation (county means and year means removed, grand mean
added back), stacked time-major in the same county order used by the W matrices.
If those panel functions are unavailable, the script falls back to the
cross-sectional diagnostics_sp.LMtests on the same within-transformed stacked
arrays and the block-diagonal panel W.

Output
------
  output/lm_diagnostics_credit.csv
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.sparse
import spreg
import spreg.diagnostics as spreg_diagnostics
from libpysal.weights import WSP

sys.path.insert(0, str(Path(__file__).parents[1]))
import utils  # noqa: applies spreg compatibility patch
from panel_data import CREDIT_CONTROLS, load_panel_with_credit
from utils import row_standardize, sparse_to_pysal_w
from panel_data import get_samples
from w_variants import load_w_geo, load_bank_variants


ROOT        = Path(__file__).parents[2]
COUNTY_PATH = ROOT / "data" / "county_order_Wgeo.csv"

DV = "Dl_nloans_b"
X_VARS = ["Linter_bra"] + CREDIT_CONTROLS

# Compatibility patch for some spreg builds where diagnostics.breusch_pagan
# references a module-level linear-algebra alias that is not imported.
if not hasattr(spreg_diagnostics, "la"):
    spreg_diagnostics.la = la


def two_way_within(df, cols):
    out = df.copy()
    for col in cols:
        grand = df[col].mean()
        county_mean = df.groupby("fips5")[col].transform("mean")
        year_mean = df.groupby("year")[col].transform("mean")
        out[col] = df[col] - county_mean - year_mean + grand
    return out


def build_arrays(panel_sub, county_order, W_all, sample_label, w_label):
    """Build two-way-demeaned y/x arrays and matching PySAL W."""
    years = sorted(panel_sub["year"].unique())
    t = len(years)
    year_pos = {yr: i for i, yr in enumerate(years)}

    any_nan = panel_sub.groupby("fips5")[[DV] + X_VARS].apply(
        lambda g: g.isna().any().any()
    )
    counts = panel_sub.groupby("fips5")["year"].nunique()
    sub_co = set(panel_sub["fips5"].unique())
    usable = [
        c for c in county_order
        if c in sub_co and counts.get(c, 0) == t and not any_nan.get(c, True)
    ]

    n = len(usable)
    usable_pos = {c: i for i, c in enumerate(usable)}
    df = (
        panel_sub[panel_sub["fips5"].isin(set(usable))]
        .assign(
            t_idx=lambda d: d["year"].map(year_pos),
            c_idx=lambda d: d["fips5"].map(usable_pos),
        )
        .sort_values(["t_idx", "c_idx"])
        .reset_index(drop=True)
    )

    df_dm = two_way_within(df, [DV] + X_VARS)
    y = df_dm[DV].values.reshape(-1, 1).astype(float)
    x = df_dm[X_VARS].values.astype(float)

    assert not np.isnan(y).any(), f"NaN in y ({sample_label}, {w_label})"
    assert not np.isnan(x).any(), f"NaN in x ({sample_label}, {w_label})"
    assert y.shape == (n * t, 1), f"y shape {y.shape}, expected {(n * t, 1)}"
    assert x.shape == (n * t, len(X_VARS)), f"x shape {x.shape}, expected {(n * t, len(X_VARS))}"

    idx = np.array([county_order.index(c) for c in usable])
    W_sub = row_standardize(W_all[idx, :][:, idx])
    return y, x, sparse_to_pysal_w(W_sub), n, n * t


def run_lm_suite(y, x, w):
    """
    Return standard and robust LM diagnostics.

    Preferred path: spreg panel-specific LM functions on within-transformed
    arrays. Fallback path: cross-sectional diagnostics_sp.LMtests on a stacked
    panel with kron(I_T, W), used only if panel LM functions are unavailable.
    """
    required = ["panel_LMerror", "panel_LMlag", "panel_rLMerror", "panel_rLMlag"]
    if all(hasattr(spreg, name) for name in required):
        lme, p_lme = spreg.panel_LMerror(y, x, w)
        lml, p_lml = spreg.panel_LMlag(y, x, w)
        rlme, p_rlme = spreg.panel_rLMerror(y, x, w)
        rlml, p_rlml = spreg.panel_rLMlag(y, x, w)
        method = "spreg.panel_LM* on two-way within-transformed arrays"
    else:
        from spreg.diagnostics_sp import LMtests

        n = w.n
        t = y.shape[0] // n
        W_nt = scipy.sparse.kron(scipy.sparse.identity(t), w.sparse, format="csr")
        w_nt = WSP(W_nt).to_W(silence_warnings=True)
        ols = spreg.OLS(y, x, w=w_nt, spat_diag=False, moran=False)
        tests = LMtests(ols, w_nt)
        lme, p_lme = tests.lme
        lml, p_lml = tests.lml
        rlme, p_rlme = tests.rlme
        rlml, p_rlml = tests.rlml
        method = "diagnostics_sp.LMtests fallback on block-diagonal panel W"

    return {
        "LM_error": float(lme),
        "p_LM_error": float(p_lme),
        "LM_lag": float(lml),
        "p_LM_lag": float(p_lml),
        "rLM_error": float(rlme),
        "p_rLM_error": float(p_rlme),
        "rLM_lag": float(rlml),
        "p_rLM_lag": float(p_rlml),
        "method": method,
    }


def decision(row):
    sig_lme = row["p_LM_error"] < 0.05
    sig_lml = row["p_LM_lag"] < 0.05
    sig_rlme = row["p_rLM_error"] < 0.05
    sig_rlml = row["p_rLM_lag"] < 0.05

    if sig_lme and not sig_lml:
        return "SEM"
    if sig_lml and not sig_lme:
        return "SAR"
    if not sig_lme and not sig_lml:
        return "None"

    if sig_rlme and not sig_rlml:
        return "SEM"
    if sig_rlml and not sig_rlme:
        return "SAR"
    if sig_rlme and sig_rlml:
        return "SEM" if row["rLM_error"] >= row["rLM_lag"] else "SAR"
    return "Indeterminate"


def run(output_dir=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    co_df = pd.read_csv(COUNTY_PATH, dtype={"fips5": str})
    county_order = co_df["fips5"].str.zfill(5).tolist()

    W_geo_all, gal_order = load_w_geo(county_order)
    assert gal_order == county_order, "GAL order mismatch"

    bank_variants = load_bank_variants(county_order, W_geo_all=W_geo_all)

    panel = load_panel_with_credit()
    panel["fips5"] = panel["fips5"].astype(str).str.zfill(5)
    panel["year"]  = panel["year"].astype(int)

    W_MATRICES = [("W_geo", W_geo_all)] + list(bank_variants.items())

    rows = []
    for sample_label, panel_sub in get_samples(panel):
        for w_label, W_all in W_MATRICES:
            print(f"  Credit LM diagnostics: {sample_label} x {w_label} ...", flush=True)
            y, x, w, n_counties, n_obs = build_arrays(
                panel_sub, county_order, W_all, sample_label, w_label
            )
            r = run_lm_suite(y, x, w)
            row = {
                "sample": sample_label,
                "w_matrix": w_label,
                "n_counties": n_counties,
                "n_obs": n_obs,
                **r,
            }
            row["decision"] = decision(row)
            rows.append(row)

    width = 126
    print()
    print("=" * width)
    print("Credit-Growth LM Diagnostics after County and Year Fixed Effects")
    if rows:
        print(f"Method: {rows[0]['method']}")
    print("=" * width)
    print(f"{'Sample':<10} {'W':<22} {'N':>6} "
          f"{'LM-Err':>10} {'p':>8} {'LM-Lag':>10} {'p':>8} "
          f"{'rLM-Err':>10} {'p':>8} {'rLM-Lag':>10} {'p':>8} {'Decision':<13}")
    print("-" * width)
    for r in rows:
        print(f"{r['sample']:<10} {r['w_matrix']:<22} {r['n_counties']:>6} "
              f"{r['LM_error']:>10.3f} {r['p_LM_error']:>8.4f} "
              f"{r['LM_lag']:>10.3f} {r['p_LM_lag']:>8.4f} "
              f"{r['rLM_error']:>10.3f} {r['p_rLM_error']:>8.4f} "
              f"{r['rLM_lag']:>10.3f} {r['p_rLM_lag']:>8.4f} "
              f"{r['decision']:<13}")
    print("=" * width)

    if output_dir is not None:
        cols = [
            "sample", "w_matrix", "n_counties", "n_obs",
            "LM_error", "p_LM_error",
            "LM_lag", "p_LM_lag",
            "rLM_error", "p_rLM_error",
            "rLM_lag", "p_rLM_lag",
            "decision",
        ]
        pd.DataFrame(rows)[cols].to_csv(
            output_dir / "lm_diagnostics_credit.csv", index=False
        )
        print(f"\nSaved lm_diagnostics_credit.csv to {output_dir}")

    return rows


if __name__ == "__main__":
    run(ROOT / "output")
