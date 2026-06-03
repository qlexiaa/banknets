"""
lambda_time_yearspecific_credit.py
====================================
Credit-DV parallel of lambda_time_yearspecific.py.
Dependent variable: Dl_nloans_b (dln commercial-bank mortgage loans).

Uses year-specific W_bank_t matrices from data/W_bank_yearly/ (already built by
lambda_time_yearspecific.py) and the fixed W_geo for cross-sectional ML_Error
year-by-year, 1997-2005.

Tests whether the bank-network spatial error parameter grows over time (IBBEA
consolidation hypothesis) in the credit-growth equation.

Prerequisites:
  data/W_bank_yearly/W_bank_1997.npz ... W_bank_2005.npz must exist.
  Run lambda_time_yearspecific.py first if they are missing.

Outputs:
  output/lambda_time_yearspecific_credit_results.csv
    columns = [year, lam_geo, se_lam_geo, lam_bank, se_lam_bank, gap, n_geo, n_bank]
  output/lambda_time_yearspecific_credit_trends.csv
    columns = [series, slope, intercept, r2, p_value]
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse
import spreg
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent))
import utils  # noqa
from utils import gal_to_W, row_standardize, sparse_to_pysal_w
from panel_fe_credit import load_panel_with_credit

ROOT        = Path(__file__).parent.parent
GAL_PATH    = ROOT / "data" / "W_geo_queen.gal"
COUNTY_PATH = ROOT / "data" / "county_order_Wgeo.csv"
WBANK_DIR   = ROOT / "data" / "W_bank_yearly"

MODEL_YEARS = list(range(1997, 2006))   # 1997-2005 (skip degenerate early years)
DV          = "Dl_nloans_b"


def run_year(df_yr, W_all, county_order, county_set, county_idx, w_label):
    """
    Subset W to the year's valid cross-section, drop islands, run ML_Error.
    Uses Dl_nloans_b as the outcome. Returns result dict or None on failure.
    """
    df = df_yr.dropna(subset=[DV, 'Linter_bra']).copy()
    df = df[df['fips5'].isin(county_set)].copy()

    present  = [c for c in county_order if c in set(df['fips5'])]
    idx      = np.array([county_idx[c] for c in present])
    W_sub    = row_standardize(W_all[idx, :][:, idx])

    # Drop islands
    row_sums  = np.array(W_sub.sum(axis=1)).flatten()
    keep_mask = row_sums > 0
    if not keep_mask.all():
        present  = [c for c, k in zip(present, keep_mask) if k]
        idx      = np.array([county_idx[c] for c in present])
        W_sub    = row_standardize(W_all[idx, :][:, idx])

    df = df[df['fips5'].isin(set(present))].copy()
    pos = {c: i for i, c in enumerate(present)}
    df  = df.assign(_r=df['fips5'].map(pos)).sort_values('_r').drop(columns='_r')

    y = df[[DV]].values.astype(float)
    x = df[['Linter_bra']].values.astype(float)
    w_pysal = sparse_to_pysal_w(W_sub)

    try:
        res = spreg.ML_Error(y, x, w=w_pysal,
                             name_y=DV, name_x=['Linter_bra'],
                             name_w=w_label)
    except Exception:
        return None

    return dict(
        n       = len(df),
        lam     = float(res.lam),
        se_lam  = float(res.std_err[-1]),
    )


def run(output_dir=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load shared inputs ────────────────────────────────────────────────────
    co_df        = pd.read_csv(COUNTY_PATH, dtype={"fips5": str})
    county_order = co_df["fips5"].str.zfill(5).tolist()
    county_idx   = {f: i for i, f in enumerate(county_order)}
    county_set   = set(county_order)

    W_geo_all, gal_order = gal_to_W(GAL_PATH, county_order)
    assert gal_order == county_order

    # Verify yearly W_bank matrices exist
    missing = [yr for yr in MODEL_YEARS
               if not (WBANK_DIR / f"W_bank_{yr}.npz").exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing W_bank_yearly matrices for years {missing}. "
            f"Run lambda_time_yearspecific.py first.")

    panel = load_panel_with_credit()
    panel["fips5"] = panel["fips5"].astype(str).str.zfill(5)

    # ── Year-by-year ML_Error ─────────────────────────────────────────────────
    rows = []
    for yr in MODEL_YEARS:
        print(f"  Year {yr} ...", flush=True)
        df_yr    = panel[panel["year"] == yr].copy()
        W_bank_t = scipy.sparse.load_npz(WBANK_DIR / f"W_bank_{yr}.npz")

        r_geo  = run_year(df_yr, W_geo_all, county_order,
                          county_set, county_idx, "W_geo")
        r_bank = run_year(df_yr, W_bank_t,  county_order,
                          county_set, county_idx, f"W_bank_{yr}")

        if r_geo is None or r_bank is None:
            print(f"  [WARN] Year {yr} skipped (convergence failure)")
            continue

        rows.append(dict(
            year       = yr,
            lam_geo    = r_geo["lam"],   se_lam_geo  = r_geo["se_lam"],
            lam_bank   = r_bank["lam"],  se_lam_bank = r_bank["se_lam"],
            gap        = r_bank["lam"] - r_geo["lam"],
            n_geo      = r_geo["n"],     n_bank      = r_bank["n"],
        ))

    if not rows:
        print("[ERROR] No years converged.")
        return {}

    df_res = pd.DataFrame(rows)
    years  = df_res["year"].values.astype(float)

    # ── OLS trend regressions ─────────────────────────────────────────────────
    trend_rows = []
    for col, label in [("lam_geo", "geo"), ("lam_bank", "bank"), ("gap", "gap")]:
        y_tr = df_res[col].values
        X_tr = sm.add_constant(years)
        fit  = sm.OLS(y_tr, X_tr).fit()
        trend_rows.append(dict(
            series    = label,
            slope     = float(fit.params[1]),
            intercept = float(fit.params[0]),
            r2        = float(fit.rsquared),
            p_value   = float(fit.pvalues[1]),
        ))

    # ── Print year table ──────────────────────────────────────────────────────
    W = 70
    print()
    print("=" * W)
    print(f"Year-specific W_bank lambda -- DV: {DV} | Cross-sectional ML_Error")
    print("=" * W)
    print(f"{'Year':>5}  {'lam_geo':>8} {'SE':>6}  {'lam_bank':>9} {'SE':>6}  "
          f"{'gap':>7}  {'n_geo':>6} {'n_bank':>7}")
    print("-" * W)
    for r in rows:
        print(f"{r['year']:>5}  {r['lam_geo']:>8.4f} {r['se_lam_geo']:>6.4f}  "
              f"{r['lam_bank']:>9.4f} {r['se_lam_bank']:>6.4f}  "
              f"{r['gap']:>+7.4f}  {r['n_geo']:>6} {r['n_bank']:>7}")
    print()
    print("OLS trend slopes (lambda ~ year):")
    for t in trend_rows:
        sig = "***" if t["p_value"] < 0.01 else ("**" if t["p_value"] < 0.05
              else ("*" if t["p_value"] < 0.10 else ""))
        print(f"  {t['series']:8}: slope={t['slope']:+.4f}/yr  "
              f"p={t['p_value']:.3f}{sig}  R2={t['r2']:.3f}")
    print("=" * W)

    # ── Save outputs ──────────────────────────────────────────────────────────
    if output_dir is not None:
        results_cols = ["year","lam_geo","se_lam_geo",
                        "lam_bank","se_lam_bank","gap","n_geo","n_bank"]
        df_res[results_cols].to_csv(
            output_dir / "lambda_time_yearspecific_credit_results.csv", index=False)

        trends_cols = ["series","slope","intercept","r2","p_value"]
        pd.DataFrame(trend_rows)[trends_cols].to_csv(
            output_dir / "lambda_time_yearspecific_credit_trends.csv", index=False)

        print(f"\nSaved lambda_time_yearspecific_credit_results.csv and "
              f"lambda_time_yearspecific_credit_trends.csv to {output_dir}")

    return dict(df_res=df_res, trend_rows=trend_rows)


if __name__ == '__main__':
    run(Path(__file__).parent.parent / 'output')
