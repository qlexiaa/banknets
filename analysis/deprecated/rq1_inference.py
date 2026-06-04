"""
rq1_inference.py -- Formal inference for RQ1: W_bank vs W_geo
==============================================================
Runs three inference tasks on the Panel_FE_Error estimates from
panel_fe_error.py (W_geo and W_bank, full and non-border samples):

  Task 1 -- Likelihood Ratio test
    LR = 2*(logll_bank - logll_geo) ~ chi2(df=1)
    Tests whether W_bank yields a significantly better fit than W_geo.

  Task 2 -- Lambda gap significance
    gap      = lam_bank - lam_geo
    se_gap   = sqrt(se_bank^2 + se_geo^2)
    z_gap    = gap / se_gap  (one-sided: H1: lam_bank > lam_geo)
    Tests whether the lambda gap is statistically significant.

  Task 3 -- Beta comparison: OLS (state-clustered) vs SEM(W_geo) vs SEM(W_bank)
    Uses the same y_long / x_long arrays from build_arrays() for OLS.
    State-clustered SE: state = fips5[:2].

Outputs (saved to output_dir):
  lr_test_results.csv
  lambda_inference_results.csv
  beta_comparison_table.csv
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.stats as stats
import statsmodels.api as sm
import spreg

import utils  # noqa: applies spreg compatibility patch
from utils import gal_to_W, row_standardize, sparse_to_pysal_w

ROOT        = Path(__file__).parent.parent
PANEL_PATH  = ROOT / "data" / "estimation_panel.csv"
GAL_PATH    = ROOT / "data" / "W_geo_queen.gal"
COUNTY_PATH = ROOT / "data" / "county_order_Wgeo.csv"
WBANK_PATH  = ROOT / "data" / "W_bank_avg.npz"


# ── Build arrays (copied from panel_fe_error.py; returns df for OLS clustering) ──
def build_arrays(panel, county_order, W_geo_all, W_bank_all, YEARS, year_pos,
                 N_ALL, panel_sub, sample_label):
    """
    Build y and x in LONG format for Panel_FE_Error.
    Returns df (sorted long-format DataFrame) in addition to arrays,
    so that state IDs for OLS clustering can be extracted.
    """
    T = len(YEARS)
    na_all     = panel_sub.groupby("fips5")["Linter_ela"].apply(lambda s: s.isna().all())
    sub_co     = set(panel_sub["fips5"].unique())
    usable     = [c for c in county_order if c in sub_co and not na_all.get(c, True)]
    N          = len(usable)
    usable_pos = {c: i for i, c in enumerate(usable)}

    df = (panel_sub[panel_sub["fips5"].isin(set(usable))]
          .assign(t_idx=lambda d: d["year"].map(year_pos),
                  c_idx=lambda d: d["fips5"].map(usable_pos))
          .sort_values(["t_idx", "c_idx"]))

    y_long       = df["Linter_ela"].values.reshape(-1, 1)
    dummy_years  = YEARS[1:]
    t_idx_vec    = df["t_idx"].values
    year_dummies = np.column_stack([
        (t_idx_vec == year_pos[yr]).astype(np.float64)
        for yr in dummy_years
    ])
    x_long = np.hstack([df["Linter_bra"].values.reshape(-1, 1), year_dummies])

    assert not np.isnan(y_long).any(), "NaN in y"
    assert y_long.shape == (N * T, 1),  f"y wrong shape: {y_long.shape}"
    assert x_long.shape == (N * T, 11), f"x wrong shape: {x_long.shape}"

    idx        = np.array([county_order.index(c) for c in usable])
    W_geo_sub  = row_standardize(W_geo_all[idx, :][:, idx])
    W_bank_sub = row_standardize(W_bank_all[idx, :][:, idx])

    return y_long, x_long, sparse_to_pysal_w(W_geo_sub), sparse_to_pysal_w(W_bank_sub), N, df


# ── Run Panel_FE_Error (copied from panel_fe_error.py) ──────────────────────────
def run_panel_fe(y, x, w_pysal, w_label, ds_label, YEARS):
    nx = ["Linter_bra"] + [f"yr{yr}" for yr in YEARS[1:]]
    return spreg.Panel_FE_Error(
        y, x, w_pysal,
        name_y="Linter_ela",
        name_x=nx,
        name_w=w_label,
        name_ds=ds_label,
    )


# ── OLS with state-clustered SE ──────────────────────────────────────────────────
def run_ols_clustered(y_long, x_long, df):
    """
    Pooled OLS with state-clustered standard errors.
    Uses the same y_long / x_long arrays as Panel_FE_Error (year dummies included).
    An intercept is added (Panel_FE_Error removes it via within-transformation;
    the pooled OLS needs it explicitly).
    State = first 2 digits of fips5.
    """
    state_ids = df["fips5"].str[:2].values
    x_ols     = sm.add_constant(x_long)   # prepend constant column
    ols_base  = sm.OLS(y_long.flatten(), x_ols).fit()
    ols_clust = ols_base.get_robustcov_results(cov_type='cluster', groups=state_ids)
    # Linter_bra is column 1 in x_ols (column 0 = constant)
    beta = float(ols_clust.params[1])
    se   = float(ols_clust.bse[1])
    z    = float(ols_clust.tvalues[1])
    p    = float(ols_clust.pvalues[1])
    return dict(beta=beta, se=se, z=z, p=p, nobs=int(ols_clust.nobs))


def run(output_dir=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load shared inputs ────────────────────────────────────────────────────
    co_df        = pd.read_csv(COUNTY_PATH, dtype={"fips5": str})
    county_order = co_df["fips5"].str.zfill(5).tolist()
    N_ALL        = len(county_order)

    W_geo_all, gal_order = gal_to_W(GAL_PATH, county_order)
    assert gal_order == county_order, "GAL order != county_order_Wgeo.csv"

    W_bank_all = row_standardize(scipy.sparse.load_npz(WBANK_PATH))

    panel = pd.read_csv(PANEL_PATH)
    panel["fips5"] = panel["fips5"].astype(str).str.zfill(5)
    YEARS    = sorted(panel["year"].unique())
    T        = len(YEARS)
    year_pos = {yr: i for i, yr in enumerate(YEARS)}

    panel_nb = panel[panel["border"] == 0].copy()

    # ── Build arrays and estimate models ──────────────────────────────────────
    print("Estimating Panel_FE_Error models...")

    y_f, x_f, w_geo_f, w_bank_f, N_f, df_f = build_arrays(
        panel, county_order, W_geo_all, W_bank_all, YEARS, year_pos, N_ALL,
        panel, "Full sample")

    res_geo_f  = run_panel_fe(y_f, x_f, w_geo_f,  "W_geo",  "Full", YEARS)
    res_bank_f = run_panel_fe(y_f, x_f, w_bank_f, "W_bank", "Full", YEARS)

    y_nb, x_nb, w_geo_nb, w_bank_nb, N_nb, df_nb = build_arrays(
        panel, county_order, W_geo_all, W_bank_all, YEARS, year_pos, N_ALL,
        panel_nb, "Non-border")

    res_geo_nb  = run_panel_fe(y_nb, x_nb, w_geo_nb,  "W_geo",  "NB", YEARS)
    res_bank_nb = run_panel_fe(y_nb, x_nb, w_bank_nb, "W_bank", "NB", YEARS)

    print("Done.\n")

    # ════════════════════════════════════════════════════════════════════════════
    # Task 1 -- Likelihood Ratio test: W_bank vs W_geo
    # ════════════════════════════════════════════════════════════════════════════
    print("=" * 65)
    print("TASK 1 -- Likelihood Ratio Test: W_bank vs W_geo")
    print("=" * 65)
    print(f"{'Sample':<14} {'logLL(geo)':>11} {'logLL(bank)':>12} "
          f"{'LR stat':>9} {'p-value':>9} {'sig':>5}")
    print("-" * 65)

    lr_rows = []
    for label, res_geo, res_bank in [
        ("Full",        res_geo_f,  res_bank_f),
        ("Non-border",  res_geo_nb, res_bank_nb),
    ]:
        ll_geo  = float(res_geo.logll)
        ll_bank = float(res_bank.logll)
        lr_stat = 2.0 * (ll_bank - ll_geo)
        p_val   = float(stats.chi2.sf(lr_stat, df=1))
        sig     = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else
                  ("*"   if p_val < 0.10 else ""))
        print(f"{label:<14} {ll_geo:>11.3f} {ll_bank:>12.3f} "
              f"{lr_stat:>9.3f} {p_val:>9.4f} {sig:>5}")
        lr_rows.append(dict(
            sample=label,
            logll_geo=ll_geo,
            logll_bank=ll_bank,
            lr_stat=lr_stat,
            p_value=p_val,
            significant_05=(p_val < 0.05),
        ))

    print()

    # ════════════════════════════════════════════════════════════════════════════
    # Task 2 -- Lambda gap significance test (one-sided)
    # ════════════════════════════════════════════════════════════════════════════
    print("=" * 72)
    print("TASK 2 -- Lambda Gap Significance: H1: lam_bank > lam_geo")
    print("=" * 72)
    print(f"{'Sample':<14} {'lam_geo':>8} {'SE':>7} {'lam_bank':>9} {'SE':>7} "
          f"{'gap':>7} {'se_gap':>7} {'z_gap':>7} {'p(1-tail)':>10} {'sig':>5}")
    print("-" * 72)

    lam_rows = []
    for label, res_geo, res_bank in [
        ("Full",       res_geo_f,  res_bank_f),
        ("Non-border", res_geo_nb, res_bank_nb),
    ]:
        lam_geo  = float(res_geo.lam)
        lam_bank = float(res_bank.lam)
        se_geo   = float(res_geo.std_err[-1])
        se_bank  = float(res_bank.std_err[-1])
        gap      = lam_bank - lam_geo
        se_gap   = np.sqrt(se_geo**2 + se_bank**2)
        z_gap    = gap / se_gap
        p_one    = float(stats.norm.sf(z_gap))   # one-sided: P(Z > z_gap)
        sig      = "***" if p_one < 0.01 else ("**" if p_one < 0.05 else
                   ("*"   if p_one < 0.10 else ""))
        print(f"{label:<14} {lam_geo:>8.4f} {se_geo:>7.4f} {lam_bank:>9.4f} "
              f"{se_bank:>7.4f} {gap:>7.4f} {se_gap:>7.4f} {z_gap:>7.3f} "
              f"{p_one:>10.4f} {sig:>5}")
        lam_rows.append(dict(
            sample=label,
            lam_geo=lam_geo,   se_geo=se_geo,
            lam_bank=lam_bank, se_bank=se_bank,
            gap=gap, se_gap=se_gap,
            z_gap=z_gap, p_onesided=p_one,
            significant_05=(p_one < 0.05),
        ))

    print()

    # ════════════════════════════════════════════════════════════════════════════
    # Task 3 -- Beta comparison: OLS (clustered) vs SEM(W_geo) vs SEM(W_bank)
    # ════════════════════════════════════════════════════════════════════════════
    print("=" * 72)
    print("TASK 3 -- Beta Comparison: OLS vs SEM(W_geo) vs SEM(W_bank)")
    print("  Dependent var: Linter_ela  |  Regressor: Linter_bra")
    print("  OLS: pooled + year dummies + state-clustered SE")
    print("  SEM: Panel_FE_Error (county+year FE, ML)")
    print("=" * 72)
    print(f"{'Sample':<14} {'Model':<18} {'beta':>8} {'SE':>7} {'z/t':>8} {'p':>8}")
    print("-" * 72)

    beta_rows = []
    for label, y_long, x_long, df_long, res_geo, res_bank in [
        ("Full",       y_f,  x_f,  df_f,  res_geo_f,  res_bank_f),
        ("Non-border", y_nb, x_nb, df_nb, res_geo_nb, res_bank_nb),
    ]:
        # OLS
        ols_r = run_ols_clustered(y_long, x_long, df_long)
        print(f"{label:<14} {'OLS (clustered)':<18} {ols_r['beta']:>8.4f} "
              f"{ols_r['se']:>7.4f} {ols_r['z']:>8.3f} {ols_r['p']:>8.4f}")
        beta_rows.append(dict(
            sample=label, model="OLS_clustered",
            beta=ols_r["beta"], se=ols_r["se"],
            z_stat=ols_r["z"],  p_value=ols_r["p"],
            nobs=ols_r["nobs"],
        ))

        # SEM(W_geo)
        b_geo  = float(res_geo.betas[0, 0])
        se_geo = float(res_geo.std_err[0])
        z_geo  = float(res_geo.z_stat[0][0])
        p_geo  = float(res_geo.z_stat[0][1])
        print(f"{'':14} {'SEM(W_geo)':<18} {b_geo:>8.4f} "
              f"{se_geo:>7.4f} {z_geo:>8.3f} {p_geo:>8.4f}")
        beta_rows.append(dict(
            sample=label, model="SEM_W_geo",
            beta=b_geo, se=se_geo,
            z_stat=z_geo, p_value=p_geo,
            nobs=int(res_geo.n * res_geo.t),
        ))

        # SEM(W_bank)
        b_bank  = float(res_bank.betas[0, 0])
        se_bank = float(res_bank.std_err[0])
        z_bank  = float(res_bank.z_stat[0][0])
        p_bank  = float(res_bank.z_stat[0][1])
        print(f"{'':14} {'SEM(W_bank)':<18} {b_bank:>8.4f} "
              f"{se_bank:>7.4f} {z_bank:>8.3f} {p_bank:>8.4f}")
        beta_rows.append(dict(
            sample=label, model="SEM_W_bank",
            beta=b_bank, se=se_bank,
            z_stat=z_bank, p_value=p_bank,
            nobs=int(res_bank.n * res_bank.t),
        ))
        print()

    # ── Save outputs ──────────────────────────────────────────────────────────
    if output_dir is not None:
        pd.DataFrame(lr_rows).to_csv(
            output_dir / "lr_test_results.csv", index=False)
        pd.DataFrame(lam_rows).to_csv(
            output_dir / "lambda_inference_results.csv", index=False)
        pd.DataFrame(beta_rows).to_csv(
            output_dir / "beta_comparison_table.csv", index=False)
        print(f"Saved lr_test_results.csv, lambda_inference_results.csv, "
              f"beta_comparison_table.csv to {output_dir}")

    return dict(lr_rows=lr_rows, lam_rows=lam_rows, beta_rows=beta_rows)


if __name__ == '__main__':
    run(ROOT / 'output')
