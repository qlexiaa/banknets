"""
Year-by-year cross-sectional SEM: lambda_geo vs lambda_bank over 1995-2005
==========================================================================
Tests whether bank consolidation post-IBBEA (1994) progressively
raises spatial error correlation under W_bank while lambda_geo stays flat.

Model each year: ML_Error on cross-section
  y  = Linter_ela
  x  = [const, Linter_bra]
  W  = W_geo  OR  W_bank_avg

Records lambda, SE(lambda), beta(Linter_bra), SE(beta) for each year x W.
"""
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import scipy.sparse
import pandas as pd
import spreg
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils  # noqa: applies spreg ML_Error / Panel_FE_* compatibility patch
from utils import gal_to_W, row_standardize, sparse_to_pysal_w

ROOT        = Path(__file__).parent.parent
PANEL_PATH  = ROOT / 'data' / 'estimation_panel.csv'
GAL_PATH    = ROOT / 'data' / 'W_geo_queen.gal'
COUNTY_PATH = ROOT / 'data' / 'county_order_Wgeo.csv'
WBANK_PATH  = ROOT / 'data' / 'W_bank_avg.npz'

YEARS = list(range(1995, 2006))


def run_year(df_yr, W_all, county_order, label):
    """
    Subset W to counties present and non-null in this year's cross-section,
    drop any resulting islands, then run ML_Error.
    Returns dict of results, or None if estimation fails.
    """
    df = df_yr.dropna(subset=['Linter_ela', 'Linter_bra']).copy()

    # Restrict to counties in the global order
    df = df[df['fips5'].isin(set(county_order))].copy()

    # Index into full W
    present  = [c for c in county_order if c in set(df['fips5'])]
    idx      = np.array([county_order.index(c) for c in present])
    W_sub_rs = row_standardize(W_all[idx, :][:, idx])

    # Drop islands (zero rows after subsetting)
    row_sums  = np.array(W_sub_rs.sum(axis=1)).flatten()
    keep_mask = row_sums > 0
    if not keep_mask.all():
        present  = [c for c, k in zip(present, keep_mask) if k]
        idx      = np.array([county_order.index(c) for c in present])
        W_sub_rs = row_standardize(W_all[idx, :][:, idx])

    df = df[df['fips5'].isin(set(present))].copy()

    # Sort to match W row order
    pos = {c: i for i, c in enumerate(present)}
    df  = df.assign(_r=df['fips5'].map(pos)).sort_values('_r').drop(columns='_r')

    y = df[['Linter_ela']].values.astype(float)
    x = df[['Linter_bra']].values.astype(float)   # spreg adds constant

    w_pysal = sparse_to_pysal_w(W_sub_rs)

    try:
        res = spreg.ML_Error(y, x, w=w_pysal,
                             name_y='Linter_ela', name_x=['Linter_bra'],
                             name_w=label)
    except Exception:
        return None

    # betas layout: [const, Linter_bra] -- lambda is separate in res.lam
    # std_err / z_stat include lambda as the last entry
    beta_bra = float(res.betas[1, 0])
    se_bra   = float(res.std_err[1])

    lam   = float(res.lam)
    se_l  = float(res.std_err[-1])
    p_l   = float(res.z_stat[-1][1])

    return dict(
        n       = len(df),
        beta    = beta_bra,
        se_beta = se_bra,
        lam     = lam,
        se_lam  = se_l,
        p_lam   = p_l,
    )


def run(output_dir=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load & verify ──────────────────────────────────────────────────────
    co_df        = pd.read_csv(COUNTY_PATH, dtype={'fips5': str})
    county_order = co_df['fips5'].str.zfill(5).tolist()
    N_ALL        = len(county_order)

    W_geo_all, gal_order = gal_to_W(GAL_PATH, county_order)
    assert gal_order == county_order, 'GAL order != county_order_Wgeo.csv'

    W_bank_all = row_standardize(scipy.sparse.load_npz(WBANK_PATH))
    assert W_bank_all.shape == (N_ALL, N_ALL)

    panel = pd.read_csv(PANEL_PATH)
    panel['fips5'] = panel['fips5'].astype(str).str.zfill(5)

    # ── 2. Year-by-year ML_Error ──────────────────────────────────────────────
    rows = []
    for yr in YEARS:
        df_yr  = panel[panel['year'] == yr].copy()
        r_geo  = run_year(df_yr, W_geo_all,  county_order, 'W_geo')
        r_bank = run_year(df_yr, W_bank_all, county_order, 'W_bank')
        if r_geo is None or r_bank is None:
            continue
        gap = r_bank['lam'] - r_geo['lam']
        rows.append(dict(year=yr,
                         lam_geo=r_geo['lam'],   se_geo=r_geo['se_lam'],
                         p_geo=r_geo['p_lam'],   n_geo=r_geo['n'],
                         lam_bank=r_bank['lam'], se_bank=r_bank['se_lam'],
                         p_bank=r_bank['p_lam'], n_bank=r_bank['n'],
                         gap=gap,
                         beta_geo=r_geo['beta'], beta_bank=r_bank['beta']))

    df_res = pd.DataFrame(rows)
    years  = df_res['year'].values

    # ── 3. OLS trend regressions ──────────────────────────────────────────────
    trend_rows = []
    for col, label in [('lam_geo', 'W_geo'), ('lam_bank', 'W_bank'), ('gap', 'gap (bank-geo)')]:
        y_trend = df_res[col].values
        X_trend = sm.add_constant(years.astype(float))
        res_ols = sm.OLS(y_trend, X_trend).fit()
        slope   = res_ols.params[1]
        p_slope = res_ols.pvalues[1]
        r2      = res_ols.rsquared
        sig     = p_slope < 0.05
        trend_rows.append(dict(
            series=label, slope=slope, p_value=p_slope,
            r_squared=r2, significant=sig,
        ))

    # ── 4. Save outputs ───────────────────────────────────────────────────────
    if output_dir is not None:
        results_rows = []
        for row in rows:
            results_rows.append(dict(
                year=row['year'],
                n_geo=row['n_geo'],    lam_geo=row['lam_geo'],   se_geo=row['se_geo'],
                n_bank=row['n_bank'],  lam_bank=row['lam_bank'], se_bank=row['se_bank'],
                gap=row['gap'],
            ))
        pd.DataFrame(results_rows).to_csv(
            output_dir / "lambda_time_avg_results.csv", index=False)

        pd.DataFrame(trend_rows).to_csv(
            output_dir / "lambda_time_avg_trends.csv", index=False)

        # ── 5. Plot ───────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 5.5))

        yrs    = df_res['year'].values
        l_geo  = df_res['lam_geo'].values
        s_geo  = df_res['se_geo'].values
        l_bank = df_res['lam_bank'].values
        s_bank = df_res['se_bank'].values

        ax.fill_between(yrs, l_geo  - 1.96*s_geo,  l_geo  + 1.96*s_geo,
                        alpha=0.18, color='steelblue')
        ax.fill_between(yrs, l_bank - 1.96*s_bank, l_bank + 1.96*s_bank,
                        alpha=0.18, color='firebrick')

        ax.plot(yrs, l_geo,  'o-', color='steelblue', lw=2,   ms=6,
                label='W_geo (queen contiguity)')
        ax.plot(yrs, l_bank, 's-', color='firebrick', lw=2,   ms=6,
                label='W_bank_avg')

        ax.axvline(1994, color='black', lw=1.3, ls='--', alpha=0.7)
        ax.text(1994.1, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0.05,
                'IBBEA\n(1994)', fontsize=8.5, va='bottom', color='black', alpha=0.8)

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Spatial error parameter lambda', fontsize=12)
        ax.set_title('Spatial error correlation over time: bank network vs geography\n'
                     '(cross-sectional ML_Error per year, 95% CI bands)', fontsize=12)
        ax.legend(fontsize=11)
        ax.set_xticks(YEARS)
        ax.set_xlim(1993.5, 2005.5)
        ax.grid(axis='y', alpha=0.35)

        plt.tight_layout()
        plt.savefig(output_dir / 'lambda_time_variation.png', dpi=150)
        plt.close()

    return dict(df_res=df_res, trend_rows=trend_rows)


if __name__ == '__main__':
    run(Path(__file__).parent.parent / 'output')
