"""
Year-specific W_bank matrices and lambda time-variation test
============================================================
Builds W_bank_t for each year 1995-2005 from that year's FDIC branch
data (binary HC-presence cosine similarity), then runs cross-sectional
ML_Error for 1997-2005 with both W_bank_t and W_geo to test whether
the lambda_bank gap over lambda_geo widens post-IBBEA as bank consolidation
builds a denser HC network.

Step 1  Build + save W_bank_1995.npz ... W_bank_2005.npz
Step 2  Run ML_Error for 1997-2005: W_bank_t and W_geo
Step 3  Table, OLS trend tests, plot
"""
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse
from scipy.sparse import csr_matrix
import spreg
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils  # noqa: applies spreg compatibility patch (ML_Error, Panel_FE_*)
from utils import gal_to_W, row_standardize, sparse_to_pysal_w

ROOT        = Path(__file__).parent.parent
FDIC_PATH   = ROOT / 'data' / 'fdic_deposits_1994_2005.csv'
ORDER_PATH  = ROOT / 'data' / 'county_order_Wgeo.csv'
GAL_PATH    = ROOT / 'data' / 'W_geo_queen.gal'
PANEL_PATH  = ROOT / 'data' / 'estimation_panel.csv'
WBANK_DIR   = ROOT / 'data' / 'W_bank_yearly'

BUILD_YEARS  = list(range(1995, 2006))   # matrices to build
MODEL_YEARS  = list(range(1997, 2006))   # years to estimate (skip 1995-96 degenerate)


def build_binary_year(df_year, county_idx, N):
    """Binary HC-presence cosine similarity, diagonal=0, row-standardised."""
    pairs  = df_year[['fips5', 'RSSDHCR']].drop_duplicates().reset_index(drop=True)
    hcs    = pairs['RSSDHCR'].unique()
    hc_map = {hc: i for i, hc in enumerate(hcs)}

    rows_m = pairs['fips5'].map(county_idx).values
    cols_m = pairs['RSSDHCR'].map(hc_map).values
    M = csr_matrix(
        (np.ones(len(pairs), dtype=np.float32), (rows_m, cols_m)),
        shape=(N, len(hcs))
    )
    B     = (M @ M.T).toarray().astype(np.float64)
    d     = B.diagonal().copy()
    denom = np.sqrt(np.outer(d, d))
    with np.errstate(divide='ignore', invalid='ignore'):
        W = np.where(denom > 0, B / denom, 0.0)
    np.fill_diagonal(W, 0.0)
    rs = W.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        W = np.where(rs > 0, W / rs, 0.0)
    return W, len(hcs)


def run_year(df_yr, W_all, county_order, county_set, county_idx, w_label):
    """
    Subset W to the year's valid cross-section, drop islands,
    run ML_Error. Returns result dict or None on failure.
    """
    df = df_yr.dropna(subset=['Linter_ela', 'Linter_bra']).copy()
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

    y = df[['Linter_ela']].values.astype(float)
    x = df[['Linter_bra']].values.astype(float)
    w_pysal = sparse_to_pysal_w(W_sub)

    try:
        res = spreg.ML_Error(y, x, w=w_pysal,
                             name_y='Linter_ela', name_x=['Linter_bra'],
                             name_w=w_label)
    except Exception:
        return None

    # betas: [const, Linter_bra]; lambda is separate (last in std_err / z_stat)
    return dict(
        n       = len(df),
        beta    = float(res.betas[1, 0]),
        se_beta = float(res.std_err[1]),
        lam     = float(res.lam),
        se_lam  = float(res.std_err[-1]),
        p_lam   = float(res.z_stat[-1][1]),
    )


def run(output_dir=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    WBANK_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load shared inputs ─────────────────────────────────────────────────
    co_df        = pd.read_csv(ORDER_PATH, dtype={'fips5': str})
    county_order = co_df['fips5'].str.zfill(5).tolist()
    N            = len(county_order)
    county_idx   = {f: i for i, f in enumerate(county_order)}
    county_set   = set(county_order)

    W_geo_all, gal_order = gal_to_W(GAL_PATH, county_order)
    assert gal_order == county_order, 'GAL order mismatch'

    fdic = pd.read_csv(FDIC_PATH, dtype={'STCNTYBR': str})
    fdic['fips5'] = fdic['STCNTYBR'].str.zfill(5)
    fdic = fdic[fdic['fips5'].isin(county_set)].copy()
    fdic = fdic[fdic['RSSDHCR'].notna() & (fdic['RSSDHCR'] != 0)].copy()

    # ── 2. Build and save each year ───────────────────────────────────────────
    network_rows = []
    for yr in BUILD_YEARS:
        out_path = WBANK_DIR / f'W_bank_{yr}.npz'
        df_yr = fdic[fdic['year'] == yr]
        W, n_hcs = build_binary_year(df_yr, county_idx, N)
        nz  = int((W > 0).sum())
        sp  = 1 - nz / (N * (N - 1))
        avg = nz / N
        network_rows.append(dict(year=yr, n_hcs=n_hcs, nz_pairs=nz,
                                 sparsity=sp, avg_nbrs=avg))
        scipy.sparse.save_npz(out_path, csr_matrix(W))

    # ── 3. Panel data ─────────────────────────────────────────────────────────
    panel = pd.read_csv(PANEL_PATH)
    panel['fips5'] = panel['fips5'].astype(str).str.zfill(5)

    # ── 4. Year-by-year ML_Error ──────────────────────────────────────────────
    rows = []
    for yr in MODEL_YEARS:
        df_yr    = panel[panel['year'] == yr].copy()
        W_bank_t = scipy.sparse.load_npz(WBANK_DIR / f'W_bank_{yr}.npz')

        r_geo  = run_year(df_yr, W_geo_all, county_order, county_set,
                          county_idx, 'W_geo')
        r_bank = run_year(df_yr, W_bank_t,  county_order, county_set,
                          county_idx, f'W_bank_{yr}')

        if r_geo is None or r_bank is None:
            continue

        gap = r_bank['lam'] - r_geo['lam']
        rows.append(dict(
            year     = yr,
            lam_geo  = r_geo['lam'],   se_geo  = r_geo['se_lam'],
            lam_bank = r_bank['lam'],  se_bank = r_bank['se_lam'],
            n_geo    = r_geo['n'],     n_bank  = r_bank['n'],
            gap      = gap,
        ))

    if not rows:
        return {}

    df_res = pd.DataFrame(rows)
    years  = df_res['year'].values.astype(float)

    # ── 5. OLS trend regressions ──────────────────────────────────────────────
    trend_rows = []
    for col, label in [('lam_geo', 'W_geo'), ('lam_bank', 'W_bank_t'), ('gap', 'gap')]:
        y_tr = df_res[col].values
        X_tr = sm.add_constant(years)
        fit  = sm.OLS(y_tr, X_tr).fit()
        slope, p = fit.params[1], fit.pvalues[1]
        sig = p < 0.05
        trend_rows.append(dict(
            series=label, slope=slope, p_value=p,
            r_squared=fit.rsquared, significant=sig,
        ))

    # ── 6. Save outputs ───────────────────────────────────────────────────────
    if output_dir is not None:
        results_out = []
        for row in rows:
            results_out.append(dict(
                year=row['year'],
                lam_geo=row['lam_geo'],   se_geo=row['se_geo'],
                lam_bank=row['lam_bank'], se_bank=row['se_bank'],
                gap=row['gap'],
                n_geo=row['n_geo'],       n_bank=row['n_bank'],
            ))
        pd.DataFrame(results_out).to_csv(
            output_dir / "lambda_time_yearspecific_results.csv", index=False)

        pd.DataFrame(network_rows).to_csv(
            output_dir / "lambda_time_yearspecific_network.csv", index=False)

        pd.DataFrame(trend_rows).to_csv(
            output_dir / "lambda_time_yearspecific_trends.csv", index=False)

        # ── 7. Plot ───────────────────────────────────────────────────────────
        trend_results = {r['series']: r for r in trend_rows}

        fig, axes = plt.subplots(2, 1, figsize=(10, 9),
                                 gridspec_kw={'height_ratios': [2.5, 1]})

        ax = axes[0]
        yrs    = df_res['year'].values
        l_geo  = df_res['lam_geo'].values;  s_geo  = df_res['se_geo'].values
        l_bank = df_res['lam_bank'].values; s_bank = df_res['se_bank'].values

        ax.fill_between(yrs, l_geo  - 1.96*s_geo,  l_geo  + 1.96*s_geo,
                        alpha=0.18, color='steelblue')
        ax.fill_between(yrs, l_bank - 1.96*s_bank, l_bank + 1.96*s_bank,
                        alpha=0.18, color='firebrick')
        ax.plot(yrs, l_geo,  'o-', color='steelblue', lw=2, ms=6,
                label='W_geo (queen contiguity)')
        ax.plot(yrs, l_bank, 's-', color='firebrick', lw=2, ms=6,
                label='W_bank_t (year-specific)')

        for vals, col, ls in [(l_geo, 'steelblue', '--'), (l_bank, 'firebrick', '--')]:
            X_tr  = sm.add_constant(yrs.astype(float))
            fit   = sm.OLS(vals, X_tr).fit()
            trend = fit.predict(X_tr)
            ax.plot(yrs, trend, color=col, lw=1.2, ls=ls, alpha=0.6)

        ax.axvline(1994, color='black', lw=1.3, ls=':', alpha=0.7)
        ax.text(1994.15, ax.get_ylim()[0] if ax.get_ylim()[0] > 0.5 else 0.6,
                'IBBEA\n(1994)', fontsize=8.5, va='bottom', color='black', alpha=0.8)

        ax.set_ylabel('Spatial error parameter lambda', fontsize=11)
        ax.set_title('Spatial error correlation over time: year-specific W_bank vs W_geo\n'
                     '(cross-sectional ML_Error per year, 95% CI bands, dashed = OLS trend)',
                     fontsize=11)
        ax.legend(fontsize=10)
        ax.set_xticks(MODEL_YEARS)
        ax.set_xlim(1996.3, 2005.7)
        ax.grid(axis='y', alpha=0.3)

        ax2 = axes[1]
        gap_vals = df_res['gap'].values
        ax2.bar(yrs, gap_vals, color='mediumpurple', alpha=0.7, width=0.6)
        ax2.axhline(0, color='black', lw=0.8)
        X_tr   = sm.add_constant(yrs.astype(float))
        fit_g  = sm.OLS(gap_vals, X_tr).fit()
        gap_tr = trend_results.get('gap', {})
        ax2.plot(yrs, fit_g.predict(X_tr), color='purple', lw=1.5, ls='--',
                 label=f'trend slope={fit_g.params[1]:+.4f}/yr  '
                       f'p={gap_tr.get("p_value", 0):.3f}')
        ax2.set_xlabel('Year', fontsize=11)
        ax2.set_ylabel('lambda_bank - lambda_geo', fontsize=11)
        ax2.set_title('Gap (lambda_bank_t - lambda_geo)', fontsize=10)
        ax2.legend(fontsize=9)
        ax2.set_xticks(MODEL_YEARS)
        ax2.set_xlim(1996.3, 2005.7)
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'lambda_time_variation_yearspecific.png', dpi=150)
        plt.close()

    return dict(df_res=df_res, trend_rows=trend_rows, network_rows=network_rows)


if __name__ == '__main__':
    run(Path(__file__).parent.parent / 'output')
