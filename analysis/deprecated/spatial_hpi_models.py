"""
Spatial HPI Models — W_geo (Queen Contiguity) Baseline
=======================================================
Runs OLS, SEM (GM) and SAR (ML) panel models for the Favara & Imbs (2015)
HPI reduced-form equation, using queen contiguity weights (W_geo).

Equation
--------
  DlnP_{c,t} = b1*D_{s,t-1} + b2*D_{s,t-1}*eta_c + b3*X_{c,t}
               + alpha_c + gamma_t + eps_{c,t}

where D_{s,t-1} = Linter_bra  (lagged Rice-Strahan branching index)
      eta_c     = Saiz supply elasticity (absorbed into Linter_ela = D x eta)
      X_{c,t}   = LDl_hpi, Dl_pop, LDl_pop, Dl_inc, LDl_inc, Dl_hsosf, LDl_hsosf

Models
------
  OLS  : county + year FEs removed by within-demeaning (FWL); state-clustered SEs
  SEM  : spreg.Panel_FE_Error -- spatial error parameter lambda  (GM estimator)
  SAR  : spreg.Panel_FE_Lag   -- spatial lag parameter rho       (ML estimator)

Each model is run on the full sample and on non-border counties (border == 0).
Moran's I is reported on OLS residuals by year as motivation for spatial models.
"""
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import libpysal
import spreg
import esda
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster

import utils  # noqa: applies spreg Panel_FE_Error compatibility patch


ROOT       = Path(__file__).parent.parent
PANEL_PATH = ROOT / "data" / "estimation_panel.csv"
GPKG_PATH  = ROOT / "data" / "merged_panel.gpkg"

YVAR  = 'Dl_hpi'
XVARS = [
    'Linter_bra', 'Linter_ela',
    'LDl_hpi',
    'Dl_pop',  'LDl_pop',
    'Dl_inc',  'LDl_inc',
    'Dl_hsosf', 'LDl_hsosf',
]
YEARS = list(range(1995, 2006))


def prepare_sample(df, counties_geo, yvar, xvars, border_filter=None, label=''):
    """
    Returns (df_sorted, w) ready for spreg Panel_FE_Error / Panel_FE_Lag.
    df_sorted is stacked as T blocks of N rows, each block in W id_order.
    """
    df = df.dropna(subset=[yvar] + xvars).copy()
    if border_filter is not None:
        df = df[df['border'] == border_filter].copy()

    cnts = df.groupby('fips5')['year'].count()
    keep = cnts[cnts == 11].index
    df   = df[df['fips5'].isin(keep)].copy()

    sub_geo = (counties_geo[counties_geo['fips5'].isin(keep)]
               .copy().reset_index(drop=True))
    w = libpysal.weights.Queen.from_dataframe(
        sub_geo, ids='fips5', silence_warnings=True)

    islands = [f for f, nbrs in w.neighbors.items() if len(nbrs) == 0]
    if islands:
        sub_geo = sub_geo[~sub_geo['fips5'].isin(islands)].reset_index(drop=True)
        df      = df[df['fips5'].isin(sub_geo['fips5'])].copy()
        w       = libpysal.weights.Queen.from_dataframe(
            sub_geo, ids='fips5', silence_warnings=True)

    w.transform = 'r'

    id_to_row = {f: i for i, f in enumerate(w.id_order)}
    df = df.copy()
    df['_wr'] = df['fips5'].map(id_to_row)
    df = df.sort_values(['year', '_wr']).drop(columns='_wr').reset_index(drop=True)

    return df, w


def within_demean(df, cols, county_col='fips5', year_col='year'):
    """Two-way within-transformation (FWL for county + year FE)."""
    out = df.copy()
    for col in cols:
        gm = df[col].mean()
        cm = df.groupby(county_col)[col].transform('mean')
        ym = df.groupby(year_col)[col].transform('mean')
        out[col] = df[col] - cm - ym + gm
    return out


def run_ols(df, yvar, xvars, label=''):
    dm_cols = [yvar] + xvars
    df_dm   = within_demean(
        df[dm_cols + ['fips5', 'year', 'state_n']].copy(), dm_cols)

    y = df_dm[[yvar]].values.astype(float)
    X = df_dm[xvars].values.astype(float)

    res = sm.OLS(y, X).fit()

    state_codes = pd.Categorical(df['state_n']).codes
    try:
        cov_cl = cov_cluster(res, state_codes)
        se_cl  = np.sqrt(np.diag(cov_cl))
    except Exception:
        se_cl = res.bse

    idx  = xvars.index('Linter_bra')
    beta = float(res.params[idx])
    se   = float(se_cl[idx])

    return {
        'label'        : label,
        'model'        : 'OLS',
        'beta'         : beta,
        'se'           : se,
        'spatial_param': np.nan,
        'param_name'   : '',
        'n_counties'   : df['fips5'].nunique(),
        'n_obs'        : len(y),
        'resid'        : res.resid,
        'year_arr'     : df['year'].values,
    }


def run_sem(df, w, yvar, xvars, label=''):
    y   = df[[yvar]].values.astype(float)
    X   = df[xvars].values.astype(float)
    sem = spreg.Panel_FE_Error(y=y, x=X, w=w, name_y=yvar, name_x=xvars)

    idx  = xvars.index('Linter_bra')
    beta = float(sem.betas[idx, 0])
    se   = float(np.sqrt(sem.vm[idx, idx]))
    lam  = float(sem.betas[-1, 0])

    return {
        'label'        : label,
        'model'        : 'SEM',
        'beta'         : beta,
        'se'           : se,
        'spatial_param': lam,
        'param_name'   : 'lam',
        'n_counties'   : w.n,
        'n_obs'        : len(y),
    }


def run_sar(df, w, yvar, xvars, label=''):
    y   = df[[yvar]].values.astype(float)
    X   = df[xvars].values.astype(float)
    sar = spreg.Panel_FE_Lag(y=y, x=X, w=w, name_y=yvar, name_x=xvars)

    idx  = xvars.index('Linter_bra')
    beta = float(sar.betas[idx, 0])
    se   = float(np.sqrt(sar.vm[idx, idx]))
    rho  = float(sar.betas[-1, 0])

    return {
        'label'        : label,
        'model'        : 'SAR',
        'beta'         : beta,
        'se'           : se,
        'spatial_param': rho,
        'param_name'   : 'rho',
        'n_counties'   : w.n,
        'n_obs'        : len(y),
    }


def run(output_dir=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    df_raw = pd.read_csv(PANEL_PATH, dtype={'fips5': str})
    df_raw['year'] = df_raw['year'].astype(int)

    gdf_all = gpd.read_file(GPKG_PATH, engine='pyogrio')
    counties_geo = (
        gdf_all[gdf_all['geometry'].notna()]
        .drop_duplicates(subset='fips5')[['fips5', 'geometry']]
        .reset_index(drop=True)
    )

    # ── 2. Build balanced samples + spatial weights ───────────────────────────
    df_full, w_full = prepare_sample(
        df_raw, counties_geo, YVAR, XVARS,
        border_filter=None, label='Full sample')

    df_nb, w_nb = prepare_sample(
        df_raw, counties_geo, YVAR, XVARS,
        border_filter=0, label='Non-border (border=0)')

    # ── 3. OLS ────────────────────────────────────────────────────────────────
    ols_full = run_ols(df_full, YVAR, XVARS, 'OLS -- full')
    ols_nb   = run_ols(df_nb,   YVAR, XVARS, 'OLS -- non-border')

    # ── 4. SEM ────────────────────────────────────────────────────────────────
    sem_full = run_sem(df_full, w_full, YVAR, XVARS, 'SEM -- full')
    sem_nb   = run_sem(df_nb,   w_nb,   YVAR, XVARS, 'SEM -- non-border')

    # ── 5. SAR ────────────────────────────────────────────────────────────────
    sar_full = run_sar(df_full, w_full, YVAR, XVARS, 'SAR -- full')
    sar_nb   = run_sar(df_nb,   w_nb,   YVAR, XVARS, 'SAR -- non-border')

    # ── 6. Moran's I on OLS residuals ────────────────────────────────────────
    resid_full = ols_full['resid']
    year_arr   = ols_full['year_arr']

    moran_rows = []
    for yr in YEARS:
        mask = year_arr == yr
        r_yr = resid_full[mask]
        n_yr = int(mask.sum())

        if n_yr != w_full.n:
            continue

        mi  = esda.Moran(r_yr, w_full)
        sig = ('***' if mi.p_norm < 0.001 else
               '**'  if mi.p_norm < 0.01  else
               '*'   if mi.p_norm < 0.05  else '')
        moran_rows.append(dict(
            year=yr, n=n_yr,
            moran_i=mi.I, expected_i=mi.EI, z_norm=mi.z_norm,
            p_norm=mi.p_norm, significant=sig,
        ))

    # ── 7. Save outputs ───────────────────────────────────────────────────────
    if output_dir is not None:
        main_rows = []
        for r in [ols_full, sem_full, sar_full, ols_nb, sem_nb, sar_nb]:
            main_rows.append(dict(
                model=r['model'],
                label=r['label'],
                beta=r['beta'],
                se=r['se'],
                spatial_param=r['spatial_param'],
                param_name=r['param_name'],
                n_counties=r['n_counties'],
                n_obs=r['n_obs'],
            ))
        pd.DataFrame(main_rows).to_csv(
            output_dir / "spatial_hpi_main_results.csv", index=False)

        pd.DataFrame(moran_rows).to_csv(
            output_dir / "spatial_hpi_morans_i.csv", index=False)

    return {
        'ols_full': ols_full, 'ols_nb': ols_nb,
        'sem_full': sem_full, 'sem_nb': sem_nb,
        'sar_full': sar_full, 'sar_nb': sar_nb,
        'moran_rows': moran_rows,
    }


if __name__ == '__main__':
    run(Path(__file__).parent.parent / 'output')
