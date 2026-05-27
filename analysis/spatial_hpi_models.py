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


print("=" * 65)
print("SPATIAL HPI MODELS - W_geo QUEEN CONTIGUITY")
print("=" * 65)

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("\n[1] Loading data")

df_raw = pd.read_csv(PANEL_PATH, dtype={'fips5': str})
df_raw['year'] = df_raw['year'].astype(int)
print(f"    Panel   : {df_raw.shape}  "
      f"({df_raw['fips5'].nunique()} counties, {df_raw['year'].nunique()} years)")

gdf_all = gpd.read_file(GPKG_PATH, engine='pyogrio')
counties_geo = (
    gdf_all[gdf_all['geometry'].notna()]
    .drop_duplicates(subset='fips5')[['fips5', 'geometry']]
    .reset_index(drop=True)
)

# ── 2. Build balanced samples + spatial weights ───────────────────────────────
print("\n[2] Building balanced samples and spatial weights")


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

    print(f"    {label:<20}: {w.n:>4} counties  {len(df):>6} obs  "
          f"{len(islands)} island(s) dropped")
    return df, w


df_full, w_full = prepare_sample(
    df_raw, counties_geo, YVAR, XVARS,
    border_filter=None, label='Full sample')

df_nb, w_nb = prepare_sample(
    df_raw, counties_geo, YVAR, XVARS,
    border_filter=0, label='Non-border (border=0)')


# ── 3. Helpers ────────────────────────────────────────────────────────────────
def within_demean(df, cols, county_col='fips5', year_col='year'):
    """Two-way within-transformation (FWL for county + year FE)."""
    out = df.copy()
    for col in cols:
        gm = df[col].mean()
        cm = df.groupby(county_col)[col].transform('mean')
        ym = df.groupby(year_col)[col].transform('mean')
        out[col] = df[col] - cm - ym + gm
    return out


# ── 4. OLS ────────────────────────────────────────────────────────────────────
print("\n[3] OLS (within-demeaned, state-clustered SEs)")


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


ols_full = run_ols(df_full, YVAR, XVARS, 'OLS -- full')
ols_nb   = run_ols(df_nb,   YVAR, XVARS, 'OLS -- non-border')

for r in [ols_full, ols_nb]:
    print(f"    {r['label']:<25}  b={r['beta']:.4f}  SE={r['se']:.4f}  "
          f"N={r['n_obs']}  counties={r['n_counties']}")


# ── 5. SEM — Panel_FE_Error ───────────────────────────────────────────────────
print("\n[4] SEM -- Panel_FE_Error (GM estimator)")


def run_sem(df, w, yvar, xvars, label=''):
    y   = df[[yvar]].values.astype(float)
    X   = df[xvars].values.astype(float)
    sem = spreg.Panel_FE_Error(y=y, x=X, w=w, name_y=yvar, name_x=xvars)

    idx  = xvars.index('Linter_bra')
    beta = float(sem.betas[idx, 0])
    se   = float(np.sqrt(sem.vm[idx, idx]))
    lam  = float(sem.betas[-1, 0])

    print(f"    {label:<25}  b={beta:.4f}  SE={se:.4f}  lam={lam:.4f}  N={len(y)}")
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


sem_full = run_sem(df_full, w_full, YVAR, XVARS, 'SEM -- full')
sem_nb   = run_sem(df_nb,   w_nb,   YVAR, XVARS, 'SEM -- non-border')


# ── 6. SAR — Panel_FE_Lag ─────────────────────────────────────────────────────
print("\n[5] SAR -- Panel_FE_Lag (ML estimator)")


def run_sar(df, w, yvar, xvars, label=''):
    y   = df[[yvar]].values.astype(float)
    X   = df[xvars].values.astype(float)
    sar = spreg.Panel_FE_Lag(y=y, x=X, w=w, name_y=yvar, name_x=xvars)

    idx  = xvars.index('Linter_bra')
    beta = float(sar.betas[idx, 0])
    se   = float(np.sqrt(sar.vm[idx, idx]))
    rho  = float(sar.betas[-1, 0])

    print(f"    {label:<25}  b={beta:.4f}  SE={se:.4f}  rho={rho:.4f}  N={len(y)}")
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


sar_full = run_sar(df_full, w_full, YVAR, XVARS, 'SAR -- full')
sar_nb   = run_sar(df_nb,   w_nb,   YVAR, XVARS, 'SAR -- non-border')


# ── 7. Results table ──────────────────────────────────────────────────────────
print()
print("=" * 78)
print("RESULTS - HPI Reduced-Form Spatial Models (W_geo Queen Contiguity)")
print("=" * 78)
print(f"{'Model':<26}  {'b(Linter_bra)':>14}  {'SE':>8}  "
      f"{'Spatial':>12}  {'Counties':>9}  {'Obs':>7}")
print("-" * 78)

for r in [ols_full, sem_full, sar_full]:
    sp = f"{r['param_name']}={r['spatial_param']:.4f}" if r['param_name'] else '-'
    print(f"{r['label']:<26}  {r['beta']:>14.4f}  {r['se']:>8.4f}  "
          f"{sp:>12}  {r['n_counties']:>9}  {r['n_obs']:>7}")

print()
for r in [ols_nb, sem_nb, sar_nb]:
    sp = f"{r['param_name']}={r['spatial_param']:.4f}" if r['param_name'] else '-'
    print(f"{r['label']:<26}  {r['beta']:>14.4f}  {r['se']:>8.4f}  "
          f"{sp:>12}  {r['n_counties']:>9}  {r['n_obs']:>7}")

print("=" * 78)
print("Notes:")
print("  OLS  -- county + year FEs via within-demeaning; SEs clustered by state.")
print("  SEM  -- spreg.Panel_FE_Error; GM estimator; model-based SEs.")
print("  SAR  -- spreg.Panel_FE_Lag;   ML estimator; model-based SEs.")
print("  Non-border subsample: counties with border == 0.")


# ── 8. Moran's I on OLS residuals ────────────────────────────────────────────
print()
print("=" * 65)
print("MORAN'S I ON OLS RESIDUALS - Full Sample, by Year")
print("=" * 65)
print(f"{'Year':<6}  {'N':>5}  {'Moran I':>9}  {'E[I]':>7}  "
      f"{'z-score':>8}  {'p-value':>9}  {'sig':>4}")
print("-" * 58)

resid_full = ols_full['resid']
year_arr   = ols_full['year_arr']

for yr in YEARS:
    mask = year_arr == yr
    r_yr = resid_full[mask]
    n_yr = int(mask.sum())

    if n_yr != w_full.n:
        print(f"{yr:<6}  {n_yr:>5}  [skip -- {n_yr} residuals vs {w_full.n} W rows]")
        continue

    mi  = esda.Moran(r_yr, w_full)
    sig = ('***' if mi.p_norm < 0.001 else
           '**'  if mi.p_norm < 0.01  else
           '*'   if mi.p_norm < 0.05  else '')
    print(f"{yr:<6}  {n_yr:>5}  {mi.I:>9.4f}  {mi.EI:>7.4f}  "
          f"{mi.z_norm:>8.3f}  {mi.p_norm:>9.4f}  {sig:>4}")

print("-" * 58)
print("Significance: *** p<0.001  ** p<0.01  * p<0.05")
print("\nDone.")
