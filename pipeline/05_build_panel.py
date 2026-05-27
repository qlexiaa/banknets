"""
Build the estimation-ready balanced panel.

Filters merged_panel.gpkg to the counties in W_geo_queen.gal,
drops the 1994 base year, enforces the exact county sort order
that aligns with the weights matrix rows, and validates key variables.

Reads  : data/merged_panel.gpkg
         data/county_order_Wgeo.csv
Writes : data/estimation_panel.csv
"""
from pathlib import Path
import pandas as pd
import geopandas as gpd

ROOT       = Path(__file__).parent.parent
GPKG_PATH  = ROOT / "data" / "merged_panel.gpkg"
ORDER_PATH = ROOT / "data" / "county_order_Wgeo.csv"
OUT_PATH   = ROOT / "data" / "estimation_panel.csv"

KEY_COLS = ['Linter_bra', 'Linter_ela', 'Dl_hpi', 'LDl_hpi', 'elasticity', 'border']

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

gdf   = gpd.read_file(GPKG_PATH, engine='pyogrio')
panel = pd.DataFrame(gdf.drop(columns='geometry'))
print(f"Full panel: {panel.shape}  ({panel['fips5'].nunique()} unique counties)")

order     = pd.read_csv(ORDER_PATH, dtype={'fips5': str})
w_fips    = set(order['fips5'].astype(str))
print(f"W_geo counties: {len(order)}")

# ── 2. Filter to W_geo counties ───────────────────────────────────────────────
panel = panel[panel['fips5'].isin(w_fips)].copy()
print(f"\nAfter filter: {len(panel):,} rows, {panel['fips5'].nunique()} counties")

# ── 3. Drop base year 1994 ────────────────────────────────────────────────────
panel['year'] = panel['year'].astype(float).astype(int)
panel = panel[panel['year'] >= 1995].copy()
print(f"Years: {sorted(panel['year'].unique())}  ({panel['year'].nunique()} years)")
print(f"Rows after dropping 1994: {len(panel):,}")

# ── 4. Sort to match W_geo row order ─────────────────────────────────────────
fips_to_row  = order.set_index('fips5')['row_index'].to_dict()
panel['_wr'] = panel['fips5'].map(fips_to_row)
panel        = panel.sort_values(['year', '_wr']).drop(columns='_wr').reset_index(drop=True)

# ── 5. Shape check ────────────────────────────────────────────────────────────
n_co   = panel['fips5'].nunique()
n_yr   = panel['year'].nunique()
n_rows = len(panel)
exp    = len(order) * 11
status = "PASS" if n_rows == exp else "WARNING"
print(f"\n{status}: {n_co} counties x {n_yr} years = {n_rows:,} rows  (expected {exp:,})")
if n_rows != exp:
    counts = panel.groupby('fips5')['year'].count()
    print(f"Counties with != 11 year obs:\n{counts[counts != 11]}")

# ── 6. Missing value check ────────────────────────────────────────────────────
print("\nMissing values in key columns:")
for col in KEY_COLS:
    if col not in panel.columns:
        print(f"  {col:<15} MISSING COLUMN")
        continue
    n_null = panel[col].isna().sum()
    tag    = "ok" if n_null == 0 else f"WARNING: {n_null} nulls ({100*n_null/len(panel):.1f}%)"
    print(f"  {col:<15} {tag}")

# ── 7. Save ───────────────────────────────────────────────────────────────────
panel.to_csv(OUT_PATH, index=False)
print(f"\nSaved -> {OUT_PATH}  ({len(panel):,} rows x {len(panel.columns)} cols)")
