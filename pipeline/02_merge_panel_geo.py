"""
Merge hp_dereg_controls panel with TIGER 2020 county geometry.

Reads  : Replication/20121416_1data/data/hp_dereg_controls.dta
         TIGER 2020 county shapefile (downloaded from Census)
Writes : data/merged_panel.gpkg
"""
from pathlib import Path
import pandas as pd
import geopandas as gpd

ROOT     = Path(__file__).parent.parent
DTA_PATH = ROOT / "Replication" / "20121416_1data" / "data" / "hp_dereg_controls.dta"
OUT_PATH = ROOT / "data" / "merged_panel.gpkg"

EXCLUDE_STATEFP = ['02', '15', '60', '66', '69', '72', '78']

# ── 1. Load panel ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading hp_dereg_controls.dta")
print("=" * 60)

panel = pd.read_stata(DTA_PATH)
print(f"Shape: {panel.shape}  ({panel.shape[0]:,} rows x {panel.shape[1]} cols)")

# ── 2. Load TIGER 2020 county shapefile ───────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Loading TIGER 2020 county shapefile")
print("=" * 60)

url = "https://www2.census.gov/geo/tiger/TIGER2020/COUNTY/tl_2020_us_county.zip"
print(f"Downloading from {url} ...")
counties = gpd.read_file(url)
counties = counties[~counties['STATEFP'].isin(EXCLUDE_STATEFP)].copy()
print(f"Counties (contiguous 48 states): {len(counties)}")

# ── 3. Identify FIPS column + zero-pad ───────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Identifying county FIPS column")
print("=" * 60)

candidates = [c for c in panel.columns
              if any(k in c.lower() for k in ('county', 'fips', 'geoid'))]
print(f"Name-based candidates: {candidates}")
for c in candidates:
    try:
        vals = pd.to_numeric(panel[c], errors='coerce').dropna()
        print(f"  {c}: min={vals.min():.0f}  max={vals.max():.0f}  "
              f"nunique={panel[c].nunique()}")
    except Exception:
        pass

FIPS_COL = 'county'
print(f"\nUsing FIPS column: '{FIPS_COL}'")

panel['fips5'] = (panel[FIPS_COL].astype(float).astype(int)
                  .astype(str).str.zfill(5))
counties['fips5'] = counties['GEOID'].astype(str).str.zfill(5)

# ── 4. Merge geometry ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Merging geometry")
print("=" * 60)

shp_slim = counties[['fips5', 'geometry']].copy()
merged   = panel.merge(shp_slim, on='fips5', how='left')
merged   = gpd.GeoDataFrame(merged, geometry='geometry', crs=counties.crs)

panel_fips    = set(panel['fips5'].unique())
shp_fips      = set(counties['fips5'].unique())
print(f"Panel rows        : {len(merged):,}")
print(f"Missing geometry  : {merged['geometry'].isna().sum()}")
print(f"Matched FIPS      : {len(panel_fips & shp_fips)}")
unmatched = panel_fips - shp_fips
if unmatched:
    print(f"Panel FIPS not in shapefile ({len(unmatched)}): {sorted(unmatched)}")

# ── 5. Save ───────────────────────────────────────────────────────────────────
merged.to_file(OUT_PATH, driver='GPKG', engine='pyogrio')
print(f"\nSaved -> {OUT_PATH}  ({len(merged):,} rows)")
