"""
Build queen contiguity spatial weights matrix (W_geo).

Reads  : data/merged_panel.gpkg
Writes : data/W_geo_queen.gal          binary queen contiguity, FIPS-labelled
         data/county_order_Wgeo.csv    row order of the matrix

Islands (counties with zero neighbours in the panel sample) are dropped.
Three counties with incomplete panel coverage are also excluded:
  05111 Scott County AR    (absent from panel entirely)
  08014 Broomfield CO      (only 2004-2005; city incorporated 2001)
  12086 Miami-Dade FL      (only 2000-2005; FIPS changed from Dade)
"""
from pathlib import Path
import pandas as pd
import geopandas as gpd
import libpysal

ROOT      = Path(__file__).parent.parent
GPKG_PATH = ROOT / "data" / "merged_panel.gpkg"
GAL_OUT   = ROOT / "data" / "W_geo_queen.gal"
ORDER_OUT = ROOT / "data" / "county_order_Wgeo.csv"

DROP_ISLANDS     = True
PANEL_INCOMPLETE = {'05111', '08014', '12086'}

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading merged_panel.gpkg")
print("=" * 60)

gdf = gpd.read_file(GPKG_PATH, engine='pyogrio')
print(f"Loaded: {len(gdf):,} rows, {gdf['fips5'].nunique()} unique counties")

gdf = gdf[gdf['geometry'].notna()].copy()
print(f"After dropping null-geometry rows: {len(gdf):,} rows")

# ── 2. One row per county ─────────────────────────────────────────────────────
counties = (gdf.drop_duplicates(subset='fips5')
              .sort_values('fips5')
              .reset_index(drop=True))
counties = counties[~counties['fips5'].isin(PANEL_INCOMPLETE)].reset_index(drop=True)
print(f"\nUnique counties (after removing incomplete-panel): {len(counties)}")

# ── 3. Queen contiguity ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Building queen contiguity matrix")
print("=" * 60)

w = libpysal.weights.Queen.from_dataframe(
    counties, ids='fips5', silence_warnings=True)

# ── 4. Drop islands ───────────────────────────────────────────────────────────
islands = [fips for fips, nbrs in w.neighbors.items() if len(nbrs) == 0]
print(f"Islands (0 neighbours): {len(islands)}")
if islands:
    print(f"  {sorted(islands)}")

if DROP_ISLANDS and islands:
    counties = counties[~counties['fips5'].isin(islands)].reset_index(drop=True)
    w = libpysal.weights.Queen.from_dataframe(
        counties, ids='fips5', silence_warnings=True)
    new_islands = [f for f, n in w.neighbors.items() if len(n) == 0]
    print(f"  Islands after rebuild: {len(new_islands)}")

# ── 5. Row-standardise ────────────────────────────────────────────────────────
w.transform = 'r'

nb_counts = pd.Series({k: len(v) for k, v in w.neighbors.items()})
print(f"\nFinal matrix: {w.n} counties  "
      f"avg={w.mean_neighbors:.2f}  "
      f"min={nb_counts.min()}  max={nb_counts.max()}")

# ── 6. Save ───────────────────────────────────────────────────────────────────
w.to_file(str(GAL_OUT))
print(f"\nSaved -> {GAL_OUT}")

order_df = pd.DataFrame({
    'row_index': range(w.n),
    'fips5':     w.id_order,
    'fips_int':  [int(f) for f in w.id_order],
})
order_df.to_csv(ORDER_OUT, index=False)
print(f"Saved -> {ORDER_OUT}  ({len(order_df)} rows)")
