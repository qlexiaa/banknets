"""
hub_map.py
==========
US county choropleth heatmaps using geopandas + matplotlib.

Two maps, credit outcome only (Full sample):

  (a) Bank-network strength centrality:
      Column sums of the UN-row-standardised time-averaged cosine W_bank.
      Captures how many other counties point to each county in the raw
      bank-network graph (in-degree weighted by cosine similarity).

  (b) Total transmission receiver values from hub_counties logic under W_geo:
      Off-diagonal column sums of S = (I - lambda*W_geo)^{-1}.
      Captures how much shock each county absorbs from the network.

Outputs
-------
  output/hub_map_bank_strength.png
  output/hub_map_geo_transmission.png

Requires geopandas and a counties shapefile.  Downloads the Census TIGER 2020
simplified counties shapefile on first run; caches to data/counties_shapefile/.
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import scipy.linalg
import scipy.sparse

sys.path.insert(0, str(Path(__file__).parents[1]))
from utils import row_standardize
from w_variants import load_w_geo, load_bank_variants
from panel_data import load_panel_with_credit

ROOT         = Path(__file__).parents[2]
COUNTY_PATH  = ROOT / "data" / "county_order_Wgeo.csv"
CREDIT_CSV   = ROOT / "output" / "panel_fe_credit_results.csv"
SHAPE_DIR    = ROOT / "data" / "counties_shapefile"
SHAPE_URL    = ("https://www2.census.gov/geo/tiger/GENZ2020/shp/"
                "cb_2020_us_county_20m.zip")


def _load_shapefile():
    """Load Census county shapefile; download and cache on first call."""
    import geopandas as gpd

    cached = list(SHAPE_DIR.glob("*.shp")) if SHAPE_DIR.exists() else []
    if cached:
        return gpd.read_file(cached[0])

    SHAPE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = SHAPE_DIR / "counties.zip"
    print(f"  Downloading Census county shapefile from {SHAPE_URL} ...")
    try:
        import urllib.request
        urllib.request.urlretrieve(SHAPE_URL, zip_path)
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(SHAPE_DIR)
        zip_path.unlink(missing_ok=True)
        shpfiles = list(SHAPE_DIR.glob("*.shp"))
        if not shpfiles:
            raise FileNotFoundError("No .shp file found after extraction")
        return gpd.read_file(shpfiles[0])
    except Exception as exc:
        print(f"  [WARN] Shapefile download failed: {exc}")
        return None


def _choropleth(gdf, fips_col, value_col, title, out_path,
                cmap="YlOrRd", label=None, vmin=None, vmax=None):
    """Draw a simple choropleth and save to out_path."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    # Exclude Alaska/Hawaii for cleaner CONUS map
    gdf_conus = gdf[~gdf[fips_col].str[:2].isin(["02", "15"])].copy()
    gdf_conus.plot(
        column=value_col, ax=ax,
        cmap=cmap, legend=True,
        legend_kwds={"label": label or value_col,
                     "orientation": "horizontal",
                     "fraction": 0.03, "pad": 0.04},
        missing_kwds={"color": "lightgrey", "label": "No data"},
        vmin=vmin, vmax=vmax,
        linewidth=0.1, edgecolor="0.6",
    )
    ax.set_axis_off()
    ax.set_title(title, fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def run(output_dir=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load matrices ─────────────────────────────────────────────────────────
    co_df        = pd.read_csv(COUNTY_PATH, dtype={"fips5": str})
    county_order = co_df["fips5"].str.zfill(5).tolist()

    W_geo_sp, _  = load_w_geo(county_order)
    bank_vars    = load_bank_variants(county_order, W_geo_all=W_geo_sp)
    W_bank_sp    = bank_vars["W_bank"]

    # ── Map (a): bank-network strength centrality (col sums, UN-standardised) ─
    print("Computing bank-network in-strength centrality ...")
    # Use the raw (not row-standardised) cosine W_bank
    # (bank_vars["W_bank"] is already row-standardised; re-load raw cosine)
    W_bank_raw_path = ROOT / "data" / "W_bank.npz"
    if W_bank_raw_path.exists():
        W_bank_raw = scipy.sparse.load_npz(str(W_bank_raw_path))
    else:
        W_bank_raw = W_bank_sp  # fallback to row-standardised
        print("  [NOTE] Raw W_bank.npz not found; using row-standardised W_bank")

    col_sums = np.array(W_bank_raw.sum(axis=0)).flatten()
    strength_df = pd.DataFrame({
        "fips5":    county_order,
        "in_strength": col_sums,
    })
    if output_dir is not None:
        strength_df.to_csv(output_dir / "bank_strength_centrality.csv", index=False)

    # ── Map (b): SEM total transmission receiver (full sample, W_geo) ─────────
    print("Computing SEM total transmission receiver values under W_geo ...")
    # Load lambda from panel_fe_credit_results.csv
    lam_geo = None
    if CREDIT_CSV.exists():
        cr   = pd.read_csv(CREDIT_CSV)
        mask = (cr["model"].str.contains("W_geo", regex=False) &
                cr["model"].str.contains("(full)", regex=False))
        rows = cr.loc[mask, "lam"]
        if len(rows) > 0:
            lam_geo = float(rows.iloc[0])
    if lam_geo is None:
        lam_geo = 0.18   # fallback
        print(f"  [NOTE] lam_geo not found in CSV; using fallback {lam_geo:.4f}")
    else:
        print(f"  lam_geo (full) = {lam_geo:.6f}")

    # Drop zero-row islands, invert
    W_sp = W_geo_sp.copy()
    rs   = np.array(W_sp.sum(axis=1)).flatten()
    keep = np.where(rs > 0)[0]
    W_sub = W_sp[keep, :][:, keep]
    co_sub = [county_order[i] for i in keep]

    W_dense = W_sub.toarray().astype(np.float64)
    N_sub   = W_dense.shape[0]
    print(f"  Inverting {N_sub}x{N_sub} matrix (lambda={lam_geo:.4f}) ...")
    S      = scipy.linalg.inv(np.eye(N_sub) - lam_geo * W_dense)
    S_off  = S.copy()
    np.fill_diagonal(S_off, 0.0)
    col_rx = S_off.sum(axis=0)  # incoming transmission for each county

    receiver_df = pd.DataFrame({
        "fips5":            co_sub,
        "total_rx":         col_rx,
    })

    # ── Shapefile for maps ────────────────────────────────────────────────────
    try:
        import geopandas as gpd
    except ImportError:
        print("  [WARN] geopandas not available -- skipping maps")
        return {"strength": strength_df, "receiver": receiver_df}

    gdf = _load_shapefile()
    if gdf is None:
        print("  [WARN] Shapefile unavailable -- skipping maps")
        return {"strength": strength_df, "receiver": receiver_df}

    # Standardise the FIPS column name
    fips_col = "GEOID" if "GEOID" in gdf.columns else "fips5"
    if fips_col == "GEOID":
        gdf["fips5"] = gdf["GEOID"].astype(str).str.zfill(5)
    gdf["fips5"] = gdf["fips5"].astype(str).str.zfill(5)

    # ── Map (a) ───────────────────────────────────────────────────────────────
    if output_dir is not None:
        gdf_a = gdf.merge(strength_df, on="fips5", how="left")
        # Cap at 99th percentile to avoid outlier distortion
        p99 = float(np.nanpercentile(gdf_a["in_strength"].dropna(), 99))
        gdf_a["in_strength_capped"] = gdf_a["in_strength"].clip(upper=p99)
        _choropleth(
            gdf_a, "fips5", "in_strength_capped",
            title="Bank-Network In-Strength Centrality\n"
                  "(column sums of un-standardised W_bank cosine, capped at 99th pct)",
            out_path=output_dir / "hub_map_bank_strength.png",
            cmap="YlOrRd",
            label="In-strength (capped at 99th pct)",
        )

    # ── Map (b) ───────────────────────────────────────────────────────────────
    if output_dir is not None:
        gdf_b = gdf.merge(receiver_df, on="fips5", how="left")
        p99b = float(np.nanpercentile(gdf_b["total_rx"].dropna(), 99))
        gdf_b["total_rx_capped"] = gdf_b["total_rx"].clip(upper=p99b)
        _choropleth(
            gdf_b, "fips5", "total_rx_capped",
            title="SEM Total Transmission Receiver Value under W_geo (Full Sample)\n"
                  r"Off-diagonal column sums of S = (I - $\lambda$ W_geo)$^{-1}$, capped at 99th pct",
            out_path=output_dir / "hub_map_geo_transmission.png",
            cmap="Blues",
            label="Total received transmission (capped at 99th pct)",
        )

    return {"strength": strength_df, "receiver": receiver_df}


if __name__ == "__main__":
    run(ROOT / "output")
