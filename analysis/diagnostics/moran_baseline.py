"""
Spatial autocorrelation diagnostics for credit-growth residuals.

Builds residuals from the baseline credit-growth reduced form, then computes
Moran's I under queen contiguity at the county and state levels.

Output
------
  output/diagnostics/moran_i_results.csv
  output/diagnostics/moran_i_by_year.png
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

import geopandas as gpd
import libpysal
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.formula.api as smf
from esda.moran import Moran

sys.path.insert(0, str(Path(__file__).parents[1]))
from panel_data import CREDIT_CONTROLS  # noqa: E402

ROOT = Path(__file__).parents[2]
DTA_DIR = ROOT / "Replication" / "20121416_1data" / "data"
DEFAULT_OUTPUT_DIR = ROOT / "output"
COUNTY_SHAPE_DIR = ROOT / "data" / "counties_shapefile"
STATE_SHAPE_DIR = ROOT / "data" / "states_shapefile"

EXCLUDE = ["02", "15", "72", "60", "66", "69", "78"]
MORAN_COLUMNS = [
    "year", "moran_I", "expected_I", "z_score", "p_value",
    "significant", "n_units",
]


def empty_moran_results():
    return pd.DataFrame(columns=MORAN_COLUMNS)


def cached_shapefile_sources(cache_dir, zip_name):
    """Return extracted shapefiles or a cached zip for one TIGER geography."""
    if not cache_dir.exists():
        return []
    sources = sorted(cache_dir.glob("*.shp"))
    zip_path = cache_dir / zip_name
    if zip_path.exists():
        sources.append(zip_path)
    return sources


def load_tiger_shapefile(label, url, cache_dir, zip_name):
    """Load cached TIGER geometry first; download only when no cache exists."""
    cached = cached_shapefile_sources(cache_dir, zip_name)
    for source in cached:
        try:
            print(f"  Loading cached {label} shapefile from {source}")
            return gpd.read_file(source)
        except Exception as exc:
            print(f"  [WARN] Cached {label} shapefile unreadable ({source}): {exc}")

    if cached:
        print(f"  [WARN] Skipping {label} Moran weights: cached geography failed to load.")
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / zip_name
    tmp_path = zip_path.with_suffix(zip_path.suffix + ".tmp")
    print(f"  Downloading {label} shapefile from {url} ...")
    try:
        import urllib.request
        urllib.request.urlretrieve(url, tmp_path)
        tmp_path.replace(zip_path)
        return gpd.read_file(zip_path)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        print(f"  [WARN] Could not load {label} geography: {exc}")
        print(f"  [WARN] Skipping {label} Moran weights.")
        return None


def subset_weights(w, present_mask):
    """Subset a PySAL weight object and row-standardize the remaining links."""
    present_idx = np.where(present_mask)[0]
    idx_map = {old: new for new, old in enumerate(present_idx)}
    neighbors = {}
    weights = {}

    for new_i, old_i in enumerate(present_idx):
        nbrs = [idx_map[n] for n in w.neighbors[old_i] if n in idx_map]
        neighbors[new_i] = nbrs
        weights[new_i] = [1.0] * len(nbrs)

    w_sub = libpysal.weights.W(neighbors, weights, silence_warnings=True)
    w_sub.transform = "r"
    return w_sub


def moran_by_year(resid_df, id_col, order, w_full, level_label):
    """Compute Moran's I year by year for one spatial level."""
    results = []

    for year in sorted(resid_df["year"].unique()):
        yr = resid_df[resid_df["year"] == year].copy()
        yr[id_col] = yr[id_col].astype(int)
        present_mask = np.isin(order, yr[id_col].values)
        n_present = int(present_mask.sum())

        if n_present < w_full.n * 0.7:
            print(
                f"  [{level_label}] {int(year)}: skipped "
                f"(too few: {n_present}/{w_full.n})"
            )
            continue

        w_sub = subset_weights(w_full, present_mask)
        sub_order = order[present_mask]
        y = (
            yr.set_index(id_col)
            .reindex(sub_order)["residual"]
            .values.astype(float)
        )

        mi = Moran(y, w_sub, permutations=999)
        results.append({
            "year": int(year),
            "moran_I": mi.I,
            "expected_I": mi.EI,
            "z_score": mi.z_sim,
            "p_value": mi.p_sim,
            "significant": mi.p_sim < 0.05,
            "n_units": n_present,
        })

        sig = (
            "***" if mi.p_sim < 0.01
            else "**" if mi.p_sim < 0.05
            else "*" if mi.p_sim < 0.10
            else ""
        )
        print(
            f"  [{level_label}] {int(year)}: "
            f"I={mi.I:+.4f}  z={mi.z_sim:+.2f}  "
            f"p={mi.p_sim:.3f}  n={n_present} {sig}"
        )

    return pd.DataFrame(results)


def plot_moran(ax_bar, ax_z, df, title_prefix):
    """Plot Moran's I and permutation z-scores for one result set."""
    colors = ["#d62728" if s else "#aec7e8" for s in df["significant"]]
    ax_bar.bar(df["year"], df["moran_I"], color=colors, edgecolor="white", width=0.6)
    ax_bar.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_bar.axhline(df["expected_I"].mean(), color="grey", linewidth=0.8, linestyle=":")
    ax_bar.set_title(f"{title_prefix} -- Moran's I by Year")
    ax_bar.set_xlabel("Year")
    ax_bar.set_ylabel("Moran's I")
    ax_bar.set_xticks(df["year"])
    ax_bar.tick_params(axis="x", rotation=45)
    ax_bar.legend(
        handles=[
            mpatches.Patch(color="#d62728", label="p < 0.05"),
            mpatches.Patch(color="#aec7e8", label="p >= 0.05"),
        ],
        loc="upper right",
    )

    ax_z.plot(df["year"], df["z_score"], marker="o", color="#1f77b4",
              linewidth=2, markersize=6)
    ax_z.axhline(1.96, color="#d62728", linestyle="--", linewidth=1,
                 label="z = +/-1.96")
    ax_z.axhline(-1.96, color="#d62728", linestyle="--", linewidth=1)
    ax_z.axhline(0, color="black", linewidth=0.6)
    ax_z.set_title(f"{title_prefix} -- Z-score by Year")
    ax_z.set_xlabel("Year")
    ax_z.set_ylabel("Z-score (permutation)")
    ax_z.set_xticks(df["year"])
    ax_z.tick_params(axis="x", rotation=45)
    ax_z.legend()


def run(output_dir=None):
    out_dir = Path(output_dir or DEFAULT_OUTPUT_DIR) / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Loading data")
    print("=" * 60)
    hmda, _ = pyreadstat.read_dta(DTA_DIR / "hmda.dta")
    controls, _ = pyreadstat.read_dta(DTA_DIR / "hp_dereg_controls.dta")
    df = hmda.merge(controls, on=["county", "year"])
    print(
        f"Merged: {len(df):,} rows, {df['county'].nunique()} counties, "
        f"{df['year'].nunique()} years"
    )

    print("\n" + "=" * 60)
    print("STEP 2: Baseline credit-growth regression")
    print("=" * 60)
    required_cols = ["Dl_nloans_b", "Linter_bra"] + CREDIT_CONTROLS + ["county", "year"]
    df_clean = df.dropna(subset=required_cols).copy()
    df_clean["state"] = (
        df_clean["county"].astype(str).str.zfill(5).str[:2].astype(int)
    )

    formula = (
        "Dl_nloans_b ~ Linter_bra + "
        + " + ".join(CREDIT_CONTROLS)
        + " + C(county) + C(year)"
    )
    model = smf.ols(formula, data=df_clean).fit(
        cov_type="cluster",
        cov_kwds={"groups": df_clean["state"]},
    )

    print(
        f"Deregulation coef : {model.params['Linter_bra']:.4f}  "
        f"SE={model.bse['Linter_bra']:.4f}  "
        f"p={model.pvalues['Linter_bra']:.4f}"
    )
    print("F&I Table 2 target: ~0.028  SE ~0.010")

    df_clean["residual"] = model.resid.values
    residuals_df = df_clean[["county", "year", "state", "residual"]].copy()

    print("\n" + "=" * 60)
    print("STEP 3: Building spatial weights")
    print("=" * 60)

    county_shp_url = (
        "https://www2.census.gov/geo/tiger/GENZ2020/shp/"
        "cb_2020_us_county_20m.zip"
    )
    counties = load_tiger_shapefile(
        "county", county_shp_url, COUNTY_SHAPE_DIR, "cb_2020_us_county_20m.zip"
    )
    if counties is not None:
        counties["fips"] = counties["GEOID"].astype(int)
        counties = counties[~counties["STATEFP"].isin(EXCLUDE)]

    state_shp_url = (
        "https://www2.census.gov/geo/tiger/GENZ2020/shp/"
        "cb_2020_us_state_20m.zip"
    )
    states_shp = load_tiger_shapefile(
        "state", state_shp_url, STATE_SHAPE_DIR, "cb_2020_us_state_20m.zip"
    )
    if states_shp is not None:
        states_shp["statefp"] = states_shp["STATEFP"].astype(int)
        states_shp = states_shp[~states_shp["STATEFP"].isin(EXCLUDE)]

    w_county = None
    county_order = None
    if counties is not None:
        sample_fips = set(residuals_df["county"].astype(int).unique())
        counties_sample = (
            counties[counties["fips"].isin(sample_fips)]
            .sort_values("fips")
            .reset_index(drop=True)
        )
        if counties_sample.empty:
            print("  [WARN] No sample counties found in county shapefile; skipping county Moran.")
        else:
            w_county = libpysal.weights.Queen.from_dataframe(counties_sample)
            w_county.transform = "r"
            county_order = counties_sample["fips"].values
            print(
                f"County weights: {w_county.n} units, "
                f"avg {w_county.mean_neighbors:.1f} neighbors"
            )

    w_state = None
    state_order = None
    if states_shp is not None:
        sample_states = set(residuals_df["state"].astype(int).unique())
        states_sample = (
            states_shp[states_shp["statefp"].isin(sample_states)]
            .sort_values("statefp")
            .reset_index(drop=True)
        )
        if states_sample.empty:
            print("  [WARN] No sample states found in state shapefile; skipping state Moran.")
        else:
            w_state = libpysal.weights.Queen.from_dataframe(states_sample)
            w_state.transform = "r"
            state_order = states_sample["statefp"].values
            print(
                f"State weights : {w_state.n} units, "
                f"avg {w_state.mean_neighbors:.1f} neighbors"
            )

    print("\n" + "=" * 60)
    print("STEP 4: Moran's I -- credit, county level")
    print("=" * 60)
    if w_county is None:
        print("  [WARN] County geography unavailable; skipping county Moran computation.")
        res_credit_county = empty_moran_results()
    else:
        res_credit_county = moran_by_year(
            residuals_df, "county", county_order, w_county, "county"
        )

    print("\n" + "=" * 60)
    print("STEP 5: Moran's I -- credit, state level")
    print("=" * 60)
    if w_state is None:
        print("  [WARN] State geography unavailable; skipping state Moran computation.")
        res_credit_state = empty_moran_results()
    else:
        state_resid = (
            residuals_df.groupby(["state", "year"])["residual"]
            .mean()
            .reset_index()
        )
        res_credit_state = moran_by_year(
            state_resid, "state", state_order, w_state, "state"
        )

    for name, res in [
        ("Credit-county", res_credit_county),
        ("Credit-state", res_credit_state),
    ]:
        if res.empty:
            print(f"\n{name}: skipped (no Moran results)")
        else:
            print(
                f"\n{name}: sig={res['significant'].sum()}/{len(res)}  "
                f"mean I={res['moran_I'].mean():.4f}"
            )

    if res_credit_county.empty and res_credit_state.empty:
        print("\n[WARN] No Moran results available; skipping plot.")
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Spatial Autocorrelation of Credit-Growth Residuals",
            fontsize=13,
            fontweight="bold",
        )
        if res_credit_county.empty:
            axes[0, 0].set_axis_off()
            axes[0, 1].set_axis_off()
            axes[0, 0].set_title("Credit -- County skipped")
        else:
            plot_moran(
                axes[0, 0], axes[0, 1], res_credit_county, "Credit -- County"
            )
        if res_credit_state.empty:
            axes[1, 0].set_axis_off()
            axes[1, 1].set_axis_off()
            axes[1, 0].set_title("Credit -- State skipped")
        else:
            plot_moran(
                axes[1, 0], axes[1, 1], res_credit_state, "Credit -- State"
            )
        plt.tight_layout()
        plt.savefig(out_dir / "moran_i_by_year.png", dpi=150, bbox_inches="tight")
        print(f"\nPlot saved -> {out_dir / 'moran_i_by_year.png'}")
        plt.close()

    for df_res, level in [
        (res_credit_county, "county"),
        (res_credit_state, "state"),
    ]:
        df_res["level"] = level
        df_res["outcome"] = "credit"

    combined = pd.concat([res_credit_county, res_credit_state], ignore_index=True)
    combined.to_csv(out_dir / "moran_i_results.csv", index=False)
    print(f"Results saved -> {out_dir / 'moran_i_results.csv'}  ({len(combined)} rows)")


if __name__ == "__main__":
    run(DEFAULT_OUTPUT_DIR)
