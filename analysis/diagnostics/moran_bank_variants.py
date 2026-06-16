"""
Bank-network Moran's I diagnostics for credit-growth residuals.

Builds residuals from the baseline credit-growth reduced form, then computes
county-level Moran's I under alternative W_bank specifications.

Outputs
-------
  output/diagnostics/moran_i_wbank_results.csv
  output/diagnostics/moran_i_wbank_summary.csv
  output/diagnostics/moran_i_wbank_by_year.png
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadstat
import scipy.sparse
import statsmodels.formula.api as smf
from esda.moran import Moran
from libpysal.weights import WSP

sys.path.insert(0, str(Path(__file__).parents[1]))
from panel_data import CREDIT_CONTROLS  # noqa: E402
from utils import gal_to_W, row_standardize  # noqa: E402


ROOT = Path(__file__).parents[2]
DTA_DIR = ROOT / "Replication" / "20121416_1data" / "data"
OUT_DIR = ROOT / "output" / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COUNTY_PATH = ROOT / "data" / "county_order_Wgeo.csv"
GAL_PATH = ROOT / "data" / "W_geo_queen.gal"
WBANK_BIN_PATH = ROOT / "data" / "W_bank_avg.npz"
WBANK_COUNT_PATH = ROOT / "data" / "W_bank_count_avg.npz"
WBANK_NONGEO_PATH = ROOT / "data" / "W_bank_nonGeo.npz"

K_VALUES = list(range(1, 21))
PERMUTATIONS = 999


def build_wbank_knn(W_sparse, k):
    """Keep each row's top-k positive weights and row-standardize."""
    W = W_sparse.toarray().astype(np.float64)
    np.fill_diagonal(W, 0.0)

    W_knn = np.zeros_like(W)
    for i in range(W.shape[0]):
        row = W[i]
        nz = np.count_nonzero(row)
        if nz == 0:
            continue
        if nz <= k:
            W_knn[i] = row
        else:
            top_k = np.argpartition(row, -k)[-k:]
            W_knn[i, top_k] = row[top_k]

    np.fill_diagonal(W_knn, 0.0)
    return row_standardize(scipy.sparse.csr_matrix(W_knn))


def matrix_stats(W_sparse):
    """Return link count and density diagnostics for an NxN W matrix."""
    W = W_sparse.tocsr()
    n = W.shape[0]
    n_links = int(W.nnz - np.count_nonzero(W.diagonal()))
    possible_links = n * (n - 1)
    density = n_links / possible_links
    return {
        "n_links": n_links,
        "possible_links": possible_links,
        "density": density,
        "sparsity": 1.0 - density,
        "avg_nbrs": n_links / n,
    }


def trim_islands(W_sparse, labels):
    """Iteratively drop rows that have no outgoing neighbors after subsetting."""
    W = W_sparse.tocsr()
    labels = np.asarray(labels)

    while True:
        row_sums = np.array(W.sum(axis=1)).ravel()
        keep = row_sums > 0
        if keep.all():
            return row_standardize(W), labels
        if keep.sum() < 3:
            return row_standardize(W[keep, :][:, keep]), labels[keep]
        W = W[keep, :][:, keep].tocsr()
        labels = labels[keep]


def load_residuals():
    """Build baseline credit-growth residuals."""
    print("=" * 60)
    print("STEP 1: Loading F&I data")
    print("=" * 60)
    hmda, _ = pyreadstat.read_dta(DTA_DIR / "hmda.dta")
    controls, _ = pyreadstat.read_dta(DTA_DIR / "hp_dereg_controls.dta")
    df = hmda.merge(controls, on=["county", "year"])
    print(f"Merged: {len(df):,} rows, {df['county'].nunique()} counties")

    print("\n" + "=" * 60)
    print("STEP 2: Credit residuals")
    print("=" * 60)
    credit_required = ["Dl_nloans_b", "Linter_bra"] + CREDIT_CONTROLS + ["county", "year"]
    df_credit = df.dropna(subset=credit_required).copy()
    df_credit["state"] = (
        df_credit["county"].astype(str).str.zfill(5).str[:2].astype(int)
    )
    credit_formula = (
        "Dl_nloans_b ~ Linter_bra + "
        + " + ".join(CREDIT_CONTROLS)
        + " + C(county) + C(year)"
    )
    credit_model = smf.ols(credit_formula, data=df_credit).fit(
        cov_type="cluster",
        cov_kwds={"groups": df_credit["state"]},
    )
    print(
        f"Linter_bra: {credit_model.params['Linter_bra']:.4f}  "
        f"SE={credit_model.bse['Linter_bra']:.4f}"
    )

    df_credit["residual"] = credit_model.resid.values
    credit_resid = df_credit[["county", "year", "state", "residual"]].copy()
    credit_resid["outcome"] = "credit"
    return credit_resid


def load_w_matrices():
    """Load W_geo baseline and W_bank matrices, then build KNN variants."""
    print("\n" + "=" * 60)
    print("STEP 3: Loading W specifications")
    print("=" * 60)
    county_order = (
        pd.read_csv(COUNTY_PATH, dtype={"fips5": str})["fips5"]
        .str.zfill(5)
        .tolist()
    )

    W_geo, _ = gal_to_W(GAL_PATH, county_order)
    W_bin = row_standardize(scipy.sparse.load_npz(WBANK_BIN_PATH))
    W_count = row_standardize(scipy.sparse.load_npz(WBANK_COUNT_PATH))
    W_nongeo = row_standardize(scipy.sparse.load_npz(WBANK_NONGEO_PATH))

    matrices = [
        ("W_geo", W_geo),
        ("W_bank_bin", W_bin),
        ("W_bank_count", W_count),
        ("W_bank_nonGeo", W_nongeo),
    ]
    for k in K_VALUES:
        matrices.append((f"W_bank_knn_{k}", build_wbank_knn(W_bin, k)))

    for label, W in matrices:
        stats = matrix_stats(W)
        print(
            f"{label:<15} density={stats['density']:.5f}  "
            f"sparsity={stats['sparsity']:.5f}  "
            f"avg_nbrs={stats['avg_nbrs']:.2f}"
        )

    return county_order, matrices


def moran_by_year(resid_df, county_order, W_sparse, w_label, w_stats):
    """Compute Moran's I for one W matrix across years."""
    rows = []
    county_to_idx = {c: i for i, c in enumerate(county_order)}

    for year in sorted(resid_df["year"].unique()):
        yr = resid_df[resid_df["year"] == year].copy()
        yr["fips5"] = yr["county"].astype(int).astype(str).str.zfill(5)

        present = [c for c in county_order if c in set(yr["fips5"])]
        if len(present) < len(county_order) * 0.7:
            print(f"  [{w_label}] {int(year)} skipped: {len(present)} counties")
            continue

        idx = np.array([county_to_idx[c] for c in present])
        kept_counties = np.array(present)
        W_sub, kept_counties = trim_islands(W_sparse[idx, :][:, idx], kept_counties)

        if len(kept_counties) < 3:
            print(f"  [{w_label}] {int(year)} skipped: too few non-islands")
            continue

        w_pysal = WSP(W_sub.tocsr()).to_W(silence_warnings=True)
        y = (
            yr.set_index("fips5")
            .reindex(kept_counties)["residual"]
            .values.astype(float)
        )
        if np.nanstd(y) == 0:
            print(f"  [{w_label}] {int(year)} skipped: zero residual variance")
            continue

        mi = Moran(y, w_pysal, permutations=PERMUTATIONS)
        sig = (
            "***" if mi.p_sim < 0.01
            else "**" if mi.p_sim < 0.05
            else "*" if mi.p_sim < 0.10
            else ""
        )
        print(
            f"  [{w_label}] {int(year)}: "
            f"I={mi.I:+.4f}  z={mi.z_sim:+.2f}  "
            f"p={mi.p_sim:.3f}  n={len(kept_counties)} {sig}"
        )

        rows.append({
            "outcome": "credit",
            "level": "county",
            "year": int(year),
            "w_matrix": w_label,
            "moran_I": mi.I,
            "expected_I": mi.EI,
            "z_score": mi.z_sim,
            "p_value": mi.p_sim,
            "significant": mi.p_sim < 0.05,
            "n_units": int(len(kept_counties)),
            **w_stats,
        })

    return rows


def plot_selected(results, out_dir):
    """Plot selected W specifications by year."""
    selected = [
        "W_geo",
        "W_bank_bin",
        "W_bank_count",
        "W_bank_nonGeo",
        "W_bank_knn_1",
        "W_bank_knn_2",
        "W_bank_knn_5",
        "W_bank_knn_10",
        "W_bank_knn_20",
    ]
    df = results[results["w_matrix"].isin(selected)].copy()

    fig, ax = plt.subplots(figsize=(12, 6))
    for w_label in selected:
        s = df[df["w_matrix"] == w_label]
        if s.empty:
            continue
        lw = 2.5 if w_label == "W_geo" else 1.5
        ms = 6 if w_label == "W_geo" else 4
        ax.plot(s["year"], s["moran_I"], marker="o", linewidth=lw,
                markersize=ms, label=w_label)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Credit residual Moran's I: W_geo vs W_bank specs")
    ax.set_xlabel("Year")
    ax.set_ylabel("Moran's I")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.22))

    plt.tight_layout()
    path = out_dir / "moran_i_wbank_by_year.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved -> {path}")


def run(output_dir=None):
    """Run all bank-W Moran diagnostics and save results."""
    if output_dir is not None:
        out = Path(output_dir) / "diagnostics"
    else:
        out = OUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    residuals = load_residuals()
    county_order, matrices = load_w_matrices()

    print("\n" + "=" * 60)
    print("STEP 4: Moran's I by year and W specification")
    print("=" * 60)
    all_rows = []
    for w_label, W in matrices:
        print(f"  {w_label}:", flush=True)
        all_rows.extend(
            moran_by_year(residuals, county_order, W, w_label, matrix_stats(W))
        )

    results = pd.DataFrame(all_rows)
    base_cols = [
        "outcome", "level", "year", "w_matrix",
        "moran_I", "expected_I", "z_score", "p_value",
        "significant", "n_units",
    ]
    extra_cols = [c for c in results.columns if c not in base_cols]
    results = results[base_cols + extra_cols]

    results.to_csv(out / "moran_i_wbank_results.csv", index=False)
    print(f"\nResults saved -> {out / 'moran_i_wbank_results.csv'}")

    summary = (
        results.groupby(["outcome", "w_matrix"], as_index=False)
        .agg(
            mean_moran_I=("moran_I", "mean"),
            median_moran_I=("moran_I", "median"),
            min_p_value=("p_value", "min"),
            significant_years=("significant", "sum"),
            n_years=("year", "count"),
            density=("density", "first"),
            sparsity=("sparsity", "first"),
            avg_nbrs=("avg_nbrs", "first"),
        )
    )
    summary.to_csv(out / "moran_i_wbank_summary.csv", index=False)
    print(f"Summary saved -> {out / 'moran_i_wbank_summary.csv'}")

    plot_selected(results, out)

    print("\nMean Moran's I, sorted descending:")
    top = summary.sort_values("mean_moran_I", ascending=False).head(8)
    print(
        top[["w_matrix", "mean_moran_I", "significant_years", "density", "avg_nbrs"]]
        .to_string(index=False)
    )

    return results, summary


if __name__ == "__main__":
    run()
