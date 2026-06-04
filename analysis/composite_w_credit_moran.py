"""
composite_w_credit_moran.py -- Moran's I under credit-DV composite W optima
============================================================================
Computes Moran's I on the Favara-Imbs credit reduced-form residuals under the
optimal composite W matrices from rq1_composite_w_credit.py.

For each W_alt x sample optimum:

  W_combo(alpha*) = alpha* W_geo + (1 - alpha*) W_alt

The script also reports the endpoint matrices W_geo and W_alt for comparison.

Outputs:
  output/diagnostics/moran_i_composite_credit_results.csv
  output/diagnostics/moran_i_composite_credit_summary.csv
  output/diagnostics/moran_i_composite_credit_by_year.png
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadstat
import scipy.sparse
import statsmodels.formula.api as smf
from esda.moran import Moran
from libpysal.weights import WSP

sys.path.insert(0, str(Path(__file__).parent))
from rq1_composite_w_credit import build_combo  # noqa: E402
from utils import gal_to_W, row_standardize  # noqa: E402


ROOT = Path(__file__).parent.parent
DTA_DIR = ROOT / "Replication" / "20121416_1data" / "data"
OUT_DIR = ROOT / "output" / "diagnostics"
COUNTY_PATH = ROOT / "data" / "county_order_Wgeo.csv"
PANEL_PATH = ROOT / "data" / "estimation_panel.csv"
GAL_PATH = ROOT / "data" / "W_geo_queen.gal"
WBANK_BIN_PATH = ROOT / "data" / "W_bank_avg.npz"
WBANK_CNT_PATH = ROOT / "data" / "W_bank_count_avg.npz"
WBANK_NG_PATH = ROOT / "data" / "W_bank_nonGeo.npz"
OPTIMA_PATH = ROOT / "output" / "composite_w_credit_optima.csv"

PERMUTATIONS = 999


def matrix_stats(W_sparse):
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
    W = W_sparse.tocsr()
    labels = np.asarray(labels)
    while True:
        keep = np.array(W.sum(axis=1)).ravel() > 0
        if keep.all():
            return row_standardize(W), labels
        if keep.sum() < 3:
            return row_standardize(W[keep, :][:, keep]), labels[keep]
        W = W[keep, :][:, keep].tocsr()
        labels = labels[keep]


def load_credit_residuals():
    hmda, _ = pyreadstat.read_dta(DTA_DIR / "hmda.dta")
    controls, _ = pyreadstat.read_dta(DTA_DIR / "hp_dereg_controls.dta")
    df = hmda.merge(controls, on=["county", "year"])
    required = [
        "Dl_nloans_b", "Linter_bra", "LDl_nloans_b",
        "Dl_inc", "LDl_inc", "Dl_pop", "LDl_pop",
        "Dl_hpi", "LDl_hpi", "Dl_her_v", "LDl_her_v",
        "county", "year",
    ]
    df = df.dropna(subset=required).copy()
    df["state"] = df["county"].astype(str).str.zfill(5).str[:2].astype(int)
    formula = (
        "Dl_nloans_b ~ Linter_bra + LDl_nloans_b "
        "+ Dl_inc + LDl_inc + Dl_pop + LDl_pop "
        "+ Dl_hpi + LDl_hpi + Dl_her_v + LDl_her_v "
        "+ C(county) + C(year)"
    )
    model = smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["state"]},
    )
    print(f"Credit residual regression: beta={model.params['Linter_bra']:.4f} "
          f"SE={model.bse['Linter_bra']:.4f}")
    df["residual"] = model.resid.values
    df["fips5"] = df["county"].astype(int).astype(str).str.zfill(5)

    panel = pd.read_csv(PANEL_PATH, dtype={"fips5": str})
    panel["fips5"] = panel["fips5"].str.zfill(5)
    panel["year"] = panel["year"].astype(int)
    border = panel[["fips5", "year", "border"]].copy()

    df = df.drop(columns=["border"], errors="ignore")
    df = df.merge(border, on=["fips5", "year"], how="left")
    return df[["county", "fips5", "year", "state", "border", "residual"]].copy()


def load_weight_specs():
    county_order = (
        pd.read_csv(COUNTY_PATH, dtype={"fips5": str})["fips5"]
        .str.zfill(5).tolist()
    )
    W_geo, _ = gal_to_W(GAL_PATH, county_order)
    W_alts = {
        "W_bank_bin": row_standardize(scipy.sparse.load_npz(WBANK_BIN_PATH)),
        "W_bank_count": row_standardize(scipy.sparse.load_npz(WBANK_CNT_PATH)),
        "W_bank_nonGeo": row_standardize(scipy.sparse.load_npz(WBANK_NG_PATH)),
    }
    optima = pd.read_csv(OPTIMA_PATH)

    specs = [("Endpoint", "All", "W_geo", 1.0, W_geo)]
    for _, row in optima.iterrows():
        W_alt = W_alts[row["w_alt"]]
        alpha = float(row["alpha_star"])
        sample = row["sample"]
        pair = row["pair"]
        w_alt = row["w_alt"]
        specs.append((pair, sample, f"{w_alt}_alpha_star", alpha,
                      build_combo(W_geo, W_alt, alpha)))
        specs.append((pair, sample, w_alt, 0.0, W_alt))

    return county_order, specs


def moran_by_year(resid_df, county_order, W_sparse, pair, sample, w_label, alpha):
    rows = []
    county_to_idx = {c: i for i, c in enumerate(county_order)}
    w_stats = matrix_stats(W_sparse)

    for year in sorted(resid_df["year"].unique()):
        yr = resid_df[resid_df["year"] == year].copy()
        present = [c for c in county_order if c in set(yr["fips5"])]

        idx = np.array([county_to_idx[c] for c in present])
        kept = np.array(present)
        W_sub, kept = trim_islands(W_sparse[idx, :][:, idx], kept)
        if len(kept) < 3:
            continue

        y = (
            yr.set_index("fips5")
            .reindex(kept)["residual"]
            .values.astype(float)
        )
        mi = Moran(y, WSP(W_sub.tocsr()).to_W(silence_warnings=True),
                   permutations=PERMUTATIONS)
        sig = ("***" if mi.p_sim < 0.01 else "**" if mi.p_sim < 0.05
               else "*" if mi.p_sim < 0.10 else "")
        print(f"  [{sample} {w_label}] {int(year)}: "
              f"I={mi.I:+.4f} z={mi.z_sim:+.2f} p={mi.p_sim:.3f} n={len(kept)} {sig}")

        rows.append({
            "outcome": "credit",
            "level": "county",
            "pair": pair,
            "sample": sample,
            "year": int(year),
            "w_matrix": w_label,
            "alpha": alpha,
            "moran_I": mi.I,
            "expected_I": mi.EI,
            "z_score": mi.z_sim,
            "p_value": mi.p_sim,
            "significant": mi.p_sim < 0.05,
            "n_units": int(len(kept)),
            **w_stats,
        })
    return rows


def plot_results(results, out_dir):
    df = results[results["w_matrix"].str.contains("alpha_star|W_geo")].copy()
    fig, ax = plt.subplots(figsize=(12, 7))
    for label, sub in df.groupby(["sample", "w_matrix"]):
        sample, w_matrix = label
        if sample not in ["Full", "Non-border", "All"]:
            continue
        lw = 2.5 if w_matrix == "W_geo" else 1.5
        ax.plot(sub["year"], sub["moran_I"], marker="o", lw=lw,
                label=f"{sample}: {w_matrix}")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title("Credit residual Moran's I under optimal composite W(alpha*)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Moran's I")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    path = out_dir / "moran_i_composite_credit_by_year.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Plot saved -> {path}")


def run(output_dir=None):
    if output_dir is not None:
        out = Path(output_dir) / "diagnostics"
    else:
        out = OUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    residuals = load_credit_residuals()
    county_order, specs = load_weight_specs()

    rows = []
    for pair, sample, w_label, alpha, W in specs:
        if sample == "Non-border":
            resid = residuals[residuals["border"] == 0].copy()
        else:
            resid = residuals.copy()
        print(f"\nMoran's I for {sample} | {w_label} (alpha={alpha:.2f})")
        rows.extend(moran_by_year(resid, county_order, W, pair, sample, w_label, alpha))

    results = pd.DataFrame(rows)
    results.to_csv(out / "moran_i_composite_credit_results.csv", index=False)
    print(f"\nResults saved -> {out / 'moran_i_composite_credit_results.csv'}")

    summary = (
        results.groupby(["pair", "sample", "w_matrix", "alpha"], as_index=False)
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
    summary.to_csv(out / "moran_i_composite_credit_summary.csv", index=False)
    print(f"Summary saved -> {out / 'moran_i_composite_credit_summary.csv'}")
    plot_results(results, out)
    return results, summary


if __name__ == "__main__":
    run(ROOT / "output")
