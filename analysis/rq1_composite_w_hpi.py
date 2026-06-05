"""
rq1_composite_w_hpi.py -- Composite W profile likelihood for house-price growth.

Profiles Panel_FE_Error log-likelihood over:

  W_combo(alpha) = alpha * W_geo + (1 - alpha) * W_bank

where alpha = 1 is pure geographic contiguity and alpha = 0 is pure bank
network overlap.

Dependent variable: Dl_hpi
Regressors: Linter_bra, Linter_ela, HPI controls, and year fixed effects.

Favara and Imbs weight house-price regressions by the inverse number of
counties per state. spreg.Panel_FE_Error does not natively support observation
weights, so this composite SEM is estimated unweighted, matching panel_fe_hpi.py.

Outputs:
  output/composite_w_hpi_results.csv
  output/composite_w_hpi_optima.csv
  output/composite_w_hpi_profile.png
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse
import spreg

import utils  # noqa: applies spreg compatibility patch
from panel_fe_hpi import (
    CONTROL_CANDIDATES,
    D_VAR,
    DETA_VAR,
    YVAR,
    load_panel,
)
from utils import gal_to_W, row_standardize, sparse_to_pysal_w


ROOT = Path(__file__).parent.parent
GAL_PATH = ROOT / "data" / "W_geo_queen.gal"
COUNTY_PATH = ROOT / "data" / "county_order_Wgeo.csv"
WBANK_PATH = ROOT / "data" / "W_bank_avg.npz"

ALPHA_GRID = np.linspace(0, 1, 21)


def build_combo(W_geo_sub, W_bank_sub, alpha):
    return alpha * W_geo_sub + (1.0 - alpha) * W_bank_sub


def build_arrays(panel_sub, county_order, W_geo_all, W_bank_all, years, year_pos, xvars):
    t = len(years)
    required = [YVAR] + xvars

    any_nan = (
        panel_sub.groupby("fips5")[required]
        .apply(lambda g: g.isna().any().any())
    )
    counts = panel_sub.groupby("fips5")["year"].nunique()
    sub_co = set(panel_sub["fips5"].unique())

    full_rs_geo = np.array(W_geo_all.sum(axis=1)).flatten()
    full_rs_bank = np.array(W_bank_all.sum(axis=1)).flatten()
    islands = (
        {county_order[i] for i, r in enumerate(full_rs_geo) if r == 0} |
        {county_order[i] for i, r in enumerate(full_rs_bank) if r == 0}
    )

    usable = [
        c for c in county_order
        if c in sub_co
        and counts.get(c, 0) == t
        and not any_nan.get(c, True)
        and c not in islands
    ]

    n = len(usable)
    usable_pos = {c: i for i, c in enumerate(usable)}
    df = (
        panel_sub[panel_sub["fips5"].isin(set(usable))]
        .assign(
            t_idx=lambda d: d["year"].map(year_pos),
            c_idx=lambda d: d["fips5"].map(usable_pos),
        )
        .sort_values(["t_idx", "c_idx"])
    )

    y_long = df[YVAR].values.reshape(-1, 1).astype(float)
    t_idx_vec = df["t_idx"].values
    year_dummies = np.column_stack([
        (t_idx_vec == year_pos[yr]).astype(np.float64)
        for yr in years[1:]
    ])
    x_long = np.hstack([df[xvars].values.astype(float), year_dummies])
    name_x = xvars + [f"yr{yr}" for yr in years[1:]]

    assert not np.isnan(y_long).any(), "NaN in HPI dependent variable"
    assert not np.isnan(x_long).any(), "NaN in HPI regressors"
    assert y_long.shape == (n * t, 1), f"y shape {y_long.shape}"
    assert x_long.shape == (n * t, len(name_x)), f"x shape {x_long.shape}"

    idx = np.array([county_order.index(c) for c in usable])
    W_geo_sub = row_standardize(W_geo_all[idx, :][:, idx])
    W_bank_sub = row_standardize(W_bank_all[idx, :][:, idx])

    return y_long, x_long, W_geo_sub, W_bank_sub, n, name_x


def run_alpha_grid(y, x, W_geo_sub, W_bank_sub, name_x, n, sample_label):
    rows = []
    for i, alpha in enumerate(ALPHA_GRID, start=1):
        alpha = float(alpha)
        print(f"  {sample_label} | alpha={alpha:.2f} ({i}/{len(ALPHA_GRID)})", flush=True)
        W_combo = build_combo(W_geo_sub, W_bank_sub, alpha)
        res = spreg.Panel_FE_Error(
            y, x, sparse_to_pysal_w(W_combo),
            name_y=YVAR,
            name_x=name_x,
            name_w=f"W_hpi_combo_a{alpha:.2f}",
            name_ds=sample_label,
        )
        idx_d = name_x.index(D_VAR)
        idx_deta = name_x.index(DETA_VAR)
        rows.append({
            "sample": sample_label,
            "alpha": alpha,
            "N": n,
            "n_obs": int(y.shape[0]),
            "lam": float(res.lam),
            "se_lam": float(res.std_err[-1]),
            "logll": float(res.logll),
            "sig2": float(np.array(res.sig2).flatten()[0]),
            "utu": float(res.utu),
            "beta_D": float(res.betas[idx_d, 0]),
            "se_beta_D": float(res.std_err[idx_d]),
            "beta_Deta": float(res.betas[idx_deta, 0]),
            "se_beta_Deta": float(res.std_err[idx_deta]),
        })
    return rows


def optima(rows_by_sample):
    opt_rows = []
    for sample_label, rows in rows_by_sample.items():
        best = max(rows, key=lambda r: r["logll"])
        geo_r = next(r for r in rows if np.isclose(r["alpha"], 1.0))
        bank_r = next(r for r in rows if np.isclose(r["alpha"], 0.0))
        baseline = max(geo_r["logll"], bank_r["logll"])
        opt_rows.append({
            "sample": sample_label,
            "alpha_star": best["alpha"],
            "lam_at_star": best["lam"],
            "se_lam_at_star": best["se_lam"],
            "beta_D_at_star": best["beta_D"],
            "se_beta_D_at_star": best["se_beta_D"],
            "beta_Deta_at_star": best["beta_Deta"],
            "se_beta_Deta_at_star": best["se_beta_Deta"],
            "logll_at_star": best["logll"],
            "logll_geo": geo_r["logll"],
            "logll_bank": bank_r["logll"],
            "logll_improvement": best["logll"] - baseline,
        })
    return opt_rows


def plot_profiles(rows_by_sample, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for sample_label, color, ls in [
        ("Full", "steelblue", "-"),
        ("Non-border", "firebrick", "--"),
    ]:
        rows = rows_by_sample[sample_label]
        alphas = [r["alpha"] for r in rows]
        axes[0].plot(alphas, [r["lam"] for r in rows],
                     color=color, ls=ls, lw=2, label=sample_label)
        axes[1].plot(alphas, [r["logll"] for r in rows],
                     color=color, ls=ls, lw=2, label=sample_label)

    axes[0].set_ylabel("lambda")
    axes[1].set_ylabel("log-likelihood")
    for ax in axes:
        ax.set_xlabel("alpha (0 = W_bank, 1 = W_geo)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("HPI-DV profile likelihood over composite W(alpha)")
    plt.tight_layout()
    plt.savefig(output_dir / "composite_w_hpi_profile.png", dpi=150)
    plt.close()


def run(output_dir=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    co_df = pd.read_csv(COUNTY_PATH, dtype={"fips5": str})
    county_order = co_df["fips5"].str.zfill(5).tolist()

    W_geo_all, gal_order = gal_to_W(GAL_PATH, county_order)
    assert gal_order == county_order, "GAL order mismatch"
    W_bank_all = row_standardize(scipy.sparse.load_npz(WBANK_PATH))

    panel = load_panel()
    years = sorted(panel["year"].unique())
    year_pos = {yr: i for i, yr in enumerate(years)}
    xvars = [D_VAR, DETA_VAR] + [c for c in CONTROL_CANDIDATES if c in panel.columns]

    rows_by_sample = {}
    all_rows = []
    for sample_label, panel_sub in [
        ("Full", panel),
        ("Non-border", panel[panel["border"] == 0].copy()),
    ]:
        y, x, W_geo_sub, W_bank_sub, n, name_x = build_arrays(
            panel_sub, county_order, W_geo_all, W_bank_all, years, year_pos, xvars,
        )
        rows = run_alpha_grid(y, x, W_geo_sub, W_bank_sub, name_x, n, sample_label)
        rows_by_sample[sample_label] = rows
        all_rows.extend(rows)

    opt_rows = optima(rows_by_sample)

    width = 104
    print()
    print("=" * width)
    print("HPI-DV Composite W Optima: alpha* = argmax logll(alpha)")
    print("alpha = 1.0 -> pure W_geo | alpha = 0.0 -> pure W_bank")
    print("=" * width)
    print(f"{'Sample':<14} {'alpha*':>7} {'lam*':>8} {'beta_D':>9} "
          f"{'beta_Deta':>10} {'logll*':>12} {'improvement':>12}")
    print("-" * width)
    for r in opt_rows:
        print(f"{r['sample']:<14} {r['alpha_star']:>7.2f} "
              f"{r['lam_at_star']:>8.4f} {r['beta_D_at_star']:>9.4f} "
              f"{r['beta_Deta_at_star']:>10.4f} {r['logll_at_star']:>12.2f} "
              f"{r['logll_improvement']:>12.4f}")
    print("=" * width)

    if output_dir is not None:
        pd.DataFrame(all_rows).to_csv(
            output_dir / "composite_w_hpi_results.csv", index=False,
        )
        pd.DataFrame(opt_rows).to_csv(
            output_dir / "composite_w_hpi_optima.csv", index=False,
        )
        plot_profiles(rows_by_sample, output_dir)
        print(f"\nSaved HPI composite W outputs to {output_dir}")

    return {"all_rows": all_rows, "optima_rows": opt_rows}


if __name__ == "__main__":
    run(ROOT / "output")
