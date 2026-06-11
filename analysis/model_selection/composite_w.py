"""
composite_w.py -- Composite W profile likelihood for credit growth
==================================================================

Profiles Panel_FE_Error log-likelihood over:

  W_combo(alpha) = alpha * W_geo + (1 - alpha) * W_bank

where alpha = 1 is pure geographic contiguity and alpha = 0 is pure bank
network overlap.

Dependent variable: Dl_nloans_b
Regressor: Linter_bra + year fixed effects

Outputs:
  output/composite_w_credit_results.csv
  output/composite_w_credit_optima.csv
  output/composite_w_credit_profiles.png
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
import scipy.sparse
import spreg

sys.path.insert(0, str(Path(__file__).parents[1]))
import utils  # noqa: applies spreg compatibility patch
from panel_data import load_panel_with_credit
from utils import row_standardize, sparse_to_pysal_w
from panel_data import get_samples
from w_variants import load_w_geo, load_bank_variants


ROOT        = Path(__file__).parents[2]
COUNTY_PATH = ROOT / "data" / "county_order_Wgeo.csv"

DV = "Dl_nloans_b"
ALPHA_GRID = np.linspace(0, 1, 21)


def build_combo(W_geo_sub, W_alt_sub, alpha):
    return alpha * W_geo_sub + (1.0 - alpha) * W_alt_sub


def run_alpha_grid(panel_sub, county_order, W_geo_all, W_alt_all, W_alt_label,
                   sample_label, years, year_pos, alpha_grid,
                   pair_label="", counter=None, total=126):
    t = len(years)

    full_rs_geo = np.array(W_geo_all.sum(axis=1)).flatten()
    full_rs_alt = np.array(W_alt_all.sum(axis=1)).flatten()
    islands = (
        {county_order[i] for i, r in enumerate(full_rs_geo) if r == 0} |
        {county_order[i] for i, r in enumerate(full_rs_alt) if r == 0}
    )

    any_nan = panel_sub.groupby("fips5")[DV].apply(lambda s: s.isna().any())
    sub_co = set(panel_sub["fips5"].unique())
    usable = [
        c for c in county_order
        if c in sub_co and not any_nan.get(c, True) and c not in islands
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

    y_long = df[DV].values.reshape(-1, 1)
    t_idx_vec = df["t_idx"].values
    year_dummies = np.column_stack([
        (t_idx_vec == year_pos[yr]).astype(np.float64)
        for yr in years[1:]
    ])
    x_long = np.hstack([df["Linter_bra"].values.reshape(-1, 1), year_dummies])

    assert not np.isnan(y_long).any(), f"NaN in y ({sample_label})"
    assert y_long.shape == (n * t, 1), f"y shape {y_long.shape}"
    assert x_long.shape == (n * t, 11), f"x shape {x_long.shape}"

    idx = np.array([county_order.index(c) for c in usable])
    W_geo_sub = row_standardize(W_geo_all[idx, :][:, idx])
    W_alt_sub = row_standardize(W_alt_all[idx, :][:, idx])
    nx = ["Linter_bra"] + [f"yr{yr}" for yr in years[1:]]

    rows = []
    for alpha in alpha_grid:
        alpha = float(alpha)
        if counter is not None:
            counter[0] += 1
            tag = f"({counter[0]}/{total})"
        else:
            tag = ""
        print(f"  {pair_label} | {sample_label} | alpha={alpha:.2f} {tag}", flush=True)

        W_combo = build_combo(W_geo_sub, W_alt_sub, alpha)
        try:
            res = spreg.Panel_FE_Error(
                y_long, x_long, sparse_to_pysal_w(W_combo),
                name_y=DV,
                name_x=nx,
                name_w=f"W_combo_a{alpha:.2f}",
                name_ds=sample_label,
            )
            rows.append({
                "alpha": alpha,
                "N": n,
                "lam": float(res.lam),
                "se_lam": float(res.std_err[-1]),
                "logll": float(res.logll),
                "sig2": float(np.array(res.sig2).flatten()[0]),
                "utu": float(res.utu),
                "beta": float(res.betas[0, 0]),
                "se_beta": float(res.std_err[0]),
            })
        except Exception as exc:
            print(f"  [WARN] {pair_label} | {sample_label} | alpha={alpha:.2f}: {exc}")

    return rows


def run(output_dir=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    co_df = pd.read_csv(COUNTY_PATH, dtype={"fips5": str})
    county_order = co_df["fips5"].str.zfill(5).tolist()

    W_geo_all, gal_order = load_w_geo(county_order)
    assert gal_order == county_order, "GAL order mismatch"

    bank_variants = load_bank_variants(county_order, W_geo_all=W_geo_all)

    panel    = load_panel_with_credit()
    panel["fips5"] = panel["fips5"].astype(str).str.zfill(5)
    years    = sorted(panel["year"].unique())
    year_pos = {yr: i for i, yr in enumerate(years)}

    pair_letters = "ABCDEFG"
    pairs = [
        (f"Pair {pair_letters[i]}", w_name, W_alt)
        for i, (w_name, W_alt) in enumerate(bank_variants.items())
    ]
    samples = get_samples(panel)

    total = len(ALPHA_GRID) * len(pairs) * len(samples)  # noqa: update if grid changes
    counter = [0]
    all_results = {}
    all_rows = []

    for sample_label, panel_sub in samples:
        for pair_label, w_alt_label, W_alt_all in pairs:
            rows = run_alpha_grid(
                panel_sub, county_order, W_geo_all, W_alt_all, w_alt_label,
                sample_label, years, year_pos, ALPHA_GRID,
                pair_label=pair_label, counter=counter, total=total,
            )
            all_results.setdefault(pair_label, {})[sample_label] = rows
            for r in rows:
                all_rows.append({
                    "pair": pair_label,
                    "w_alt": w_alt_label,
                    "sample": sample_label,
                    **r,
                })

    optima_rows = []
    for pair_label, w_alt_label, _ in pairs:
        for sample_label, _ in samples:
            rows = all_results.get(pair_label, {}).get(sample_label, [])
            if not rows:
                continue
            best = max(rows, key=lambda r: r["logll"])
            geo_r = next((r for r in rows if np.isclose(r["alpha"], 1.0)), None)
            bank_r = next((r for r in rows if np.isclose(r["alpha"], 0.0)), None)
            ll_geo = geo_r["logll"] if geo_r else np.nan
            ll_bank = bank_r["logll"] if bank_r else np.nan
            baseline = np.nanmax([ll_geo, ll_bank])
            optima_rows.append({
                "pair": pair_label,
                "w_alt": w_alt_label,
                "sample": sample_label,
                "alpha_star": best["alpha"],
                "lam_at_star": best["lam"],
                "se_lam_at_star": best["se_lam"],
                "beta_at_star": best["beta"],
                "se_beta_at_star": best["se_beta"],
                "logll_at_star": best["logll"],
                "logll_geo": ll_geo,
                "logll_bank": ll_bank,
                "logll_improvement": best["logll"] - baseline,
            })

    width = 110
    print()
    print("=" * width)
    print("Credit-DV Composite W Optima: alpha* = argmax logll(alpha)")
    print("alpha = 1.0 -> pure W_geo | alpha = 0.0 -> pure W_bank")
    print("=" * width)
    print(f"{'Pair':<8} {'W_alt':<18} {'Sample':<14} {'alpha*':>7} "
          f"{'lam*':>8} {'beta*':>8} {'logll*':>12} {'improvement':>12}")
    print("-" * width)
    for r in optima_rows:
        print(f"{r['pair']:<8} {r['w_alt']:<18} {r['sample']:<14} "
              f"{r['alpha_star']:>7.2f} {r['lam_at_star']:>8.4f} "
              f"{r['beta_at_star']:>8.4f} {r['logll_at_star']:>12.2f} "
              f"{r['logll_improvement']:>12.4f}")
    print("=" * width)

    if output_dir is not None:
        results_cols = [
            "pair", "w_alt", "sample", "alpha", "N",
            "lam", "se_lam", "logll", "sig2", "utu", "beta", "se_beta",
        ]
        pd.DataFrame(all_rows)[results_cols].to_csv(
            output_dir / "composite_w_credit_results.csv", index=False,
        )

        optima_cols = [
            "pair", "w_alt", "sample", "alpha_star",
            "lam_at_star", "se_lam_at_star",
            "beta_at_star", "se_beta_at_star",
            "logll_at_star", "logll_geo", "logll_bank", "logll_improvement",
        ]
        pd.DataFrame(optima_rows)[optima_cols].to_csv(
            output_dir / "composite_w_credit_optima.csv", index=False,
        )

        n_pairs = len(pairs)
        fig, axes = plt.subplots(2, n_pairs, figsize=(4 * n_pairs, 8))
        if n_pairs == 1:
            axes = axes.reshape(2, 1)
        for col, (pair_label, w_alt_label, _) in enumerate(pairs):
            for row_idx, (metric, ylabel) in enumerate([
                ("lam", "lambda"),
                ("logll", "log-likelihood"),
            ]):
                ax = axes[row_idx, col]
                for sample_label, color, ls in [
                    ("Full", "steelblue", "-"),
                    ("Border", "firebrick", "--"),
                ]:
                    prows = all_results.get(pair_label, {}).get(sample_label, [])
                    ax.plot([r["alpha"] for r in prows], [r[metric] for r in prows],
                            color=color, ls=ls, lw=2, label=sample_label)
                ax.set_title(f"W_geo vs {w_alt_label}" if row_idx == 0 else "")
                ax.set_xlabel("alpha (0 = W_bank, 1 = W_geo)")
                ax.set_ylabel(ylabel)
                ax.grid(alpha=0.3)
                if col == 0:
                    ax.legend(fontsize=8)
        fig.suptitle("Credit-DV profile likelihood over composite W(alpha)")
        plt.tight_layout()
        plt.savefig(output_dir / "composite_w_credit_profiles.png", dpi=150)
        plt.close()
        print(f"\nSaved credit composite W outputs to {output_dir}")

    return {
        "all_results": all_results,
        "all_rows": all_rows,
        "optima_rows": optima_rows,
    }


if __name__ == "__main__":
    run(ROOT / "output")
