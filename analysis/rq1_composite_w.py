"""
rq1_composite_w.py -- Composite W profile likelihood (Conley & Topa 2002 style)
================================================================================
Parameterises the spatial weight matrix as a convex combination:

  W_combo(alpha) = alpha * W_geo + (1 - alpha) * W_bank

and profiles the Panel_FE_Error log-likelihood over alpha in [0, 1] to find
the maximum-likelihood mixing weight alpha*.

alpha = 1.0  ->  pure W_geo  (queen contiguity)
alpha = 0.0  ->  pure W_bank (network matrix)

VERIFIED: the convex combination of two row-standardised sparse matrices is
automatically row-standardised (row sums = 1 for all alpha). No re-normalisation
is needed or correct.

Three W_bank variants (W_alt):
  Pair A: W_bank_bin    (data/W_bank_avg.npz)
  Pair B: W_bank_count  (data/W_bank_count_avg.npz)
  Pair C: W_bank_nonGeo (data/W_bank_nonGeo.npz — build via rq1_four_w_comparison.py)

Alpha grid: 21 values from 0.00 to 1.00 in steps of 0.05.
Two samples: Full and Non-border (border == 0).
Total Panel_FE_Error calls: 21 * 2 * 3 = 126.

Outputs:
  output/composite_w_results.csv   — all 126 runs
  output/composite_w_optima.csv    — alpha* per pair × sample
  output/composite_w_profiles.png  — 2x3 figure: lambda and logll profiles
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse
import spreg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils  # noqa: applies spreg compatibility patch
from utils import gal_to_W, row_standardize, sparse_to_pysal_w

ROOT           = Path(__file__).parent.parent
PANEL_PATH     = ROOT / "data" / "estimation_panel.csv"
GAL_PATH       = ROOT / "data" / "W_geo_queen.gal"
COUNTY_PATH    = ROOT / "data" / "county_order_Wgeo.csv"
WBANK_BIN_PATH = ROOT / "data" / "W_bank_avg.npz"
WBANK_CNT_PATH = ROOT / "data" / "W_bank_count_avg.npz"
WBANK_NG_PATH  = ROOT / "data" / "W_bank_nonGeo.npz"

ALPHA_GRID = np.linspace(0, 1, 21)   # 0.00, 0.05, 0.10, ..., 0.95, 1.00


# ── W_combo constructor ───────────────────────────────────────────────────────

def build_combo(W_geo_sub, W_alt_sub, alpha):
    """Return composite sparse W: alpha*W_geo + (1-alpha)*W_alt.
    Both inputs must be row-standardised; result is automatically row-standardised."""
    return alpha * W_geo_sub + (1 - alpha) * W_alt_sub


# ── Core grid-runner ─────────────────────────────────────────────────────────

def run_alpha_grid(panel_sub, county_order, W_geo_all, W_alt_all, W_alt_label,
                   sample_label, YEARS, year_pos, ALPHA_GRID,
                   pair_label="", counter=None, total=126):
    """
    Run Panel_FE_Error for every alpha in ALPHA_GRID on a fixed county set.

    County set built once per (pair, sample):
      - Same all-NaN filter as panel_fe_error.py
      - Additionally removes full-matrix islands from EITHER W (ensures the
        composite W has valid row sums for all alpha values)

    Parameters
    ----------
    panel_sub   : DataFrame — pre-filtered sample (full or non-border)
    county_order: list of fips5 in W row order
    W_geo_all   : scipy sparse N×N, row-standardised
    W_alt_all   : scipy sparse N×N, row-standardised (W_bank variant)
    W_alt_label : string label for W_alt
    sample_label: 'Full' or 'Non-border'
    YEARS       : sorted list of years
    year_pos    : dict {year: index}
    ALPHA_GRID  : iterable of alpha values
    pair_label  : label for progress printing
    counter     : mutable list [int] for global call count
    total       : total expected calls (for progress display)

    Returns
    -------
    List of result dicts, one per successful alpha.
    """
    T = len(YEARS)

    # Islands in the FULL N×N matrices — union excluded for consistent county set
    full_rs_geo = np.array(W_geo_all.sum(axis=1)).flatten()
    full_rs_alt = np.array(W_alt_all.sum(axis=1)).flatten()
    islands = (
        {county_order[i] for i, r in enumerate(full_rs_geo) if r == 0} |
        {county_order[i] for i, r in enumerate(full_rs_alt)  if r == 0}
    )

    # County subsetting (same logic as panel_fe_error.py + island exclusion)
    na_all = panel_sub.groupby("fips5")["Linter_ela"].apply(lambda s: s.isna().all())
    sub_co = set(panel_sub["fips5"].unique())
    usable = [
        c for c in county_order
        if c in sub_co and not na_all.get(c, True) and c not in islands
    ]
    N          = len(usable)
    usable_pos = {c: i for i, c in enumerate(usable)}

    df = (
        panel_sub[panel_sub["fips5"].isin(set(usable))]
        .assign(
            t_idx=lambda d: d["year"].map(year_pos),
            c_idx=lambda d: d["fips5"].map(usable_pos),
        )
        .sort_values(["t_idx", "c_idx"])
    )

    y_long       = df["Linter_ela"].values.reshape(-1, 1)
    t_idx_vec    = df["t_idx"].values
    year_dummies = np.column_stack([
        (t_idx_vec == year_pos[yr]).astype(np.float64)
        for yr in YEARS[1:]
    ])
    x_long = np.hstack([df["Linter_bra"].values.reshape(-1, 1), year_dummies])

    assert not np.isnan(y_long).any(), f"NaN in y ({sample_label})"
    assert y_long.shape == (N * T, 1),  f"y shape {y_long.shape}"
    assert x_long.shape == (N * T, 11), f"x shape {x_long.shape}"

    # Submatrices built ONCE — reused for every alpha
    idx       = np.array([county_order.index(c) for c in usable])
    W_geo_sub = row_standardize(W_geo_all[idx, :][:, idx])
    W_alt_sub = row_standardize(W_alt_all[idx, :][:, idx])

    nx = ["Linter_bra"] + [f"yr{yr}" for yr in YEARS[1:]]

    results = []
    for alpha in ALPHA_GRID:
        alpha = float(alpha)
        if counter is not None:
            counter[0] += 1
            tag = f"({counter[0]}/{total})"
        else:
            tag = ""
        print(f"  {pair_label} | {sample_label} | alpha={alpha:.2f} {tag}", flush=True)

        W_combo = build_combo(W_geo_sub, W_alt_sub, alpha)
        w_pysal = sparse_to_pysal_w(W_combo)

        try:
            res = spreg.Panel_FE_Error(
                y_long, x_long, w_pysal,
                name_y="Linter_ela",
                name_x=nx,
                name_w=f"W_combo_a{alpha:.2f}",
                name_ds=sample_label,
            )
            sig2_val = float(np.array(res.sig2).flatten()[0])
            results.append(dict(
                alpha   = alpha,
                N       = N,
                lam     = float(res.lam),
                se_lam  = float(res.std_err[-1]),
                logll   = float(res.logll),
                sig2    = sig2_val,
                utu     = float(res.utu),
                beta    = float(res.betas[0, 0]),
                se_beta = float(res.std_err[0]),
            ))
        except Exception as e:
            print(f"  [WARN] {pair_label} | {sample_label} | alpha={alpha:.2f} skipped: {e}")

    return results


# ── Master run ────────────────────────────────────────────────────────────────

def run(output_dir=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load shared inputs ────────────────────────────────────────────────────
    co_df        = pd.read_csv(COUNTY_PATH, dtype={"fips5": str})
    county_order = co_df["fips5"].str.zfill(5).tolist()

    W_geo_all, gal_order = gal_to_W(GAL_PATH, county_order)
    assert gal_order == county_order, "GAL order mismatch"

    W_bin_all = row_standardize(scipy.sparse.load_npz(WBANK_BIN_PATH))
    W_cnt_all = row_standardize(scipy.sparse.load_npz(WBANK_CNT_PATH))

    if not WBANK_NG_PATH.exists():
        raise FileNotFoundError(
            f"{WBANK_NG_PATH} not found — run rq1_four_w_comparison.py first."
        )
    W_ng_all = row_standardize(scipy.sparse.load_npz(WBANK_NG_PATH))

    panel = pd.read_csv(PANEL_PATH)
    panel["fips5"] = panel["fips5"].astype(str).str.zfill(5)
    YEARS    = sorted(panel["year"].unique())
    year_pos = {yr: i for i, yr in enumerate(YEARS)}

    panel_nb = panel[panel["border"] == 0].copy()

    PAIRS = [
        ("Pair A", "W_bank_bin",    W_bin_all),
        ("Pair B", "W_bank_count",  W_cnt_all),
        ("Pair C", "W_bank_nonGeo", W_ng_all),
    ]

    TOTAL   = len(ALPHA_GRID) * 2 * len(PAIRS)   # 21 * 2 * 3 = 126
    counter = [0]

    # ── Run all 126 models ────────────────────────────────────────────────────
    # Storage: all_results[pair_label][sample_label] = list of result dicts
    all_results = {}
    all_rows    = []   # flat list for composite_w_results.csv

    for sample_label, panel_sub in [("Full", panel), ("Non-border", panel_nb)]:
        for pair_label, W_alt_label, W_alt_all in PAIRS:

            results = run_alpha_grid(
                panel_sub, county_order, W_geo_all, W_alt_all, W_alt_label,
                sample_label, YEARS, year_pos, ALPHA_GRID,
                pair_label=pair_label, counter=counter, total=TOTAL,
            )

            all_results.setdefault(pair_label, {})[sample_label] = results

            for r in results:
                all_rows.append(dict(
                    pair    = pair_label,
                    w_alt   = W_alt_label,
                    sample  = sample_label,
                    **r,
                ))

    # ── Compute optima ────────────────────────────────────────────────────────
    optima_rows = []
    for pair_label, W_alt_label, _ in PAIRS:
        for sample_label in ["Full", "Non-border"]:
            results = all_results.get(pair_label, {}).get(sample_label, [])
            if not results:
                continue

            best   = max(results, key=lambda r: r["logll"])
            geo_r  = next((r for r in results if np.isclose(r["alpha"], 1.0)), None)
            bank_r = next((r for r in results if np.isclose(r["alpha"], 0.0)), None)
            ll_geo  = geo_r["logll"]  if geo_r  else np.nan
            ll_bank = bank_r["logll"] if bank_r else np.nan

            valid_lls = [x for x in [ll_geo, ll_bank] if not np.isnan(x)]
            ll_baseline    = max(valid_lls) if valid_lls else np.nan
            ll_improvement = best["logll"] - ll_baseline if not np.isnan(ll_baseline) else np.nan

            optima_rows.append(dict(
                pair             = pair_label,
                w_alt            = W_alt_label,
                sample           = sample_label,
                alpha_star       = best["alpha"],
                lam_at_star      = best["lam"],
                se_lam_at_star   = best["se_lam"],
                beta_at_star     = best["beta"],
                se_beta_at_star  = best["se_beta"],
                logll_at_star    = best["logll"],
                logll_geo        = ll_geo,
                logll_bank       = ll_bank,
                logll_improvement = ll_improvement,
            ))

    # ── Print optima table ────────────────────────────────────────────────────
    W = 106
    print()
    print("=" * W)
    print("Profile Likelihood Optima:  alpha* = argmax logll(alpha)")
    print("alpha = 1.0 -> pure W_geo   |   alpha = 0.0 -> pure W_alt (W_bank)")
    print("logll_improvement = logll(alpha*) - max(logll_geo, logll_bank)")
    print("=" * W)
    print(
        f"{'Pair':<8} {'W_alt':<18} {'Sample':<14} "
        f"{'alpha*':>7} {'lam*':>7} {'beta*':>7} "
        f"{'logll*':>12} {'ll_geo':>12} {'ll_bank':>12} {'improvement':>12}"
    )
    print("-" * W)
    for r in optima_rows:
        print(
            f"{r['pair']:<8} {r['w_alt']:<18} {r['sample']:<14} "
            f"{r['alpha_star']:>7.2f} {r['lam_at_star']:>7.4f} {r['beta_at_star']:>7.4f} "
            f"{r['logll_at_star']:>12.2f} {r['logll_geo']:>12.2f} {r['logll_bank']:>12.2f} "
            f"{r['logll_improvement']:>12.4f}"
        )
    print("=" * W)

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    if output_dir is not None:
        results_cols = [
            "pair", "w_alt", "sample", "alpha", "N",
            "lam", "se_lam", "logll", "sig2", "utu", "beta", "se_beta",
        ]
        pd.DataFrame(all_rows)[results_cols].to_csv(
            output_dir / "composite_w_results.csv", index=False
        )

        optima_cols = [
            "pair", "sample", "alpha_star",
            "lam_at_star", "se_lam_at_star",
            "beta_at_star", "se_beta_at_star",
            "logll_at_star", "logll_geo", "logll_bank", "logll_improvement",
        ]
        pd.DataFrame(optima_rows)[optima_cols].to_csv(
            output_dir / "composite_w_optima.csv", index=False
        )

        # ── Figure: 2×3 profiles ──────────────────────────────────────────────
        pair_labels = ["Pair A", "Pair B", "Pair C"]
        pair_titles = [
            "W_geo vs W_bank_bin",
            "W_geo vs W_bank_count",
            "W_geo vs W_bank_nonGeo",
        ]
        metrics     = [("lam",   "lambda (spatial error param)"),
                       ("logll", "log-likelihood")]
        sample_styles = [
            ("Full",       "steelblue", "-",  "Full sample"),
            ("Non-border", "firebrick", "--", "Non-border"),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(14, 8))

        for col, (pair_label, pair_title) in enumerate(zip(pair_labels, pair_titles)):
            for row, (metric, ylabel) in enumerate(metrics):
                ax = axes[row, col]

                # alpha_star from full sample (for vertical line)
                alpha_star_full = None
                full_opt = next(
                    (r for r in optima_rows
                     if r["pair"] == pair_label and r["sample"] == "Full"),
                    None,
                )
                if full_opt:
                    alpha_star_full = full_opt["alpha_star"]

                for sample_label, color, ls, leg_label in sample_styles:
                    results = all_results.get(pair_label, {}).get(sample_label, [])
                    if not results:
                        continue
                    alphas = [r["alpha"] for r in results]
                    values = [r[metric]  for r in results]
                    ax.plot(alphas, values, color=color, ls=ls, lw=2, label=leg_label)

                if alpha_star_full is not None:
                    ax.axvline(
                        alpha_star_full,
                        color="black", ls=":", lw=1.5, alpha=0.7,
                        label=f"alpha*={alpha_star_full:.2f}",
                    )

                ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                ax.set_xlim(-0.03, 1.03)
                ax.set_xlabel("alpha  (0 = pure W_bank, 1 = pure W_geo)", fontsize=8)
                ax.set_ylabel(ylabel, fontsize=9)
                ax.grid(alpha=0.3)

                if row == 0:
                    ax.set_title(pair_title, fontsize=10, fontweight="bold")
                if col == 0 or (row == 0 and col == 0):
                    ax.legend(fontsize=8, loc="best")

        fig.suptitle(
            "Profile likelihood over composite W(alpha) = alpha*W_geo + (1-alpha)*W_bank\n"
            "Panel_FE_Error (county + year FE, ML)  |  alpha grid: 0, 0.05, ..., 1.0",
            fontsize=11,
        )
        plt.tight_layout()
        plt.savefig(output_dir / "composite_w_profiles.png", dpi=150)
        plt.close()

        print(
            f"\nSaved composite_w_results.csv, composite_w_optima.csv, "
            f"composite_w_profiles.png to {output_dir}"
        )

    return dict(all_results=all_results, all_rows=all_rows, optima_rows=optima_rows)


if __name__ == '__main__':
    run(Path(__file__).parent.parent / 'output')
