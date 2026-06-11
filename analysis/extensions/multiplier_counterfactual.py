"""
multiplier_counterfactual.py
============================
Spatial multiplier decomposition and counterfactual analysis for the SAR
(Panel_FE_Lag) credit-growth model.

Loads point estimates from output/sar_robustness_credit.csv. For each
sample x W combination, computes the Lesage & Pace (2009) scalar multiplier:

  M = 1 / (1 - rho)

which applies to any row-standardised W where W*1 = 1. The counterfactual
comparison is:

  Naive   : deltay_naive   = beta_D              (unit shock, no network)
  Network : deltay_network = beta_D / (1 - rho)  (unit shock with SAR feedback)

The network amplification equals the indirect effect:

  deltay_network - deltay_naive = beta_D * rho / (1 - rho)

Parametric bootstrap (1000 draws, seed = 0):
  (rho*, beta*) ~ N([rho_hat, beta_hat], diag([se_rho^2, se_beta^2]))
  Draws with |rho*| >= 1 are discarded (SAR stability).  CIs are the
  5th and 95th percentiles of accepted draws.

Output
------
  output/multiplier_counterfactual.csv   -- multiplier table + bootstrap CI
  output/multiplier_counterfactual.png   -- effect decomposition plot
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
import scipy.stats as st

ROOT       = Path(__file__).parents[2]
SAR_CSV    = ROOT / "output" / "sar_robustness_credit.csv"
BOOT_SEED  = 0
N_BOOT     = 1000
# W variants to display in the plot (subset for readability)
PLOT_WS    = ["W_geo", "W_bank", "W_bank_binary", "W_bank_knn4"]


# ==============================================================================
# Multiplier analytics
# ==============================================================================

def _effects(rho, beta):
    """
    Scalar multiplier effects for row-stochastic W.

    Returns (direct, indirect, total) where:
      direct   = beta
      total    = beta / (1 - rho)
      indirect = total - direct
    """
    M        = 1.0 / (1.0 - rho)
    direct   = float(beta)
    total    = float(beta * M)
    indirect = total - direct
    return direct, indirect, total, float(M)


def _bootstrap_effects(rho_hat, rho_se, beta_hat, beta_se):
    """
    Parametric bootstrap for the effect decomposition.

    Draws from N([rho_hat, beta_hat], diag([rho_se^2, beta_se^2])).
    Discards draws with |rho*| >= 1 (SAR stability).

    Returns dict of (mean, p5, p95) for direct, indirect, total, multiplier.
    """
    rng  = np.random.default_rng(BOOT_SEED)
    rhos = rng.normal(rho_hat, rho_se, size=N_BOOT * 10)   # oversample
    bets = rng.normal(beta_hat, beta_se, size=N_BOOT * 10)

    stable = np.abs(rhos) < 1
    rhos   = rhos[stable][:N_BOOT]
    bets   = bets[stable][:N_BOOT]

    if len(rhos) < 100:
        return None   # too few stable draws -- skip

    Ms   = 1.0 / (1.0 - rhos)
    dirs = bets
    tots = bets * Ms
    ints = tots - dirs

    def _ci(arr):
        return float(arr.mean()), float(np.percentile(arr, 5)), float(np.percentile(arr, 95))

    mean_M,   p5_M,   p95_M   = _ci(Ms)
    mean_dir, p5_dir, p95_dir = _ci(dirs)
    mean_int, p5_int, p95_int = _ci(ints)
    mean_tot, p5_tot, p95_tot = _ci(tots)

    return dict(
        n_draws      = len(rhos),
        mean_M       = mean_M,   p5_M    = p5_M,   p95_M    = p95_M,
        mean_direct  = mean_dir, p5_dir  = p5_dir,  p95_dir  = p95_dir,
        mean_indirect= mean_int, p5_int  = p5_int,  p95_int  = p95_int,
        mean_total   = mean_tot, p5_tot  = p5_tot,  p95_tot  = p95_tot,
    )


# ==============================================================================
# Plot
# ==============================================================================

def _make_plot(csv_rows, output_dir):
    """
    Effect decomposition plot: direct, indirect, total for each W matrix.
    One column per sample (Full, Border), three grouped bars per W variant.
    """
    df = pd.DataFrame(csv_rows)
    sample_order = ["Full", "Contig", "NonContig"]
    samples = [s for s in sample_order if s in set(df["sample"])]
    n_sam   = len(samples)

    fig, axes = plt.subplots(1, n_sam, figsize=(5 * n_sam + 1, 5), sharey=False)
    if n_sam == 1:
        axes = [axes]

    colors = {"Direct": "#4878cf", "Indirect": "#d65f5f", "Total": "#6acc65"}
    x_labels = []

    for ax, samp in zip(axes, samples):
        sub = df[(df["sample"] == samp) & (df["w_matrix"].isin(PLOT_WS))].copy()
        sub = sub.set_index("w_matrix")
        sub = sub.reindex([w for w in PLOT_WS if w in sub.index])
        n_w  = len(sub)
        x    = np.arange(n_w)
        bw   = 0.25   # bar width
        offsets = [-bw, 0, bw]
        names   = ["Direct", "Indirect", "Total"]
        cols    = ["point_direct", "point_indirect", "point_total"]
        lo_cols = ["p5_dir",       "p5_int",          "p5_tot"]
        hi_cols = ["p95_dir",      "p95_int",          "p95_tot"]

        for offset, name, col, lo_col, hi_col in zip(offsets, names, cols, lo_cols, hi_cols):
            vals = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
            lo   = pd.to_numeric(sub[lo_col], errors="coerce").to_numpy(dtype=float)
            hi   = pd.to_numeric(sub[hi_col], errors="coerce").to_numpy(dtype=float)
            err_lo = np.maximum(vals - lo, 0.0)
            err_hi = np.maximum(hi - vals, 0.0)
            ax.bar(x + offset, vals, width=bw, label=name,
                   color=colors[name], alpha=0.85)
            ax.errorbar(x + offset, vals,
                        yerr=[np.abs(err_lo), np.abs(err_hi)],
                        fmt="none", color="black", capsize=3, linewidth=0.8)

        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels([w.replace("W_", "") for w in sub.index],
                           rotation=20, ha="right", fontsize=9)
        ax.set_title(f"{samp} sample", fontsize=11)
        ax.set_ylabel("Effect on deltaln(loans)")
        if ax is axes[0]:
            ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        "SAR multiplier decomposition -- unit interstate deregulation shock\n"
        "(direct, indirect, total; 90% bootstrap CI)",
        fontsize=10, y=1.01)
    plt.tight_layout()
    out_path = output_dir / "multiplier_counterfactual.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved multiplier_counterfactual.png to {output_dir}")


# ==============================================================================
# Public entry point
# ==============================================================================

def run(output_dir=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # -- Load SAR results -----------------------------------------------------
    if not SAR_CSV.exists():
        print(f"[SKIP] {SAR_CSV} not found. Run sar/sar_credit.py first.")
        return []

    sar = pd.read_csv(SAR_CSV)
    required_cols = {"sample", "w_matrix", "rho", "rho_se", "beta_D", "beta_D_se"}
    if not required_cols.issubset(sar.columns):
        print(f"[SKIP] {SAR_CSV} missing columns: {required_cols - set(sar.columns)}")
        return []

    csv_rows = []
    W_print  = 88

    print("=" * W_print)
    print("SAR Spatial Multiplier Decomposition (Lesage & Pace 2009)")
    print("Unit shock: deltaD = 1 for all counties  |  Row-stochastic W: M = 1/(1-rho)")
    print("=" * W_print)
    print(f"{'Sample':<10} {'W':<22} {'rho':>7} {'beta_D':>8}  "
          f"{'M':>6} {'direct':>8} {'indirect':>9} {'total':>8}  "
          f"{'total [90%CI]':<22}")
    print("-" * W_print)

    for _, row in sar.iterrows():
        samp   = row["sample"]
        w_name = row["w_matrix"]
        rho    = float(row["rho"])
        rho_se = float(row["rho_se"])
        beta   = float(row["beta_D"])
        beta_se= float(row["beta_D_se"])

        if abs(rho) >= 1:
            print(f"{samp:<10} {w_name:<22}  [unstable rho = {rho:.3f} -- skip]")
            continue

        direct, indirect, total, M = _effects(rho, beta)
        boot = _bootstrap_effects(rho, rho_se, beta, beta_se)

        p5_tot  = boot["p5_tot"]  if boot else np.nan
        p95_tot = boot["p95_tot"] if boot else np.nan
        ci_str  = f"[{p5_tot:+.4f}, {p95_tot:+.4f}]" if boot else "n/a"

        print(f"{samp:<10} {w_name:<22} {rho:>7.4f} {beta:>8.4f}  "
              f"{M:>6.3f} {direct:>8.4f} {indirect:>9.4f} {total:>8.4f}  "
              f"{ci_str}")

        row_dict = dict(
            sample        = samp,
            w_matrix      = w_name,
            rho           = rho,
            rho_se        = rho_se,
            beta_D        = beta,
            beta_D_se     = beta_se,
            multiplier    = M,
            point_direct  = direct,
            point_indirect= indirect,
            point_total   = total,
        )
        if boot:
            row_dict.update({
                "n_boot_draws" : boot["n_draws"],
                "mean_M"       : boot["mean_M"],
                "p5_M"         : boot["p5_M"],
                "p95_M"        : boot["p95_M"],
                "mean_direct"  : boot["mean_direct"],
                "p5_dir"       : boot["p5_dir"],
                "p95_dir"      : boot["p95_dir"],
                "mean_indirect": boot["mean_indirect"],
                "p5_int"       : boot["p5_int"],
                "p95_int"      : boot["p95_int"],
                "mean_total"   : boot["mean_total"],
                "p5_tot"       : boot["p5_tot"],
                "p95_tot"      : boot["p95_tot"],
            })
        else:
            for c in ["n_boot_draws","mean_M","p5_M","p95_M","mean_direct",
                      "p5_dir","p95_dir","mean_indirect","p5_int","p95_int",
                      "mean_total","p5_tot","p95_tot"]:
                row_dict[c] = np.nan

        csv_rows.append(row_dict)

    print("=" * W_print)

    # -- Save CSV -------------------------------------------------------------
    if output_dir and csv_rows:
        cols = ["sample", "w_matrix", "rho", "rho_se", "beta_D", "beta_D_se",
                "multiplier",
                "point_direct", "point_indirect", "point_total",
                "n_boot_draws",
                "mean_M",        "p5_M",    "p95_M",
                "mean_direct",   "p5_dir",   "p95_dir",
                "mean_indirect", "p5_int",   "p95_int",
                "mean_total",    "p5_tot",   "p95_tot"]
        pd.DataFrame(csv_rows)[cols].to_csv(
            output_dir / "multiplier_counterfactual.csv", index=False)
        print(f"\nSaved multiplier_counterfactual.csv to {output_dir}")

    # -- Plot -----------------------------------------------------------------
    if output_dir and csv_rows:
        try:
            _make_plot(csv_rows, output_dir)
        except Exception as exc:
            print(f"[WARN] plot failed: {exc}")

    return csv_rows


if __name__ == "__main__":
    run(Path(__file__).parents[2] / "output")
