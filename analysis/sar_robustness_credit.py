"""
sar_robustness_credit.py
========================
SAR (Spatial Lag) robustness check for the credit-growth SEM result.

Estimates spreg.Panel_FE_Lag on Dl_nloans_b under W_geo and W_bank for
full and non-border samples using exactly the same panel construction,
county filters, and W-loading as panel_fe_credit.py.

Panel_FE_Lag structural equation:
  y = rho * W*y + X*beta + county FE + u
  (spatial autoregressive process in the outcome, vs SEM's error process)

Reports rho, SE(rho), beta_D (Linter_bra coefficient), SE(beta_D),
and the gap delta_rho = rho_bank - rho_geo with combined SE and one-sided
z-statistic (same formula as the lambda gap in panel_fe_credit.py).

Side-by-side comparison with the SEM lambdas is printed at the end.

Output: output/sar_robustness_credit.csv
Columns: sample, w_matrix, n_counties, n_obs, rho, rho_se,
         beta_D, beta_D_se, delta_rho, delta_rho_se, z_stat
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
import scipy.sparse
import scipy.stats as stats_mod
import spreg

import utils  # noqa: applies spreg Panel_FE_Lag compatibility patch
from utils import gal_to_W, row_standardize, sparse_to_pysal_w
from panel_fe_credit import load_panel_with_credit, build_arrays

ROOT        = Path(__file__).parent.parent
GAL_PATH    = ROOT / "data" / "W_geo_queen.gal"
COUNTY_PATH = ROOT / "data" / "county_order_Wgeo.csv"
WBANK_PATH  = ROOT / "data" / "W_bank_avg.npz"
CREDIT_SEM_CSV = ROOT / "output" / "panel_fe_credit_results.csv"


def run_panel_fe_lag(y, x, w_pysal, w_label, ds_label, YEARS):
    nx = ["Linter_bra"] + [f"yr{yr}" for yr in YEARS[1:]]
    return spreg.Panel_FE_Lag(
        y, x, w_pysal,
        name_y="Dl_nloans_b",
        name_x=nx,
        name_w=w_label,
        name_ds=ds_label,
    )


def extract_lag(res, N, T):
    """Extract rho and beta_D from a Panel_FE_Lag result.

    betas layout (Panel_FE_Lag): [beta_Linter_bra, yr1996, ..., yr2005, rho]
    rho is last in betas, std_err, and z_stat — same convention as lambda
    in Panel_FE_Error.
    """
    beta_D = float(res.betas[0, 0])
    rho    = float(res.rho)                # spatial lag parameter
    se_b   = float(res.std_err[0])
    se_r   = float(res.std_err[-1])
    z_b, p_b = res.z_stat[0]
    z_r, p_r = res.z_stat[-1]
    return dict(
        beta_D=beta_D, se_beta_D=se_b, z_beta_D=float(z_b), p_beta_D=float(p_b),
        rho=rho,       rho_se=se_r,    z_rho=float(z_r),    p_rho=float(p_r),
        n_co=N,        n_obs=N * T,
    )


def run(output_dir=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load shared inputs (identical to panel_fe_credit.py) ─────────────────
    co_df        = pd.read_csv(COUNTY_PATH, dtype={"fips5": str})
    county_order = co_df["fips5"].str.zfill(5).tolist()
    N_ALL        = len(county_order)

    W_geo_all, gal_order = gal_to_W(GAL_PATH, county_order)
    assert gal_order == county_order, "GAL order mismatch"

    W_bank_all = row_standardize(scipy.sparse.load_npz(WBANK_PATH))

    panel    = load_panel_with_credit()
    panel["fips5"] = panel["fips5"].astype(str).str.zfill(5)
    YEARS    = sorted(panel["year"].unique())
    T        = len(YEARS)
    year_pos = {yr: i for i, yr in enumerate(YEARS)}
    panel_nb = panel[panel["border"] == 0].copy()

    # ── Estimate Panel_FE_Lag (reuse build_arrays from panel_fe_credit) ───────
    results = {}

    y_f, x_f, w_geo_f, w_bank_f, N_f = build_arrays(
        panel, county_order, W_geo_all, W_bank_all, YEARS, year_pos, N_ALL,
        panel, "Full")

    print("Estimating SAR (Full, W_geo)  ...", flush=True)
    results[("Full", "W_geo")]  = extract_lag(
        run_panel_fe_lag(y_f, x_f, w_geo_f,  "W_geo",  "Full", YEARS), N_f, T)

    print("Estimating SAR (Full, W_bank) ...", flush=True)
    results[("Full", "W_bank")] = extract_lag(
        run_panel_fe_lag(y_f, x_f, w_bank_f, "W_bank", "Full", YEARS), N_f, T)

    y_nb, x_nb, w_geo_nb, w_bank_nb, N_nb = build_arrays(
        panel, county_order, W_geo_all, W_bank_all, YEARS, year_pos, N_ALL,
        panel_nb, "Non-border")

    print("Estimating SAR (Non-border, W_geo)  ...", flush=True)
    results[("Non-border", "W_geo")]  = extract_lag(
        run_panel_fe_lag(y_nb, x_nb, w_geo_nb,  "W_geo",  "NB", YEARS), N_nb, T)

    print("Estimating SAR (Non-border, W_bank) ...", flush=True)
    results[("Non-border", "W_bank")] = extract_lag(
        run_panel_fe_lag(y_nb, x_nb, w_bank_nb, "W_bank", "NB", YEARS), N_nb, T)

    # ── Gap statistics (same formula as lambda gap in panel_fe_credit.py) ─────
    def gap_stat(r_geo, r_bank, one_sided=True):
        gap    = r_bank["rho"]   - r_geo["rho"]
        se_gap = np.sqrt(r_geo["rho_se"]**2 + r_bank["rho_se"]**2)
        z      = gap / se_gap
        p      = float(stats_mod.norm.sf(z)) if one_sided else float(2 * stats_mod.norm.sf(abs(z)))
        return gap, se_gap, z, p

    def stars(p):
        return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))

    # ── Print table ───────────────────────────────────────────────────────────
    W = 86
    print()
    print("=" * W)
    print("Panel_FE_Lag (SAR) -- DV: Dl_nloans_b | Robustness vs SEM result")
    print("Regressor: Linter_bra | Two-way FE (county + year dummies 1996-2005)")
    print("=" * W)
    print(f"{'Sample':<14} {'W':<10} {'N':>6}  "
          f"{'rho':>8} {'SE':>6}  {'beta_D':>8} {'SE':>6}")
    print("-" * W)

    for sample in ["Full", "Non-border"]:
        for w in ["W_geo", "W_bank"]:
            r  = results[(sample, w)]
            sr = stars(r["p_rho"])
            sb = stars(r["p_beta_D"])
            print(f"{sample:<14} {w:<10} {r['n_co']:>6}  "
                  f"{r['rho']:>8.4f}{sr} {r['rho_se']:>6.4f}  "
                  f"{r['beta_D']:>8.4f}{sb} {r['se_beta_D']:>6.4f}")
        gap, se_gap, z, p = gap_stat(results[(sample, "W_geo")],
                                      results[(sample, "W_bank")])
        print(f"  delta_rho ({sample}):  "
              f"{gap:+.4f}  SE={se_gap:.4f}  z={z:.3f}  p={p:.4f}{stars(p)}")
        print()

    print("Significance: *** p<0.01  ** p<0.05  * p<0.10")

    # ── Side-by-side SEM lambda vs SAR rho ───────────────────────────────────
    # Load SEM results for comparison
    try:
        sem = pd.read_csv(CREDIT_SEM_CSV)
        lam_geo_f  = float(sem.loc[sem["model"].str.contains("W_geo")  &
                                    sem["model"].str.contains("full"), "lam"].iloc[0])
        lam_bank_f = float(sem.loc[sem["model"].str.contains("W_bank") &
                                    sem["model"].str.contains("full"), "lam"].iloc[0])
        lam_geo_nb  = float(sem.loc[sem["model"].str.contains("W_geo")  &
                                     sem["model"].str.contains("non-border"), "lam"].iloc[0])
        lam_bank_nb = float(sem.loc[sem["model"].str.contains("W_bank") &
                                     sem["model"].str.contains("non-border"), "lam"].iloc[0])
        sem_available = True
    except Exception:
        sem_available = False

    print()
    print("=" * W)
    print("SIDE-BY-SIDE: SEM lambda vs SAR rho  (Full sample)")
    print("=" * W)
    rho_geo_f  = results[("Full", "W_geo")]["rho"]
    rho_bank_f = results[("Full", "W_bank")]["rho"]
    gap_lam_f  = lam_bank_f - lam_geo_f  if sem_available else float("nan")
    gap_rho_f  = rho_bank_f - rho_geo_f
    lam_geo_str  = f"{lam_geo_f:.3f}"  if sem_available else "n/a"
    lam_bank_str = f"{lam_bank_f:.3f}" if sem_available else "n/a"
    gap_lam_str  = f"{gap_lam_f:+.3f}" if sem_available else "n/a"

    print(f"{'':20} {'lambda (SEM)':>14} {'rho (SAR)':>12}")
    print(f"{'W_geo  full:':20} {lam_geo_str:>14} {rho_geo_f:>12.3f}")
    print(f"{'W_bank full:':20} {lam_bank_str:>14} {rho_bank_f:>12.3f}")
    print(f"{'Gap:':20} {gap_lam_str:>14} {gap_rho_f:>+12.3f}")
    print()
    rho_gap_full, _, rho_z_full, rho_p_full = gap_stat(
        results[("Full", "W_geo")], results[("Full", "W_bank")])
    confirms = "CONFIRMS" if rho_gap_full > 0 and rho_p_full < 0.05 else "DOES NOT CONFIRM"
    print(f"Verdict: SAR rho gap = {rho_gap_full:+.4f}  z={rho_z_full:.2f}  "
          f"p={rho_p_full:.4f}  --> {confirms} the SEM result")
    print("=" * W)

    # ── Build CSV rows ─────────────────────────────────────────────────────────
    csv_rows = []
    for sample in ["Full", "Non-border"]:
        r_geo  = results[(sample, "W_geo")]
        r_bank = results[(sample, "W_bank")]
        gap_val, se_gap_val, z_gap_val, _ = gap_stat(r_geo, r_bank)

        for w_label, r in [("W_geo", r_geo), ("W_bank", r_bank)]:
            csv_rows.append(dict(
                sample     = sample,
                w_matrix   = w_label,
                n_counties = r["n_co"],
                n_obs      = r["n_obs"],
                rho        = r["rho"],
                rho_se     = r["rho_se"],
                beta_D     = r["beta_D"],
                beta_D_se  = r["se_beta_D"],
                delta_rho    = gap_val    if w_label == "W_bank" else np.nan,
                delta_rho_se = se_gap_val if w_label == "W_bank" else np.nan,
                z_stat       = z_gap_val  if w_label == "W_bank" else np.nan,
            ))

    if output_dir is not None:
        cols = ["sample", "w_matrix", "n_counties", "n_obs",
                "rho", "rho_se", "beta_D", "beta_D_se",
                "delta_rho", "delta_rho_se", "z_stat"]
        pd.DataFrame(csv_rows)[cols].to_csv(
            output_dir / "sar_robustness_credit.csv", index=False)
        print(f"\nSaved sar_robustness_credit.csv to {output_dir}")

    return results, csv_rows


if __name__ == "__main__":
    run(Path(__file__).parent.parent / "output")
