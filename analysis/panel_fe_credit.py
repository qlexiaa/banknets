"""
panel_fe_credit.py -- Panel_FE_Error with credit growth as the DV
=================================================================
Parallel to panel_fe_error.py but uses Dl_nloans_b (Δln number of
commercial-bank mortgage loans, from Favara & Imbs HMDA data) as the
dependent variable instead of HPI.

Structural equation:
  Dl_nloans_b_it = β * Linter_bra_it + county FE + year FE + spatial error

Tests whether banking deregulation (Linter_bra) caused credit growth and
whether credit-supply shocks propagate through the same geographic or
bank-network spatial channels.

Data construction
-----------------
Dl_nloans_b is not in estimation_panel.csv. It is loaded from
  Replication/20121416_1data/data/hmda.dta
and left-merged into the estimation panel on [fips5, year].

NaN policy: Panel_FE_Error requires a balanced panel with no NaN in y.
  - No panel county is all-NaN for Dl_nloans_b (all 1023 appear in hmda).
  - 67 counties have 1-3 missing years (partial NaN).
  - These are dropped via an any-NaN filter (stricter than panel_fe_error.py's
    all-NaN filter for Linter_ela) to guarantee a clean balanced panel.

Outputs
-------
  output/panel_fe_credit_results.csv   -- same schema as panel_fe_results.csv
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
import scipy.sparse
import scipy.stats as stats
import spreg
import pyreadstat

import utils  # noqa: applies spreg Panel_FE_Error compatibility patch
from utils import gal_to_W, row_standardize, sparse_to_pysal_w

ROOT        = Path(__file__).parent.parent
PANEL_PATH  = ROOT / "data" / "estimation_panel.csv"
GAL_PATH    = ROOT / "data" / "W_geo_queen.gal"
COUNTY_PATH = ROOT / "data" / "county_order_Wgeo.csv"
WBANK_PATH  = ROOT / "data" / "W_bank_avg.npz"
HMDA_PATH   = ROOT / "Replication" / "20121416_1data" / "data" / "hmda.dta"


# ── Data loading helper ───────────────────────────────────────────────────────

def load_panel_with_credit():
    """Load estimation_panel.csv merged with Dl_nloans_b from hmda.dta."""
    panel = pd.read_csv(PANEL_PATH)
    panel["fips5"] = panel["fips5"].astype(str).str.zfill(5)

    if "Dl_nloans_b" in panel.columns:
        return panel   # already present -- nothing to do

    # Load hmda.dta and prepare fips5
    hmda_df, _ = pyreadstat.read_dta(HMDA_PATH)
    hmda_df["fips5"] = (
        hmda_df["county"].dropna().astype(int).astype(str).str.zfill(5)
    )
    hmda_df["year"] = hmda_df["year"].astype(int)
    hmda = hmda_df[["fips5", "year", "Dl_nloans_b"]].copy()

    panel = panel.merge(hmda, on=["fips5", "year"], how="left")
    return panel


# ── Array builder (mirrors panel_fe_error.py but y = Dl_nloans_b) ────────────

def build_arrays(panel, county_order, W_geo_all, W_bank_all,
                 YEARS, year_pos, N_ALL, panel_sub, sample_label):
    """
    Build long-format arrays for Panel_FE_Error with y = Dl_nloans_b.

    County filter (stricter than panel_fe_error.py):
      - Exclude counties where Dl_nloans_b has ANY NaN (67 counties with
        1-3 missing years) to guarantee a balanced panel.
      - Same island exclusion as panel_fe_error.py (no additional logic needed
        because W_geo and W_bank have no full-matrix islands for these counties).

    Arrays:
      y : (N*T, 1)  -- Dl_nloans_b, time-major
      x : (N*T, 11) -- [Linter_bra, yr1996, ..., yr2005] (same as panel_fe_error.py)
      w : N×N PySAL W (cross-sectional)
    """
    T = len(YEARS)

    # Drop counties with ANY NaN in Dl_nloans_b (required for balanced panel)
    any_nan  = panel_sub.groupby("fips5")["Dl_nloans_b"].apply(lambda s: s.isna().any())
    sub_co   = set(panel_sub["fips5"].unique())
    usable   = [c for c in county_order
                if c in sub_co and not any_nan.get(c, True)]
    N        = len(usable)
    usable_pos = {c: i for i, c in enumerate(usable)}

    df = (panel_sub[panel_sub["fips5"].isin(set(usable))]
          .assign(t_idx=lambda d: d["year"].map(year_pos),
                  c_idx=lambda d: d["fips5"].map(usable_pos))
          .sort_values(["t_idx", "c_idx"]))

    y_long       = df["Dl_nloans_b"].values.reshape(-1, 1)    # <-- changed DV
    dummy_years  = YEARS[1:]
    t_idx_vec    = df["t_idx"].values
    year_dummies = np.column_stack([
        (t_idx_vec == year_pos[yr]).astype(np.float64)
        for yr in dummy_years
    ])
    x_long = np.hstack([df["Linter_bra"].values.reshape(-1, 1), year_dummies])

    assert not np.isnan(y_long).any(), f"NaN in y ({sample_label})"
    assert not np.isnan(x_long[:, 0]).any(), f"NaN in Linter_bra ({sample_label})"
    assert y_long.shape == (N * T, 1),  f"y shape {y_long.shape}"
    assert x_long.shape == (N * T, 11), f"x shape {x_long.shape}"

    idx        = np.array([county_order.index(c) for c in usable])
    W_geo_sub  = row_standardize(W_geo_all[idx, :][:, idx])
    W_bank_sub = row_standardize(W_bank_all[idx, :][:, idx])

    return y_long, x_long, sparse_to_pysal_w(W_geo_sub), sparse_to_pysal_w(W_bank_sub), N


# ── Model runner ──────────────────────────────────────────────────────────────

def run_panel_fe(y, x, w_pysal, w_label, ds_label, YEARS):
    nx = ["Linter_bra"] + [f"yr{yr}" for yr in YEARS[1:]]
    return spreg.Panel_FE_Error(
        y, x, w_pysal,
        name_y="Dl_nloans_b",
        name_x=nx,
        name_w=w_label,
        name_ds=ds_label,
    )


def extract(res, N, T):
    beta  = float(res.betas[0, 0])
    lam   = float(res.lam)
    se_b  = float(res.std_err[0])
    se_l  = float(res.std_err[-1])
    z_b, p_b = res.z_stat[0]
    z_l, p_l = res.z_stat[-1]
    return dict(beta=beta, se_beta=se_b, z_beta=float(z_b), p_beta=float(p_b),
                lam=lam,   se_lam=se_l,  z_lam=float(z_l),  p_lam=float(p_l),
                n_co=N,    n_obs=N * T)


# ── Master run ────────────────────────────────────────────────────────────────

def run(output_dir=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    co_df        = pd.read_csv(COUNTY_PATH, dtype={"fips5": str})
    county_order = co_df["fips5"].str.zfill(5).tolist()
    N_ALL        = len(county_order)

    W_geo_all, gal_order = gal_to_W(GAL_PATH, county_order)
    assert gal_order == county_order, "GAL order mismatch"

    W_bank_all = row_standardize(scipy.sparse.load_npz(WBANK_PATH))

    panel = load_panel_with_credit()
    panel["fips5"] = panel["fips5"].astype(str).str.zfill(5)
    YEARS    = sorted(panel["year"].unique())
    T        = len(YEARS)
    year_pos = {yr: i for i, yr in enumerate(YEARS)}

    panel_nb = panel[panel["border"] == 0].copy()

    # ── Estimate ──────────────────────────────────────────────────────────────
    results = {}

    y_f, x_f, w_geo_f, w_bank_f, N_f = build_arrays(
        panel, county_order, W_geo_all, W_bank_all, YEARS, year_pos, N_ALL,
        panel, "Full")
    results["FE W_geo (full)"]  = extract(
        run_panel_fe(y_f, x_f, w_geo_f,  "W_geo",  "Full", YEARS), N_f, T)
    results["FE W_bank (full)"] = extract(
        run_panel_fe(y_f, x_f, w_bank_f, "W_bank", "Full", YEARS), N_f, T)

    y_nb, x_nb, w_geo_nb, w_bank_nb, N_nb = build_arrays(
        panel, county_order, W_geo_all, W_bank_all, YEARS, year_pos, N_ALL,
        panel_nb, "Non-border")
    results["FE W_geo (non-border)"]  = extract(
        run_panel_fe(y_nb, x_nb, w_geo_nb,  "W_geo",  "NB", YEARS), N_nb, T)
    results["FE W_bank (non-border)"] = extract(
        run_panel_fe(y_nb, x_nb, w_bank_nb, "W_bank", "NB", YEARS), N_nb, T)

    # ── Lambda gap tests ──────────────────────────────────────────────────────
    def gap_test(r_geo, r_bank):
        gap    = r_bank["lam"] - r_geo["lam"]
        se_gap = np.sqrt(r_geo["se_lam"]**2 + r_bank["se_lam"]**2)
        z_gap  = gap / se_gap
        p_gap  = float(stats.norm.sf(z_gap))   # one-sided H1: lam_bank > lam_geo
        return gap, se_gap, z_gap, p_gap

    def stars(p):
        return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))

    # ── Print results table ───────────────────────────────────────────────────
    W = 88
    print()
    print("=" * W)
    print("Panel_FE_Error -- DV: Dl_nloans_b (dln commercial-bank mortgage loans)")
    print("Regressor: Linter_bra (deregulation instrument) | Two-way FE (ML)")
    print("=" * W)
    print(f"{'Model':<28} {'N':>5}  "
          f"{'beta':>8} {'SE':>6}  "
          f"{'lambda':>8} {'SE':>6}  {'p_lam':>8}")
    print("-" * W)

    for model_name, r in results.items():
        sl = stars(r["p_lam"])
        sb = stars(r["p_beta"])
        print(f"{model_name:<28} {r['n_co']:>5}  "
              f"{r['beta']:>8.4f}{sb} {r['se_beta']:>6.4f}  "
              f"{r['lam']:>8.4f}{sl} {r['se_lam']:>6.4f}  {r['p_lam']:>8.4f}")

    print()
    print("Lambda gap (W_bank vs W_geo) -- H1: lam_bank > lam_geo (one-sided):")
    print(f"{'Sample':<14} {'gap':>7} {'se_gap':>7} {'z_gap':>7} {'p(1-tail)':>10} {'sig':>5}")
    print("-" * 50)

    for sample, geo_key, bank_key in [
        ("Full",       "FE W_geo (full)",       "FE W_bank (full)"),
        ("Non-border", "FE W_geo (non-border)", "FE W_bank (non-border)"),
    ]:
        gap, se_gap, z_gap, p_gap = gap_test(results[geo_key], results[bank_key])
        print(f"{sample:<14} {gap:>7.4f} {se_gap:>7.4f} {z_gap:>7.3f} "
              f"{p_gap:>10.4f} {stars(p_gap):>5}")

    print()
    print("Significance: *** p<0.01  ** p<0.05  * p<0.10")
    print("=" * W)

    # Reference: HPI results for comparison
    print()
    print("Reference (panel_fe_error.py -- DV: Linter_ela = HPI):")
    print("  Full:       lam_geo=0.8268  lam_bank=0.9879  gap=+0.1611 z=37.6 ***")
    print("  Non-border: lam_geo=0.8316  lam_bank=0.9873  gap=+0.1558 z=34.3 ***")

    # ── Save CSV (same schema as panel_fe_results.csv) ────────────────────────
    if output_dir is not None:
        rows = []
        for model_name, r in results.items():
            rows.append(dict(
                model=model_name,
                beta=r["beta"],     se_beta=r["se_beta"],
                z_beta=r["z_beta"], p_beta=r["p_beta"],
                lam=r["lam"],       se_lam=r["se_lam"],
                z_lam=r["z_lam"],   p_lam=r["p_lam"],
                n_co=r["n_co"],     n_obs=r["n_obs"],
            ))
        pd.DataFrame(rows).to_csv(
            output_dir / "panel_fe_credit_results.csv", index=False)
        print(f"\nSaved panel_fe_credit_results.csv to {output_dir}")

    return results


if __name__ == '__main__':
    run(Path(__file__).parent.parent / 'output')
