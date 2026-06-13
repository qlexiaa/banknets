"""
sem_credit.py -- Panel_FE_Error with credit growth as the DV
=================================================================
Uses Dl_nloans_b (dln number of commercial-bank mortgage loans, from
Favara & Imbs HMDA data) as the dependent variable.

Structural equation, matching Favara & Imbs (2015), equation (1):
  Dl_nloans_b_it = beta * Linter_bra_it + gamma * X_ct
                    + county FE + year FE + spatial error

where X_ct contains LDl_nloans_b plus current and lagged changes in income per
capita, population, HPI, and Herfindahl loan concentration.

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
  - These are dropped via an any-NaN filter to guarantee a clean balanced panel.

Outputs
-------
  output/panel_fe_credit_results.csv   -- same schema as panel_fe_results.csv
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import scipy.sparse
import scipy.stats as stats
import spreg

sys.path.insert(0, str(Path(__file__).parents[1]))
import utils  # noqa: applies spreg Panel_FE_Error compatibility patch
from utils import row_standardize, sparse_to_pysal_w
from panel_data import CREDIT_CONTROLS, load_panel_with_credit, build_arrays, get_samples
from w_variants import load_w_geo, load_bank_variants

ROOT        = Path(__file__).parents[2]
COUNTY_PATH = ROOT / "data" / "county_order_Wgeo.csv"
X_VARS      = ["Linter_bra"] + CREDIT_CONTROLS


# ── Model runner ──────────────────────────────────────────────────────────────

def run_panel_fe(y, x, w_pysal, w_label, ds_label, YEARS):
    nx = X_VARS + [f"yr{yr}" for yr in YEARS[1:]]
    return spreg.Panel_FE_Error(
        y, x, w_pysal,
        name_y="Dl_nloans_b",
        name_x=nx,
        name_w=w_label,
        name_ds=ds_label,
    )


def extract(res, N, T):
    """Extract beta on Linter_bra, which remains first in name_x."""
    beta  = float(res.betas[0, 0])
    lam   = float(res.lam)
    se_b  = float(res.std_err[0])
    se_l  = float(res.std_err[-1])
    z_b, p_b = res.z_stat[0]
    z_l, p_l = res.z_stat[-1]
    return dict(beta=beta, se_beta=se_b, z_beta=float(z_b), p_beta=float(p_b),
                lam=lam,   se_lam=se_l,  z_lam=float(z_l),  p_lam=float(p_l),
                n_co=N,    n_obs=N * T)


def _build_arrays_variant(panel_sub, county_order, W_all, YEARS, year_pos,
                          sample_label, w_label):
    """Build long-format arrays for a variant W, reusing panel_data.build_arrays logic."""
    T       = len(YEARS)
    DV      = "Dl_nloans_b"
    nan_y   = panel_sub.groupby("fips5")[DV].apply(lambda s: s.isna().any())
    nan_x   = panel_sub.groupby("fips5")[X_VARS].apply(lambda g: g.isna().any().any())
    sub_co  = set(panel_sub["fips5"].unique())

    full_rs = np.array(W_all.sum(axis=1)).flatten()
    islands = {county_order[i] for i, r in enumerate(full_rs) if r == 0}

    before_controls = [
        c for c in county_order
        if c in sub_co and not nan_y.get(c, True) and c not in islands
    ]
    usable = [
        c for c in before_controls
        if not nan_x.get(c, True)
    ]
    N      = len(usable)
    u_pos  = {c: i for i, c in enumerate(usable)}
    print(
        f"    N before controls={len(before_controls)} | "
        f"after control NaN-drop={N} ({sample_label} x {w_label})",
        flush=True,
    )

    df = (panel_sub[panel_sub["fips5"].isin(set(usable))]
          .assign(t_idx=lambda d: d["year"].map(year_pos),
                  c_idx=lambda d: d["fips5"].map(u_pos))
          .sort_values(["t_idx", "c_idx"]))

    y_long       = df[DV].values.reshape(-1, 1)
    t_idx_vec    = df["t_idx"].values
    year_dummies = np.column_stack([
        (t_idx_vec == year_pos[yr]).astype(np.float64) for yr in YEARS[1:]])
    x_long = np.hstack([df[X_VARS].values.astype(np.float64), year_dummies])

    assert not np.isnan(y_long).any(), f"NaN in y ({sample_label})"
    assert not np.isnan(x_long).any(), f"NaN in X ({sample_label})"
    assert y_long.shape == (N * T, 1)
    assert x_long.shape == (N * T, len(X_VARS) + len(YEARS) - 1)

    idx   = np.array([county_order.index(c) for c in usable])
    W_sub = row_standardize(W_all[idx, :][:, idx])
    return y_long, x_long, sparse_to_pysal_w(W_sub), N


# ── Master run ────────────────────────────────────────────────────────────────

def run(output_dir=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    co_df        = pd.read_csv(COUNTY_PATH, dtype={"fips5": str})
    county_order = co_df["fips5"].str.zfill(5).tolist()

    W_geo_all, gal_order = load_w_geo(county_order)
    assert gal_order == county_order, "GAL order mismatch"

    bank_variants = load_bank_variants(county_order, W_geo_all=W_geo_all)

    panel = load_panel_with_credit()
    panel["fips5"] = panel["fips5"].astype(str).str.zfill(5)
    YEARS    = sorted(panel["year"].unique())
    T        = len(YEARS)
    year_pos = {yr: i for i, yr in enumerate(YEARS)}

    samples = get_samples(panel)

    # ── Estimate ──────────────────────────────────────────────────────────────
    results = {}

    for sample_label, panel_sub in samples:
        # W_geo benchmark
        print(f"  Estimating: {sample_label} x W_geo ...", flush=True)
        try:
            y, x, w, N = _build_arrays_variant(
                panel_sub, county_order, W_geo_all, YEARS, year_pos,
                sample_label, "W_geo")
            res = run_panel_fe(y, x, w, "W_geo", sample_label, YEARS)
            results[(sample_label, "W_geo")] = extract(res, N, T)
        except Exception as exc:
            print(f"  [SKIP] {sample_label} x W_geo: {exc}")
            results[(sample_label, "W_geo")] = None

        # All bank variants
        for w_name, W_all in bank_variants.items():
            print(f"  Estimating: {sample_label} x {w_name} ...", flush=True)
            try:
                y, x, w, N = _build_arrays_variant(
                    panel_sub, county_order, W_all, YEARS, year_pos,
                    sample_label, w_name)
                res = run_panel_fe(y, x, w, w_name, sample_label, YEARS)
                results[(sample_label, w_name)] = extract(res, N, T)
            except Exception as exc:
                print(f"  [SKIP] {sample_label} x {w_name}: {exc}")
                results[(sample_label, w_name)] = None

    # ── Lambda gap tests ──────────────────────────────────────────────────────
    def gap_test(r_geo, r_bank):
        if r_geo is None or r_bank is None:
            return (float("nan"),) * 4
        gap    = r_bank["lam"] - r_geo["lam"]
        se_gap = np.sqrt(r_geo["se_lam"]**2 + r_bank["se_lam"]**2)
        z_gap  = gap / se_gap
        p_gap  = float(stats.norm.sf(z_gap))
        return gap, se_gap, z_gap, p_gap

    def stars(p):
        if np.isnan(p):
            return ""
        return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))

    # ── Print results table ───────────────────────────────────────────────────
    W_COL = 90
    print()
    print("=" * W_COL)
    print("Panel_FE_Error -- DV: Dl_nloans_b (dln commercial-bank mortgage loans)")
    print("Regressors: " + ", ".join(X_VARS) + " | Two-way FE (ML)")
    print("=" * W_COL)
    print(f"{'Sample':<12} {'W matrix':<22} {'N':>5}  "
          f"{'beta':>8} {'SE':>6}  "
          f"{'lambda':>8} {'SE':>6}  {'p_lam':>8}")
    print("-" * W_COL)

    w_keys = ["W_geo"] + list(bank_variants.keys())
    for sample_label, _ in samples:
        for w_name in w_keys:
            r = results.get((sample_label, w_name))
            if r is None:
                print(f"{sample_label:<12} {w_name:<22}  SKIP")
                continue
            sl = stars(r["p_lam"])
            sb = stars(r["p_beta"])
            print(f"{sample_label:<12} {w_name:<22} {r['n_co']:>5}  "
                  f"{r['beta']:>8.4f}{sb} {r['se_beta']:>6.4f}  "
                  f"{r['lam']:>8.4f}{sl} {r['se_lam']:>6.4f}  {r['p_lam']:>8.4f}")
        print()

    print()
    print("Lambda gap (W_bank_* vs W_geo) -- H1: lam_bank > lam_geo (one-sided):")
    print(f"{'Sample':<12} {'W variant':<22} {'gap':>7} {'se_gap':>7} "
          f"{'z_gap':>7} {'p(1-tail)':>10} {'sig':>5}")
    print("-" * 72)

    for sample_label, _ in samples:
        r_geo = results.get((sample_label, "W_geo"))
        for w_name in bank_variants:
            r_bank = results.get((sample_label, w_name))
            gap, se_gap, z_gap, p_gap = gap_test(r_geo, r_bank)
            print(f"{sample_label:<12} {w_name:<22} {gap:>7.4f} {se_gap:>7.4f} "
                  f"{z_gap:>7.3f} {p_gap:>10.4f} {stars(p_gap):>5}")
        print()

    print("Significance: *** p<0.01  ** p<0.05  * p<0.10")
    print("=" * W_COL)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if output_dir is not None:
        rows = []
        for sample_label, _ in samples:
            for w_name in w_keys:
                r = results.get((sample_label, w_name))
                if r is None:
                    continue
                rows.append(dict(
                    model    = f"FE {w_name} ({sample_label.lower()})",
                    sample   = sample_label,
                    w_matrix = w_name,
                    **{k: r[k] for k in
                       ["beta","se_beta","z_beta","p_beta",
                        "lam","se_lam","z_lam","p_lam",
                        "n_co","n_obs"]},
                ))
        pd.DataFrame(rows).to_csv(
            output_dir / "panel_fe_credit_results.csv", index=False)
        print(f"\nSaved panel_fe_credit_results.csv to {output_dir}")

    return results


if __name__ == '__main__':
    run(Path(__file__).parents[2] / 'output')
