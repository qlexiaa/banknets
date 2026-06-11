"""
bank_interstate_credit.py
==========================
Endogeneity robustness check: W_bank restricted to cross-state links only.

Motivation
----------
W_bank links counties that share bank holding companies, but within-state
links may simply reflect banks selecting into economically similar counties
inside the same state economy.  Retaining only *interstate* bank links
isolates the part of the network most directly attributable to post-IBBEA
cross-state branch expansion -- the specific regulatory shock this thesis
studies -- and least likely to reflect within-state economic sorting.

Construction
------------
For every (i, j) entry in W_bank_avg where county i and county j belong
to the same state (identical first two FIPS digits), set w_ij = 0.
Re-row-standardise the residual matrix.  Counties that lose all neighbours
become structural islands and are excluded from the spatial-parameter
estimation (but kept in the beta_1 estimation).

W matrices
----------
  W_geo              -- queen contiguity benchmark (unchanged)
  W_bank_interstate  -- W_bank with all intra-state entries zeroed out

Output
------
  output/bank_interstate_credit_results.csv
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse
import scipy.stats as stats
import spreg

sys.path.insert(0, str(Path(__file__).parent))
import utils  # noqa: applies spreg compatibility patch
from utils import gal_to_W, row_standardize, sparse_to_pysal_w
from panel_fe_credit import load_panel_with_credit

ROOT          = Path(__file__).parent.parent
GAL_PATH      = ROOT / "data" / "W_geo_queen.gal"
COUNTY_PATH   = ROOT / "data" / "county_order_Wgeo.csv"
WBANK_PATH    = ROOT / "data" / "W_bank_avg.npz"
WBANK_IS_PATH = ROOT / "data" / "W_bank_interstate.npz"   # written by this script

DV = "Dl_nloans_b"


# ── Matrix construction ───────────────────────────────────────────────────────

def build_wbank_interstate(W_bank_sp, county_order, output_path):
    """
    Zero all W_bank entries where counties i and j share the same state
    (identical first two digits of their 5-digit FIPS code).

    Parameters
    ----------
    W_bank_sp    : scipy sparse CSR, N×N, row-standardised W_bank
    county_order : list of N FIPS strings (zero-padded to 5 chars)
    output_path  : path to save the resulting .npz

    Returns
    -------
    W_is_sp : scipy sparse CSR, N×N, row-standardised W_bank_interstate
    """
    N = len(county_order)
    states = np.array([c[:2] for c in county_order])   # shape (N,)

    W = W_bank_sp.toarray().astype(np.float64)
    np.fill_diagonal(W, 0.0)

    # Build boolean mask: True where i and j are in the SAME state
    same_state = states[:, None] == states[None, :]    # (N, N) broadcast
    np.fill_diagonal(same_state, False)                # diagonal already 0

    W[same_state] = 0.0                                # zero intra-state entries
    np.fill_diagonal(W, 0.0)                           # ensure clean diagonal

    W_is_sp = row_standardize(scipy.sparse.csr_matrix(W))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scipy.sparse.save_npz(str(output_path), W_is_sp)
    print(f"  Saved W_bank_interstate -> {output_path}")
    return W_is_sp


# ── Sparsity diagnostics ──────────────────────────────────────────────────────

def sparsity_stats(W_sp, label):
    """
    Report link count, density, sparsity, and average neighbours for W_sp.

    Returns a dict and also prints a formatted summary line.
    """
    N = W_sp.shape[0]
    W_arr = W_sp.toarray()
    np.fill_diagonal(W_arr, 0.0)

    n_links   = int(np.count_nonzero(W_arr))
    possible  = N * (N - 1)
    density   = n_links / possible
    sparsity  = 1.0 - density
    avg_nbrs  = n_links / N

    # Number of islands (rows that sum to 0 after row-standardisation)
    row_sums  = W_sp.sum(axis=1).A1
    n_islands = int((row_sums == 0).sum())

    print(f"  {label:<25}  N={N}  links={n_links:>9,}  "
          f"density={density:.6f}  sparsity={sparsity:.6f}  "
          f"avg_nbrs={avg_nbrs:6.2f}  islands={n_islands}")

    return dict(
        label     = label,
        N         = N,
        n_links   = n_links,
        possible  = possible,
        density   = density,
        sparsity  = sparsity,
        avg_nbrs  = avg_nbrs,
        n_islands = n_islands,
    )


# ── Array builder (mirrors bank_nongeo_credit.py exactly) ────────────────────

def build_arrays(panel_sub, county_order, W_all, YEARS, year_pos, sample_label):
    T = len(YEARS)

    # Counties with any NaN in the DV
    any_nan = panel_sub.groupby("fips5")[DV].apply(lambda s: s.isna().any())
    sub_co  = set(panel_sub["fips5"].unique())

    # Islands in the full N×N matrix (zero row-sum before subsetting)
    full_rs = np.array(W_all.sum(axis=1)).flatten()
    islands = {county_order[i] for i, r in enumerate(full_rs) if r == 0}

    usable     = [c for c in county_order
                  if c in sub_co and not any_nan.get(c, True) and c not in islands]
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

    y_long       = df[DV].values.reshape(-1, 1)
    t_idx_vec    = df["t_idx"].values
    year_dummies = np.column_stack([
        (t_idx_vec == year_pos[yr]).astype(np.float64) for yr in YEARS[1:]
    ])
    x_long = np.hstack([df["Linter_bra"].values.reshape(-1, 1), year_dummies])

    assert not np.isnan(y_long).any(), f"NaN in y ({sample_label})"
    assert y_long.shape == (N * T, 1),  f"y shape mismatch: {y_long.shape}"
    assert x_long.shape == (N * T, 11), f"x shape mismatch: {x_long.shape}"

    idx   = np.array([county_order.index(c) for c in usable])
    W_sub = row_standardize(W_all[idx, :][:, idx])
    return y_long, x_long, sparse_to_pysal_w(W_sub), N


# ── Estimation ────────────────────────────────────────────────────────────────

def run_fe(y, x, w, w_label, ds_label, YEARS):
    nx = ["Linter_bra"] + [f"yr{yr}" for yr in YEARS[1:]]
    return spreg.Panel_FE_Error(
        y, x, w,
        name_y=DV, name_x=nx, name_w=w_label, name_ds=ds_label,
    )


def extract(res, N, T, model_name):
    return dict(
        model   = model_name,
        beta    = float(res.betas[0, 0]),
        se_beta = float(res.std_err[0]),
        z_beta  = float(res.z_stat[0][0]),
        p_beta  = float(res.z_stat[0][1]),
        lam     = float(res.lam),
        se_lam  = float(res.std_err[-1]),
        z_lam   = float(res.z_stat[-1][0]),
        p_lam   = float(res.z_stat[-1][1]),
        n_co    = N,
        n_obs   = N * T,
    )


# ── Main entry point ──────────────────────────────────────────────────────────

def run(output_dir=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load county order and W matrices ───────────────────────────────────
    co_df        = pd.read_csv(COUNTY_PATH, dtype={"fips5": str})
    county_order = co_df["fips5"].str.zfill(5).tolist()

    W_geo_all, gal_order = gal_to_W(GAL_PATH, county_order)
    assert gal_order == county_order, "GAL order != county_order_Wgeo.csv"

    W_bank_raw = row_standardize(scipy.sparse.load_npz(WBANK_PATH))

    # ── 2. Build W_bank_interstate (or load if already saved) ─────────────────
    if WBANK_IS_PATH.exists():
        print("Loading cached W_bank_interstate ...")
        W_is_all = row_standardize(scipy.sparse.load_npz(WBANK_IS_PATH))
    else:
        print("Building W_bank_interstate ...")
        W_is_all = build_wbank_interstate(W_bank_raw, county_order, WBANK_IS_PATH)

    # ── 3. Sparsity diagnostics ───────────────────────────────────────────────
    print("\nSparsity diagnostics:")
    stats_geo  = sparsity_stats(W_geo_all,  "W_geo")
    stats_bank = sparsity_stats(W_bank_raw, "W_bank (full)")
    stats_is   = sparsity_stats(W_is_all,   "W_bank_interstate")

    pct_retained = 100 * stats_is["n_links"] / stats_bank["n_links"]
    print(f"\n  Interstate links retained: {pct_retained:.1f}% of W_bank links")
    print(f"  New islands created by interstate restriction: "
          f"{stats_is['n_islands'] - stats_bank['n_islands']}")

    # ── 4. Panel data ─────────────────────────────────────────────────────────
    panel = load_panel_with_credit()
    panel["fips5"] = panel["fips5"].astype(str).str.zfill(5)
    YEARS    = sorted(panel["year"].unique())
    T        = len(YEARS)
    year_pos = {yr: i for i, yr in enumerate(YEARS)}

    panel_nb = panel[panel["border"] == 0].copy()

    # ── 5. Estimate models ────────────────────────────────────────────────────
    results = {}
    print("\nEstimating Panel_FE_Error models ...")

    # Full sample
    # Each W matrix may exclude a different set of island counties, so y/x
    # arrays must be built separately for each W to keep dimensions consistent.
    y_f,    x_f,    w_geo_f, N_f    = build_arrays(panel, county_order, W_geo_all, YEARS, year_pos, "Full")
    y_is_f, x_is_f, w_is_f,  N_is_f = build_arrays(panel, county_order, W_is_all,  YEARS, year_pos, "Full")

    results["W_geo (full)"]              = extract(run_fe(y_f,    x_f,    w_geo_f, "W_geo",             "Full", YEARS), N_f,    T, "W_geo (full)")
    results["W_bank_interstate (full)"]  = extract(run_fe(y_is_f, x_is_f, w_is_f,  "W_bank_interstate", "Full", YEARS), N_is_f, T, "W_bank_interstate (full)")

    # Non-border sample
    y_nb,    x_nb,    w_geo_nb, N_nb    = build_arrays(panel_nb, county_order, W_geo_all, YEARS, year_pos, "Non-border")
    y_is_nb, x_is_nb, w_is_nb,  N_is_nb = build_arrays(panel_nb, county_order, W_is_all,  YEARS, year_pos, "Non-border")

    results["W_geo (non-border)"]             = extract(run_fe(y_nb,    x_nb,    w_geo_nb, "W_geo",             "NB", YEARS), N_nb,    T, "W_geo (non-border)")
    results["W_bank_interstate (non-border)"] = extract(run_fe(y_is_nb, x_is_nb, w_is_nb,  "W_bank_interstate", "NB", YEARS), N_is_nb, T, "W_bank_interstate (non-border)")

    # ── 6. Lambda gap tests ───────────────────────────────────────────────────
    def gap_test(r_geo, r_is):
        gap    = r_is["lam"] - r_geo["lam"]
        se_gap = np.sqrt(r_geo["se_lam"]**2 + r_is["se_lam"]**2)
        z_gap  = gap / se_gap
        p_gap  = float(stats.norm.sf(z_gap))   # one-sided: H1: lam_is > lam_geo
        return gap, se_gap, z_gap, p_gap

    def stars(p):
        return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))

    # ── 7. Print results ──────────────────────────────────────────────────────
    W = 90
    print()
    print("=" * W)
    print("Panel_FE_Error -- DV: Dl_nloans_b | W_bank_interstate endogeneity check")
    print("Regressors: Linter_bra | Two-way FE (county + year, ML)")
    print("=" * W)
    print(f"{'Model':<32} {'N':>5}  "
          f"{'beta':>8} {'SE':>6}  "
          f"{'lambda':>8} {'SE':>6}  {'p_lam':>8}")
    print("-" * W)

    for name, r in results.items():
        sl = stars(r["p_lam"])
        sb = stars(r["p_beta"])
        print(f"{name:<32} {r['n_co']:>5}  "
              f"{r['beta']:>8.4f}{sb} {r['se_beta']:>6.4f}  "
              f"{r['lam']:>8.4f}{sl} {r['se_lam']:>6.4f}  "
              f"{r['p_lam']:>8.4f}")

    print("-" * W)

    # Gap statistics
    for sample_tag, geo_key, is_key in [
        ("Full",       "W_geo (full)",       "W_bank_interstate (full)"),
        ("Non-border", "W_geo (non-border)", "W_bank_interstate (non-border)"),
    ]:
        g, se_g, z_g, p_g = gap_test(results[geo_key], results[is_key])
        print(f"  Lambda gap [{sample_tag:>11}]  "
              f"interstate - geo = {g:+.4f}  SE={se_g:.4f}  "
              f"z={z_g:.2f}  p(one-sided)={p_g:.4f}{stars(p_g)}")

    print()
    print("Note: SE(gap) ignores covariance between estimators -- treat as approximate.")
    print("Note: N may differ between W_geo and W_bank_interstate models because")
    print("      the interstate restriction creates additional island counties.")

    # ── 8. Sparsity summary table ─────────────────────────────────────────────
    print()
    print("Sparsity summary:")
    print(f"  {'Matrix':<25}  {'Links':>10}  {'Density':>10}  {'Sparsity':>10}  {'Avg nbrs':>9}  {'Islands':>7}")
    for s in [stats_geo, stats_bank, stats_is]:
        print(f"  {s['label']:<25}  {s['n_links']:>10,}  "
              f"{s['density']:>10.6f}  {s['sparsity']:>10.6f}  "
              f"{s['avg_nbrs']:>9.2f}  {s['n_islands']:>7}")

    # ── 9. Save results ───────────────────────────────────────────────────────
    df_results = pd.DataFrame(results.values())
    df_sparse  = pd.DataFrame([stats_geo, stats_bank, stats_is])

    if output_dir is not None:
        res_path = output_dir / "bank_interstate_credit_results.csv"
        sp_path  = output_dir / "bank_interstate_sparsity.csv"
        df_results.to_csv(res_path, index=False)
        df_sparse.to_csv(sp_path,  index=False)
        print(f"\nSaved: {res_path}")
        print(f"Saved: {sp_path}")

    return df_results, df_sparse


if __name__ == "__main__":
    run(output_dir=Path(__file__).parent.parent / "output")