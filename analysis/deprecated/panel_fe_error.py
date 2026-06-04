"""
Panel_FE_Error (ML) — W_geo vs W_bank, two-way fixed effects

Data format required by spreg.Panel_FE_Error
---------------------------------------------
  y : (N*T, 1)     -- long format, time-major: y[0:N]=T0, y[N:2N]=T1, ...
  x : (N*T, k)     -- same row ordering; k = 1 (Linter_bra) + 10 (year dummies)
  w : cross-sectional n x n PySAL W object (NOT block-diagonal)

Two-way FE achieved by:
  - County FE: removed internally by Panel_FE_Error within-transformation
  - Year FE:   explicit year dummies yr1996-yr2005 in X (base = 1995)

Result attribute layout:
  betas  : (k+1, 1)  -- [beta_1, ..., beta_k, lambda]  (lambda is LAST)
  lam    : float      -- same value as betas[-1, 0]
  std_err: (k+1,)     -- standard errors including SE(lambda) at end
  z_stat : list of (k+1) (z, p) tuples
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
import scipy.sparse
import spreg

import utils  # noqa: applies spreg Panel_FE_Error compatibility patch
from utils import gal_to_W, row_standardize, sparse_to_pysal_w


ROOT        = Path(__file__).parent.parent
PANEL_PATH  = ROOT / "data" / "estimation_panel.csv"
GAL_PATH    = ROOT / "data" / "W_geo_queen.gal"
COUNTY_PATH = ROOT / "data" / "county_order_Wgeo.csv"
WBANK_PATH  = ROOT / "data" / "W_bank_avg.npz"


# ── Build arrays for Panel_FE_Error ──────────────────────────────────────────
def build_arrays(panel, county_order, W_geo_all, W_bank_all, YEARS, year_pos,
                 N_ALL, panel_sub, sample_label):
    """
    Build y and x in LONG format for Panel_FE_Error.

    Long format required by check_panel (wide-format strips constant columns):
      y : (N*T, 1)  -- time-major: y[0:N]=T0, y[N:2N]=T1, ...
      x : (N*T, k)  -- same row ordering, k columns (one per variable)
      w : n x n cross-sectional PySAL W (NOT block-diagonal)

    Counties within each time block are sorted in W county_order row order.
    """
    T = len(YEARS)
    na_all   = panel_sub.groupby("fips5")["Linter_ela"].apply(lambda s: s.isna().all())
    sub_co   = set(panel_sub["fips5"].unique())
    usable   = [c for c in county_order if c in sub_co and not na_all.get(c, True)]
    N        = len(usable)
    usable_pos = {c: i for i, c in enumerate(usable)}

    df = (panel_sub[panel_sub["fips5"].isin(set(usable))]
          .assign(t_idx=lambda d: d["year"].map(year_pos),
                  c_idx=lambda d: d["fips5"].map(usable_pos))
          .sort_values(["t_idx", "c_idx"]))

    y_long = df["Linter_ela"].values.reshape(-1, 1)

    dummy_years  = YEARS[1:]
    t_idx_vec    = df["t_idx"].values
    year_dummies = np.column_stack([
        (t_idx_vec == year_pos[yr]).astype(np.float64)
        for yr in dummy_years
    ])

    x_long = np.hstack([
        df["Linter_bra"].values.reshape(-1, 1),
        year_dummies
    ])

    assert not np.isnan(y_long).any(), "NaN in y"
    assert y_long.shape == (N * T, 1),  f"y wrong shape: {y_long.shape}"
    assert x_long.shape == (N * T, 11), f"x wrong shape: {x_long.shape}"

    idx        = np.array([county_order.index(c) for c in usable])
    W_geo_sub  = row_standardize(W_geo_all[idx, :][:, idx])
    W_bank_sub = row_standardize(W_bank_all[idx, :][:, idx])

    return y_long, x_long, sparse_to_pysal_w(W_geo_sub), sparse_to_pysal_w(W_bank_sub), N


# ── Run Panel_FE_Error ────────────────────────────────────────────────────────
def run_panel_fe(y, x, w_pysal, w_label, ds_label, YEARS):
    nx = ["Linter_bra"] + [f"yr{yr}" for yr in YEARS[1:]]
    return spreg.Panel_FE_Error(
        y, x, w_pysal,
        name_y="Linter_ela",
        name_x=nx,
        name_w=w_label,
        name_ds=ds_label
    )


def extract(res, N, T):
    """
    Extract key results from Panel_FE_Error.
    betas layout: [beta_Linter_bra, yr1996, ..., yr2005, lambda]
    std_err/z_stat: same length, lambda is last.
    """
    beta = float(res.betas[0, 0])
    lam  = float(res.lam)
    se_b = float(res.std_err[0])
    se_l = float(res.std_err[-1])
    z_b, p_b = res.z_stat[0]
    z_l, p_l = res.z_stat[-1]
    return dict(beta=beta, se_beta=se_b, z_beta=z_b, p_beta=p_b,
                lam=lam,   se_lam=se_l,  z_lam=z_l,  p_lam=p_l,
                n_co=N,    n_obs=N * T)


def run(output_dir=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load shared inputs ────────────────────────────────────────────────────
    co_df        = pd.read_csv(COUNTY_PATH, dtype={"fips5": str})
    county_order = co_df["fips5"].str.zfill(5).tolist()
    N_ALL        = len(county_order)

    W_geo_all, gal_order = gal_to_W(GAL_PATH, county_order)
    assert gal_order == county_order, "GAL order != county_order_Wgeo.csv"

    W_bank_all = row_standardize(scipy.sparse.load_npz(WBANK_PATH))

    panel = pd.read_csv(PANEL_PATH)
    panel["fips5"] = panel["fips5"].astype(str).str.zfill(5)
    YEARS    = sorted(panel["year"].unique())
    T        = len(YEARS)
    year_pos = {yr: i for i, yr in enumerate(YEARS)}

    # ── Estimate ──────────────────────────────────────────────────────────────
    results = {}

    y_f, x_f, w_geo_f, w_bank_f, N_f = build_arrays(
        panel, county_order, W_geo_all, W_bank_all, YEARS, year_pos, N_ALL,
        panel, "Full sample")
    results["FE W_geo (full)"]  = extract(
        run_panel_fe(y_f, x_f, w_geo_f,  "W_geo",  "Full", YEARS), N_f, T)
    results["FE W_bank (full)"] = extract(
        run_panel_fe(y_f, x_f, w_bank_f, "W_bank", "Full", YEARS), N_f, T)

    panel_nb = panel[panel["border"] == 0].copy()
    y_nb, x_nb, w_geo_nb, w_bank_nb, N_nb = build_arrays(
        panel, county_order, W_geo_all, W_bank_all, YEARS, year_pos, N_ALL,
        panel_nb, "Non-border")
    results["FE W_geo (non-border)"]  = extract(
        run_panel_fe(y_nb, x_nb, w_geo_nb,  "W_geo",  "NB", YEARS), N_nb, T)
    results["FE W_bank (non-border)"] = extract(
        run_panel_fe(y_nb, x_nb, w_bank_nb, "W_bank", "NB", YEARS), N_nb, T)

    # ── Save results to CSV ───────────────────────────────────────────────────
    if output_dir is not None:
        rows = []
        for model_name, r in results.items():
            rows.append(dict(
                model=model_name,
                beta=r["beta"],    se_beta=r["se_beta"], z_beta=r["z_beta"], p_beta=r["p_beta"],
                lam=r["lam"],      se_lam=r["se_lam"],   z_lam=r["z_lam"],   p_lam=r["p_lam"],
                n_co=r["n_co"],    n_obs=r["n_obs"],
            ))
        pd.DataFrame(rows).to_csv(output_dir / "panel_fe_results.csv", index=False)

    return results


if __name__ == '__main__':
    run(Path(__file__).parent.parent / 'output')
