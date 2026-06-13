"""
slx_exposure.py
===============
Constructs the bank-network SLX exposure variable and augments the Favara &
Imbs (2015) credit regression.

  E_{c,t} = sum_{c'} w^{interstate}_{cc'} * D_{s(c'),t}

where D = Linter_bra (state-level interstate branching deregulation) and
w^{interstate} is the row-standardised interstate bank-network weight matrix
(cross-state connections only; Kelejian & Prucha 1998, Favara & Imbs 2015).

Regressions (two-way county + year FE, within estimator):
  (1) Base:    Dl_nloans_b ~ D + Favara-Imbs controls
  (2) SLX:     Dl_nloans_b ~ D + Favara-Imbs controls + E
  (3) Placebo: Dl_nloans_pl ~ D + placebo controls + E
     (Dl_nloans_pl = non-bank mortgage lending; should not respond to
      interstate branching deregulation if the exclusion restriction holds)

SE estimators applied to the SLX regression:
  (a) OLS (homoskedastic)
  (b) Conley (1999) spatial HAC under W_interstate
  (c) State-clustered (Favara-Imbs 2015 baseline)

Permutation test: 999 random permutations of the county identities of E,
RNG seed = 42. Empirical p-value = fraction |beta_E_perm| >= |beta_E_obs|.

Output: output/slx_exposure_results.csv
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse
import scipy.stats as st

sys.path.insert(0, str(Path(__file__).parents[1]))
import utils  # noqa: applies spreg Panel_FE_Error compatibility patch
from utils import row_standardize
from panel_data import (
    CREDIT_CONTROLS,
    PLACEBO_CONTROLS,
    load_panel_with_placebo,
    get_samples,
)
from w_variants import load_w_geo, load_bank_variants

ROOT        = Path(__file__).parents[2]
COUNTY_PATH = ROOT / "data" / "county_order_Wgeo.csv"
DV          = "Dl_nloans_b"
DV_PLACEBO  = "Dl_nloans_pl"
X_VARS      = ["Linter_bra"] + CREDIT_CONTROLS
PLACEBO_X_VARS = ["Linter_bra"] + PLACEBO_CONTROLS
PERM_SEED   = 42
N_PERM      = 999


# ==============================================================================
# Within-transformation (two-way FE)
# ==============================================================================

def _within(arr_TN):
    """
    Two-way within transform for a (T, N) array.
    Equivalent to: arr_it - county_mean_i - year_mean_t + grand_mean.
    """
    grand = float(arr_TN.mean())
    z     = arr_TN - arr_TN.mean(axis=0)[None, :]   # subtract county means
    z     = z + grand                                # restore grand mean
    z     = z - z.mean(axis=1, keepdims=True)        # subtract year means
    return z


def _within_3d(arr_TNK):
    """Apply the two-way within transform to each regressor column."""
    return np.stack([_within(arr_TNK[:, :, j]) for j in range(arr_TNK.shape[2])], axis=2)


# ==============================================================================
# OLS estimator
# ==============================================================================

def _ols(y_flat, X_flat):
    """
    OLS.  y_flat: (NT,), X_flat: (NT, k).
    Returns (beta (k,), u (NT,), XtX (k, k)).
    """
    XtX  = X_flat.T @ X_flat
    beta = np.linalg.solve(XtX, X_flat.T @ y_flat)
    u    = y_flat - X_flat @ beta
    return beta, u, XtX


# ==============================================================================
# Sandwich SE estimators
# ==============================================================================

def _se_ols(XtX, u, NT, k):
    """Homoskedastic OLS variance."""
    sigma2 = float(u @ u) / (NT - k)
    V      = sigma2 * np.linalg.inv(XtX)
    return np.sqrt(np.maximum(np.diag(V), 0.0))


def _meat_spatial(u_TN, X_TN, W_sp):
    """
    Conley (1999) spatial HAC meat for k regressors.

    B[j, l] = sum_t  score_j_t' W score_l_t
    where score_j_t[c] = u_{ct} * x_{j,ct}

    X_TN : (T, N, k) stacked regressors
    """
    T = u_TN.shape[0]
    k = X_TN.shape[2]
    B = np.zeros((k, k))
    for t in range(T):
        S_t  = u_TN[t, :, None] * X_TN[t, :, :]  # (N, k)
        WS_t = W_sp @ S_t                          # (N, k)
        B   += S_t.T @ WS_t
    return B


def _meat_clustered(u_TN, X_TN, county_states):
    """
    State-clustered sandwich meat for k regressors.
    Returns (B (k, k), G: number of clusters).
    """
    k      = X_TN.shape[2]
    scores = (u_TN[:, :, None] * X_TN).sum(axis=0)  # (N, k) -- summed over t
    B      = np.zeros((k, k))
    states = np.unique(county_states)
    for s in states:
        sv = scores[county_states == s].sum(axis=0)  # (k,)
        B += np.outer(sv, sv)
    return B, len(states)


def _sandwich_se(XtX, B, df):
    """sqrt(diag( (X'X)^{-1} B (X'X)^{-1} * df ))."""
    iXtX = np.linalg.inv(XtX)
    V    = iXtX @ B @ iXtX * df
    return np.sqrt(np.maximum(np.diag(V), 0.0))


def _se_conley(XtX, u_TN, X_TN, W_sp, NT, k):
    B  = _meat_spatial(u_TN, X_TN, W_sp)
    df = NT / (NT - k - 1)
    return _sandwich_se(XtX, B, df)


def _se_clustered(XtX, u_TN, X_TN, county_states):
    B, G = _meat_clustered(u_TN, X_TN, county_states)
    df   = G / (G - 1)
    return _sandwich_se(XtX, B, df)


# ==============================================================================
# Exposure matrix construction
# ==============================================================================

def _build_exposure(panel, county_order, W_is_all, YEARS):
    """
    Build D_all (N, T) and E_all (N, T) for all counties in county_order.

    D_all[i, t] = Linter_bra for county county_order[i] in year YEARS[t]
                  (state-level variable; filled from state lookup in panel)
    E_all[:, t] = W_is_all @ D_all[:, t]   (interstate network exposure)
    """
    N_all = len(county_order)
    T     = len(YEARS)

    # State-level deregulation by (state_fips2, year) from panel
    panel_tmp  = panel.assign(state_fips2=panel["fips5"].str[:2])
    state_d    = (panel_tmp.groupby(["state_fips2", "year"])["Linter_bra"]
                  .first()
                  .reset_index())
    state_map  = {(r["state_fips2"], r["year"]): float(r["Linter_bra"])
                  for _, r in state_d.iterrows()}

    D_all = np.zeros((N_all, T))
    for i, c in enumerate(county_order):
        sf = c[:2]
        for t_idx, yr in enumerate(YEARS):
            key = (sf, yr)
            if key in state_map:
                D_all[i, t_idx] = state_map[key]

    # E = W_interstate @ D (scipy sparse mat x dense matrix)
    E_all = np.column_stack([
        np.array(W_is_all @ D_all[:, t]).flatten() for t in range(T)
    ])
    return D_all, E_all


# ==============================================================================
# Sample construction
# ==============================================================================

def _build_sample(panel_sub, county_order, W_is_all, YEARS, year_pos,
                  D_all, E_all):
    """
    Build within-transformed regression arrays for one sample.

    Excludes counties with any NaN in either DV or its controls.
    Returns dict with all arrays needed for estimation and SE computation.
    """
    T = len(YEARS)

    filter_cols = [DV, DV_PLACEBO] + list(dict.fromkeys(X_VARS + PLACEBO_X_VARS))
    # County filter: present in sample, no NaN in either DV or control set
    any_nan = panel_sub.groupby("fips5")[filter_cols].apply(
        lambda g: g.isna().any().any()
    )
    sub_co  = set(panel_sub["fips5"].unique())
    usable  = [c for c in county_order
               if c in sub_co and not any_nan.get(c, True)]
    N       = len(usable)
    u_pos   = {c: i for i, c in enumerate(usable)}

    df = (panel_sub[panel_sub["fips5"].isin(set(usable))]
          .assign(t_idx=lambda d: d["year"].map(year_pos),
                  c_idx=lambda d: d["fips5"].map(u_pos))
          .sort_values(["t_idx", "c_idx"]))

    # (T, N) matrices
    y_TN  = df[DV].values.reshape(T, N)
    yp_TN = df[DV_PLACEBO].values.reshape(T, N)
    controls_TNK = df[CREDIT_CONTROLS].values.astype(float).reshape(
        T, N, len(CREDIT_CONTROLS)
    )
    placebo_controls_TNK = df[PLACEBO_CONTROLS].values.astype(float).reshape(
        T, N, len(PLACEBO_CONTROLS)
    )

    # D, E subsets via all-county precomputed arrays
    idx   = np.array([county_order.index(c) for c in usable])
    D_TN  = D_all[idx, :].T   # (T, N)
    E_TN  = E_all[idx, :].T   # (T, N)

    # W_interstate submatrix for Conley SE
    W_is_sub = row_standardize(W_is_all[idx, :][:, idx])

    # State integer IDs for clustering
    county_states = np.array([int(c[:2]) for c in usable])

    return dict(
        usable=usable, N=N, T=T, NT=N * T,
        y_TN=y_TN, yp_TN=yp_TN,
        controls_TNK=controls_TNK,
        placebo_controls_TNK=placebo_controls_TNK,
        D_TN=D_TN, E_TN=E_TN,
        W_is_sub=W_is_sub,
        county_states=county_states,
    )


# ==============================================================================
# Permutation test
# ==============================================================================

def _perm_test(y_TN, D_TN, E_TN, controls_TNK, beta_E_obs):
    """
    Permutation test for the SLX coefficient beta_E.

    Permutes county identities of E_TN (cols), leaving y and D fixed.
    Returns empirical p-value = fraction |beta_E_perm| >= |beta_E_obs|.
    """
    T, N = E_TN.shape
    rng  = np.random.default_rng(PERM_SEED)

    y_w = _within(y_TN).ravel()
    D_w = _within(D_TN).ravel()
    C_w = _within_3d(controls_TNK).reshape(T * N, controls_TNK.shape[2])

    count = 0
    for _ in range(N_PERM):
        perm   = rng.permutation(N)
        E_w    = _within(E_TN[:, perm]).ravel()
        Xp     = np.column_stack([D_w, C_w, E_w])
        beta_p, _, _ = _ols(y_w, Xp)
        if abs(beta_p[-1]) >= abs(beta_E_obs):
            count += 1
    return count / N_PERM


# ==============================================================================
# Main estimation for one sample
# ==============================================================================

def _estimate_sample(samp, sample_label):
    """
    Run all regressions for one sample dict (from _build_sample).
    Returns list of result-row dicts.
    """
    y_TN      = samp["y_TN"]
    yp_TN     = samp["yp_TN"]
    controls_TNK = samp["controls_TNK"]
    placebo_controls_TNK = samp["placebo_controls_TNK"]
    D_TN      = samp["D_TN"]
    E_TN      = samp["E_TN"]
    W_is_sub  = samp["W_is_sub"]
    c_states  = samp["county_states"]
    T, N      = y_TN.shape
    NT        = T * N
    k_base    = 1 + len(CREDIT_CONTROLS)
    k_slx     = k_base + 1
    k_placebo = 1 + len(PLACEBO_CONTROLS) + 1

    rows = []

    # -- (1) Base: y ~ D ------------------------------------------------------
    y_w    = _within(y_TN).ravel()
    D_w    = _within(D_TN).ravel()
    C_w    = _within_3d(controls_TNK).reshape(NT, len(CREDIT_CONTROLS))
    X_base = np.column_stack([D_w, C_w])
    beta_b, u_b, XtX_b = _ols(y_w, X_base)
    se_b_ols = _se_ols(XtX_b, u_b, NT, k_base)
    rows.append(_make_row(
        sample_label, "Base", DV, beta_b, se_b_ols, None, None, NT, N))

    # -- (2) SLX: y ~ D + E ---------------------------------------------------
    E_w    = _within(E_TN).ravel()
    X_slx  = np.column_stack([D_w, C_w, E_w])
    beta_s, u_s, XtX_s = _ols(y_w, X_slx)

    # Three SE types
    se_s_ols = _se_ols(XtX_s, u_s, NT, k_slx)

    # Reshape u and X for sandwich estimators: (T, N) and (T, N, k)
    u_TN  = u_s.reshape(T, N)
    # Re-within each for the sandwich (meat uses within-transformed values)
    D_3d_w = np.concatenate([
        _within(D_TN)[:, :, None],
        _within_3d(controls_TNK),
        _within(E_TN)[:, :, None],
    ], axis=2)

    se_s_con = _se_conley(XtX_s, u_TN, D_3d_w, W_is_sub, NT, k_slx)
    se_s_clu = _se_clustered(XtX_s, u_TN, D_3d_w, c_states)

    rows.append(_make_row(
        sample_label, "SLX-OLS",     DV, beta_s, se_s_ols, None, None, NT, N, e_idx=-1))
    rows.append(_make_row(
        sample_label, "SLX-Conley",  DV, beta_s, se_s_con, None, None, NT, N, e_idx=-1))
    rows.append(_make_row(
        sample_label, "SLX-Cluster", DV, beta_s, se_s_clu, None, None, NT, N, e_idx=-1))

    # -- (3) Placebo DV: yp ~ D + E -------------------------------------------
    yp_w    = _within(yp_TN).ravel()
    C_pl_w  = _within_3d(placebo_controls_TNK).reshape(NT, len(PLACEBO_CONTROLS))
    X_pl    = np.column_stack([D_w, C_pl_w, E_w])
    beta_pl, u_pl, XtX_pl = _ols(yp_w, X_pl)
    se_pl = _se_ols(XtX_pl, u_pl, NT, k_placebo)
    rows.append(_make_row(
        sample_label, "Placebo-OLS", DV_PLACEBO, beta_pl, se_pl, None, None, NT, N, e_idx=-1))

    # -- Permutation test on SLX beta_E ---------------------------------------
    beta_E_obs = beta_s[-1]
    p_perm     = _perm_test(y_TN, D_TN, E_TN, controls_TNK, beta_E_obs)

    # Attach permutation p-value to all SLX rows
    for row in rows:
        if row["spec"].startswith("SLX"):
            row["p_perm"] = round(p_perm, 4)

    return rows, beta_s, se_s_ols, se_s_con, se_s_clu, beta_pl, se_pl, p_perm


def _make_row(sample, spec, dv, beta, se, p_perm, extra, NT, N, e_idx=None):
    """Build a CSV result row dict."""
    k = len(beta)
    row = dict(
        sample   = sample,
        spec     = spec,
        dv       = dv,
        n_co     = N,
        n_obs    = NT,
        beta_D   = float(beta[0]),
        se_D     = float(se[0]),
        t_D      = float(beta[0] / se[0]) if se[0] > 0 else np.nan,
        p_D      = float(2 * st.norm.sf(abs(beta[0] / se[0]))) if se[0] > 0 else np.nan,
        beta_E   = float(beta[e_idx]) if e_idx is not None else np.nan,
        se_E     = float(se[e_idx])   if e_idx is not None else np.nan,
        t_E      = float(beta[e_idx] / se[e_idx]) if (e_idx is not None and se[e_idx] > 0) else np.nan,
        p_E      = float(2 * st.norm.sf(abs(beta[e_idx] / se[e_idx]))) if (e_idx is not None and se[e_idx] > 0) else np.nan,
        p_perm   = np.nan,
    )
    return row


def _stars(p):
    if np.isnan(p): return ""
    return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))


# ==============================================================================
# Public entry point
# ==============================================================================

def run(output_dir=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # -- Load shared inputs ---------------------------------------------------
    co_df        = pd.read_csv(COUNTY_PATH, dtype={"fips5": str})
    county_order = co_df["fips5"].str.zfill(5).tolist()

    W_geo_all, gal_order = load_w_geo(county_order)
    assert gal_order == county_order, "GAL order mismatch"

    bank_variants = load_bank_variants(county_order, W_geo_all=W_geo_all)
    W_is_all      = bank_variants["W_bank_interstate"]

    panel    = load_panel_with_placebo()
    panel["fips5"] = panel["fips5"].astype(str).str.zfill(5)
    YEARS    = sorted(panel["year"].unique())
    T        = len(YEARS)
    year_pos = {yr: i for i, yr in enumerate(YEARS)}

    # -- Precompute D and E for all counties ----------------------------------
    print("Computing interstate SLX exposure E_{c,t} ...", flush=True)
    D_all, E_all = _build_exposure(panel, county_order, W_is_all, YEARS)

    # -- Estimate for each sample ---------------------------------------------
    all_rows = []
    for sample_label, panel_sub in get_samples(panel):
        print(f"\n{'='*70}")
        print(f"Sample: {sample_label}  (N counties TBD)")
        print(f"{'='*70}")
        try:
            samp = _build_sample(
                panel_sub, county_order, W_is_all, YEARS, year_pos, D_all, E_all)
            print(f"  Usable counties: {samp['N']}  |  NT = {samp['NT']}", flush=True)

            print("  Running base regression ...", flush=True)
            rows, beta_s, se_ols, se_con, se_clu, beta_pl, se_pl, p_perm = \
                _estimate_sample(samp, sample_label)
            all_rows.extend(rows)

            # -- Print summary table ------------------------------------------
            W_print = 70
            print()
            print(f"{'Spec':<18} {'DV':<18} {'beta_D':>8} {'SE':>7}  "
                  f"{'beta_E':>8} {'SE':>7}  {'p_perm':>7}")
            print("-" * W_print)
            for row in rows:
                bd = f"{row['beta_D']:>8.4f}{_stars(row['p_D'])}"
                be = f"{row['beta_E']:>8.4f}{_stars(row['p_E'])}" \
                     if not np.isnan(row['beta_E']) else f"{'---':>11}"
                se_d = f"{row['se_D']:>7.4f}"
                se_e = f"{row['se_E']:>7.4f}" \
                       if not np.isnan(row['se_E']) else f"{'---':>7}"
                pp   = f"{row['p_perm']:>7.4f}" \
                       if not np.isnan(row['p_perm']) else f"{'':>7}"
                print(f"{row['spec']:<18} {row['dv']:<18} {bd} {se_d}  {be} {se_e}  {pp}")
            print()
            print(f"Permutation test: |beta_E_perm| >= |{beta_s[-1]:.4f}| "
                  f"in {round(p_perm * N_PERM)}/{N_PERM} reps  ->  p_perm = {p_perm:.4f}")
            print("Significance: *** p<0.01  ** p<0.05  * p<0.10")

        except Exception as exc:
            print(f"  [SKIP] {sample_label}: {exc}")

    # -- Save CSV -------------------------------------------------------------
    if output_dir is not None and all_rows:
        cols = ["sample", "spec", "dv", "n_co", "n_obs",
                "beta_D", "se_D", "t_D", "p_D",
                "beta_E", "se_E", "t_E", "p_E", "p_perm"]
        df_out = pd.DataFrame(all_rows)[cols]
        out_path = output_dir / "slx_exposure_results.csv"
        df_out.to_csv(out_path, index=False)
        print(f"\nSaved slx_exposure_results.csv to {output_dir}")

    return all_rows


if __name__ == "__main__":
    run(Path(__file__).parents[2] / "output")
