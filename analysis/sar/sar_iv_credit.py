"""
sar_iv_credit.py
================
IV-SAR estimation of credit growth following Kelejian & Prucha (1998).

Model:
  y_it = rho (W y)_it + beta * Linter_bra_it + alpha_i + tau_t + xi_it

The spatial lag (Wy)_it is endogenous. Instruments are spatial lags of the
exogenous regressor:
  q1 = W * Linter_bra  (WD)
  q2 = W * W * Linter_bra  (W2D)

Estimation proceeds via 2SLS on within-transformed data:
  Step 1 — within-transform all variables (two-way FE)
  Step 2 — first-stage OLS: z_tilde ~ [x_tilde, q1_tilde, q2_tilde]
  Step 3 — second-stage OLS: y_tilde ~ [z_hat_tilde, x_tilde]
  Step 4 — 2SLS SE using ORIGINAL spatial lag for residuals (not z_hat)
  Step 5 — state-clustered sandwich correction
  Step 6 — Moran's I on SAR residuals under W_bank by year

Outputs
-------
  output/sar_iv_results.csv
  output/sar_iv_comparison.png
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.stats as st

sys.path.insert(0, str(Path(__file__).parents[1]))
import utils  # noqa
from utils import row_standardize, sparse_to_pysal_w
from panel_data import load_panel_with_credit
from w_variants import load_w_geo, load_bank_variants

# esda is used only for Moran's I
try:
    from esda.moran import Moran as _Moran
    _HAS_ESDA = True
except ImportError:
    _HAS_ESDA = False

ROOT        = Path(__file__).parents[2]
COUNTY_PATH = ROOT / "data" / "county_order_Wgeo.csv"
SAR_CSV     = ROOT / "output" / "sar_robustness_credit.csv"
CONLEY_CSV  = ROOT / "output" / "conley_se_comparison.csv"
DV          = "Dl_nloans_b"
MORAN_PERMS = 999


# ══════════════════════════════════════════════════════════════════════════════
# Within transformation (identical to conley_se_comparison.py)
# ══════════════════════════════════════════════════════════════════════════════

def two_way_within(arr_TN):
    """
    Two-way (county + year) within transformation for a (T, N) array.

    arr - county_mean[None,:] - year_mean[:,None] + grand_mean
    Equivalent step-form: county-demean → add grand mean → year-demean.
    """
    county_mean = arr_TN.mean(axis=0)
    grand_mean  = float(arr_TN.mean())
    z = arr_TN - county_mean[None, :]
    z = z + grand_mean
    z = z - z.mean(axis=1, keepdims=True)
    return z


# ══════════════════════════════════════════════════════════════════════════════
# OLS helper
# ══════════════════════════════════════════════════════════════════════════════

def ols_fit(X, y):
    """
    OLS: delta = (X'X)^{-1} X'y.
    X : (n, k) np.ndarray
    y : (n,)   np.ndarray
    Returns (delta (k,), y_hat (n,), resid (n,), XtX_inv (k,k)).
    """
    XtX     = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    delta   = XtX_inv @ (X.T @ y)
    y_hat   = X @ delta
    resid   = y - y_hat
    return delta, y_hat, resid, XtX_inv


# ══════════════════════════════════════════════════════════════════════════════
# First-stage F-test on excluded instruments
# ══════════════════════════════════════════════════════════════════════════════

def f_test_excluded(z_tilde, X_full, n_excl):
    """
    F-test for joint significance of the last n_excl columns in X_full.

    Compares:
      unrestricted: z_tilde ~ X_full        (k columns)
      restricted:   z_tilde ~ X_full[:,:-n_excl]  (k - n_excl columns)

    Returns (F_stat, df1, df2, R2_unr, R2_restr).
    """
    NT = len(z_tilde)
    k  = X_full.shape[1]

    # Unrestricted
    _, _, resid_u, _ = ols_fit(X_full, z_tilde)
    rss_u = float(resid_u @ resid_u)

    # Restricted (drop last n_excl columns = excluded instruments)
    X_restr = X_full[:, :-n_excl]
    _, _, resid_r, _ = ols_fit(X_restr, z_tilde)
    rss_r = float(resid_r @ resid_r)

    df1 = n_excl
    df2 = NT - k
    F   = ((rss_r - rss_u) / df1) / (rss_u / df2)

    # R² for reference
    tss   = float(((z_tilde - z_tilde.mean()) ** 2).sum())
    R2_u  = 1 - rss_u / tss if tss > 0 else 0.0
    R2_r  = 1 - rss_r / tss if tss > 0 else 0.0

    return float(F), df1, df2, float(R2_u), float(R2_r)


# ══════════════════════════════════════════════════════════════════════════════
# 2SLS standard errors with state clustering
# ══════════════════════════════════════════════════════════════════════════════

def cluster_se_2sls(Z_hat, xi, state_idx_flat, G, k_params):
    """
    State-clustered sandwich variance matrix for the 2SLS estimator.

    Var_cluster = (Z_hat'Z_hat)^{-1} B_cluster (Z_hat'Z_hat)^{-1} * correction

    B_cluster = sum_s  score_s score_s'
    score_s   = Z_hat_s' xi_s   (k-vector, sum over all obs in state s)

    correction = G/(G-1) * (NT-1)/(NT-k_params)

    Parameters
    ----------
    Z_hat         : (NT, k) matrix of instrumented regressors
    xi            : (NT,) 2SLS residuals (computed from ORIGINAL spatial lag)
    state_idx_flat: (NT,) integer state for each observation
    G             : number of states
    k_params      : number of estimated parameters (= 2: rho, beta)

    Returns
    -------
    vcv  : (k, k) variance-covariance matrix
    se   : (k,) standard errors
    """
    NT  = len(xi)
    ZtZ = Z_hat.T @ Z_hat
    ZtZ_inv = np.linalg.inv(ZtZ)

    # ── Cluster meat B = sum_s score_s score_s' ────────────────────────────
    states = np.unique(state_idx_flat)
    B = np.zeros((k_params, k_params))
    for s in states:
        mask    = state_idx_flat == s
        score_s = Z_hat[mask].T @ xi[mask]   # (k,) vector
        B      += np.outer(score_s, score_s)

    # ── Small-sample correction ───────────────────────────────────────────
    corr = (G / (G - 1)) * ((NT - 1) / (NT - k_params))

    vcv = ZtZ_inv @ B @ ZtZ_inv * corr
    se  = np.sqrt(np.maximum(np.diag(vcv), 0.0))
    return vcv, se


# ══════════════════════════════════════════════════════════════════════════════
# OLS state-clustered SE (for comparison row)
# ══════════════════════════════════════════════════════════════════════════════

def ols_cluster_se_scalar(x_flat, y_flat, state_idx_flat, G):
    """
    State-clustered SE for scalar OLS (k=1, just Linter_bra).
    Returns (beta, se).
    """
    XtX   = float(x_flat @ x_flat)
    beta  = float(x_flat @ y_flat) / XtX
    u     = y_flat - beta * x_flat
    NT    = len(y_flat)
    k     = 1

    states = np.unique(state_idx_flat)
    B = 0.0
    for s in states:
        mask    = state_idx_flat == s
        score_s = float((x_flat[mask] * u[mask]).sum())
        B      += score_s ** 2

    corr = (G / (G - 1)) * ((NT - 1) / (NT - k))
    var  = (B / (XtX ** 2)) * corr
    return beta, float(np.sqrt(max(var, 0.0)))


# ══════════════════════════════════════════════════════════════════════════════
# Moran's I on SAR residuals year by year
# ══════════════════════════════════════════════════════════════════════════════

def moran_i_sar_resid(xi_TN, W_bank_sub_rs, YEARS):
    """
    Compute Moran's I on within-transformed SAR residuals year by year,
    weighted by W_bank.

    xi_TN       : (T, N) array of 2SLS within-residuals (time-major)
    W_bank_sub_rs: (N, N) row-standardised scipy sparse W_bank submatrix
    YEARS       : list of year labels, length T

    Returns list of dicts {year, moran_I, z_score, p_value, significant}.
    """
    if not _HAS_ESDA:
        print("  [WARN] esda not available -- skipping Moran's I")
        return []

    T, N = xi_TN.shape

    # Identify islands (zero-row counties in W_bank_sub_rs)
    row_sums     = np.array(W_bank_sub_rs.sum(axis=1)).flatten()
    non_island   = np.where(row_sums > 0)[0]
    W_ni         = W_bank_sub_rs[non_island, :][:, non_island]
    W_ni         = row_standardize(W_ni)   # re-standardize after removing islands
    w_pysal      = sparse_to_pysal_w(W_ni)

    results = []
    for t, year in enumerate(YEARS):
        xi_t = xi_TN[t, non_island]      # residuals for non-island counties

        mi = _Moran(xi_t, w_pysal, permutations=MORAN_PERMS)
        sig = mi.p_sim < 0.05 if MORAN_PERMS > 0 else mi.p_norm < 0.05
        p_val = mi.p_sim if MORAN_PERMS > 0 else mi.p_norm
        results.append({
            "year": int(year),
            "moran_I": float(mi.I),
            "expected_I": float(mi.EI),
            "z_score": float(mi.z_sim if MORAN_PERMS > 0 else mi.z_norm),
            "p_value": float(p_val),
            "significant": bool(sig),
            "n_non_island": int(len(non_island)),
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Sample builder
# ══════════════════════════════════════════════════════════════════════════════

def build_sample(panel_sub, county_order, W_geo_all, W_bank_all, YEARS, year_pos,
                 sample_label):
    """
    Build all arrays needed for IV-SAR, using the same county filter as
    panel_fe_credit.build_arrays (any-NaN in Dl_nloans_b is dropped).

    Returns a dict with:
      N, T, NT, beta, XtX — basic stats
      y_TN, d_TN           — raw (T,N) arrays for credit growth, Linter_bra
      county_states        — (N,) state ids
      state_idx_flat       — (NT,) state ids for each obs (time-major)
      W_geo_sub, W_bank_sub — (N,N) row-standardised sparse submatrices
      usable               — list of fips5 strings in county order
    """
    T = len(YEARS)

    # ── County filter (same as panel_data.py) ─────────────────────────
    any_nan    = panel_sub.groupby("fips5")[DV].apply(lambda s: s.isna().any())
    sub_co     = set(panel_sub["fips5"].unique())
    usable     = [c for c in county_order
                  if c in sub_co and not any_nan.get(c, True)]
    N          = len(usable)
    usable_pos = {c: i for i, c in enumerate(usable)}

    df = (panel_sub[panel_sub["fips5"].isin(set(usable))]
          .assign(t_idx=lambda d: d["year"].map(year_pos),
                  c_idx=lambda d: d["fips5"].map(usable_pos))
          .sort_values(["t_idx", "c_idx"]))

    y_flat  = df[DV].values.astype(np.float64)
    d_flat  = df["Linter_bra"].values.astype(np.float64)

    assert not np.isnan(y_flat).any()
    assert not np.isnan(d_flat).any()
    assert len(y_flat) == N * T

    # ── State mapping ─────────────────────────────────────────────────────
    county_states = (
        df.groupby("c_idx")["state_n"].first().sort_index().values.astype(int)
    )
    state_idx_flat = np.tile(county_states, T)   # (NT,) time-major

    # ── W submatrices (row-standardised, same as panel_data.py) ──────
    idx        = np.array([county_order.index(c) for c in usable])
    W_geo_sub  = row_standardize(W_geo_all[idx, :][:, idx])
    W_bank_sub = row_standardize(W_bank_all[idx, :][:, idx])

    return dict(
        N=N, T=T, NT=N * T,
        y_TN=y_flat.reshape(T, N),
        d_TN=d_flat.reshape(T, N),
        county_states=county_states,
        state_idx_flat=state_idx_flat,
        W_geo_sub=W_geo_sub,
        W_bank_sub=W_bank_sub,
        YEARS=YEARS,
        usable=usable,
    )


# ══════════════════════════════════════════════════════════════════════════════
# IV-SAR estimator for one (sample, W) pair
# ══════════════════════════════════════════════════════════════════════════════

def run_iv_sar(s, W_label, sample_label, verbose=True):
    """
    Run IV-SAR for one sample dict and weight choice ('W_geo' or 'W_bank').

    Steps:
      1. Select W = W_geo_sub or W_bank_sub
      2. Compute spatial lags: z = W@y, q1 = W@d, q2 = W@q1
      3. Two-way within-transform all variables
      4. First stage: z_tilde ~ [x_tilde, q1_tilde, q2_tilde]
      5. Second stage: y_tilde ~ [z_hat_tilde, x_tilde]
      6. 2SLS residuals from ORIGINAL z_tilde (not z_hat)
      7. State-clustered sandwich SE
      8. Moran's I on residuals under W_bank

    Returns result dict.
    """
    N, T, NT = s["N"], s["T"], s["NT"]
    W_sub = s["W_geo_sub"] if W_label == "W_geo" else s["W_bank_sub"]
    y_TN  = s["y_TN"]
    d_TN  = s["d_TN"]

    if verbose:
        print(f"\n  [{sample_label} | {W_label}] N={N} T={T} NT={NT:,}")

    # ── Step 1: Compute spatial lags ────────────────────────────────────────
    # z = W @ y (endogenous spatial lag of credit growth)
    # q1 = W @ d (WD: first instrument)
    # q2 = W @ q1 (W2D: second instrument)
    z_TN  = np.zeros((T, N))
    q1_TN = np.zeros((T, N))
    q2_TN = np.zeros((T, N))
    for t in range(T):
        z_TN[t]  = W_sub @ y_TN[t]
        q1_TN[t] = W_sub @ d_TN[t]
        q2_TN[t] = W_sub @ q1_TN[t]

    # ── Step 2: Two-way within transformation ───────────────────────────────
    y_tilde  = two_way_within(y_TN)   # (T, N)
    z_tilde  = two_way_within(z_TN)
    x_tilde  = two_way_within(d_TN)
    q1_tilde = two_way_within(q1_TN)
    q2_tilde = two_way_within(q2_TN)

    # Flatten to (NT,) in time-major order
    y_f  = y_tilde.flatten()
    z_f  = z_tilde.flatten()
    x_f  = x_tilde.flatten()
    q1_f = q1_tilde.flatten()
    q2_f = q2_tilde.flatten()

    # ── Step 3: OLS baseline (for comparison table row) ─────────────────────
    G   = len(np.unique(s["state_idx_flat"]))
    ols_beta, ols_se = ols_cluster_se_scalar(
        x_f, y_f, s["state_idx_flat"], G
    )

    # ── Step 4: First stage ─────────────────────────────────────────────────
    # z_tilde ~ x_tilde + q1_tilde + q2_tilde
    # Excluded instruments: q1_tilde, q2_tilde (last 2 columns)
    X_fs = np.column_stack([x_f, q1_f, q2_f])   # (NT, 3)
    delta_fs, z_hat_f, resid_fs, _ = ols_fit(X_fs, z_f)
    F_stat, df1, df2, R2_unr, R2_restr = f_test_excluded(z_f, X_fs, n_excl=2)

    if verbose:
        print(f"  First stage: alpha_WD={delta_fs[1]:.4f}  "
              f"alpha_W2D={delta_fs[2]:.4f}  "
              f"R2={R2_unr:.4f}  F(2,{df2})={F_stat:.2f}")
        if F_stat < 10:
            print(f"  [WEAK INSTRUMENT] F={F_stat:.2f} < 10 — interpret IV-SAR"
                  f" under {W_label} with caution")

    # ── Step 5: Second stage ─────────────────────────────────────────────────
    # y_tilde ~ z_hat_tilde + x_tilde
    Z_hat = np.column_stack([z_hat_f, x_f])   # (NT, 2) with instrumented z
    delta_ss, _, _, ZtZ_inv = ols_fit(Z_hat, y_f)
    rho_hat  = float(delta_ss[0])    # spatial autoregressive parameter
    beta_hat = float(delta_ss[1])    # deregulation effect

    # ── Step 6: 2SLS residuals (use ORIGINAL z, not z_hat) ──────────────────
    # xi = y_tilde - z_tilde * rho - x_tilde * beta   [CORRECT 2SLS residual]
    Z_orig = np.column_stack([z_f, x_f])   # (NT, 2) with original (endogenous) z
    xi_f   = y_f - Z_orig @ delta_ss

    # ── Step 7: State-clustered 2SLS sandwich SE ────────────────────────────
    # Bread uses Z_hat (instrumented); meat uses xi from original Z
    vcv, se_2sls = cluster_se_2sls(Z_hat, xi_f, s["state_idx_flat"], G, k_params=2)
    rho_se   = float(se_2sls[0])
    beta_se  = float(se_2sls[1])

    if verbose:
        z_crit = st.norm.ppf(0.975)
        print(f"  Second stage: rho={rho_hat:.4f}({rho_se:.4f})  "
              f"beta={beta_hat:.4f}({beta_se:.4f})")

    # ── Step 8: Moran's I on SAR residuals under W_bank ─────────────────────
    if verbose:
        print(f"  Moran's I on residuals under W_bank ({MORAN_PERMS} perms) ...",
              flush=True)

    xi_TN   = xi_f.reshape(T, N)
    moran_rows = moran_i_sar_resid(xi_TN, s["W_bank_sub"], s["YEARS"])
    moran_mean = float(np.mean([r["moran_I"] for r in moran_rows])) \
                 if moran_rows else float("nan")

    if verbose and moran_rows:
        for r in moran_rows:
            sig = ("***" if r["p_value"] < 0.01
                   else "**" if r["p_value"] < 0.05
                   else "*"  if r["p_value"] < 0.10 else "")
            print(f"    {r['year']}  I={r['moran_I']:+.4f}  "
                  f"z={r['z_score']:+.2f}  p={r['p_value']:.3f}  {sig}")
        print(f"  Mean Moran I (W_bank): {moran_mean:.4f}")

    z_crit = st.norm.ppf(0.975)
    return dict(
        sample          = sample_label,
        W               = W_label,
        # OLS baseline (same sample, for comparison)
        ols_beta        = ols_beta,
        ols_se          = ols_se,
        # First stage
        first_stage_F   = float(F_stat),
        first_stage_df1 = int(df1),
        first_stage_df2 = int(df2),
        first_stage_R2  = float(R2_unr),
        alpha_WD        = float(delta_fs[1]),
        alpha_W2D       = float(delta_fs[2]),
        # Second stage
        rho             = rho_hat,
        rho_se          = rho_se,
        rho_ci_lower    = rho_hat - z_crit * rho_se,
        rho_ci_upper    = rho_hat + z_crit * rho_se,
        beta            = beta_hat,
        beta_se         = beta_se,
        beta_ci_lower   = beta_hat - z_crit * beta_se,
        beta_ci_upper   = beta_hat + z_crit * beta_se,
        # Diagnostics
        residual_moran_i_mean = moran_mean,
        moran_rows      = moran_rows,
        G               = G,
        N_counties      = N,
        N_obs           = NT,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Comparison table
# ══════════════════════════════════════════════════════════════════════════════

def print_comparison_table(results_by_sample, sar_df, conley_df_csv):
    """
    Print formatted comparison table for all estimators.
    """
    z_crit = st.norm.ppf(0.975)

    # ── Read ML-SAR results from sar_robustness_credit.csv ─────────────────
    def ml_sar_row(sample_label, w_label):
        slabel = "Full" if "Full" in sample_label else "Border"
        wlabel = w_label
        row = sar_df[(sar_df["sample"] == slabel) &
                     (sar_df["w_matrix"] == wlabel)]
        if row.empty:
            return None
        r     = row.iloc[0]
        beta  = float(r["beta_D"])
        se    = float(r["beta_D_se"])
        rho   = float(r["rho"])
        rho_se = float(r["rho_se"])
        return dict(rho=rho, rho_se=rho_se,
                    beta=beta, se=se,
                    ci_lower=beta - z_crit * se,
                    ci_upper=beta + z_crit * se)

    # ── Read OLS from conley_se_comparison.csv ─────────────────────────────
    def ols_row_from_conley(sample_label):
        slabel = "Full" if "Full" in sample_label else "Border"
        row = conley_df_csv[
            (conley_df_csv["sample"] == slabel) &
            (conley_df_csv["estimator"] == "State clustering (Favara-Imbs)")
        ]
        if row.empty:
            return None
        r = row.iloc[0]
        return dict(beta=float(r["beta"]), se=float(r["se"]),
                    ci_lower=float(r["ci_lower"]), ci_upper=float(r["ci_upper"]))

    OLS_MORAN_BASELINE = 0.170

    W  = 80
    for sample_label, ivres in results_by_sample.items():
        print()
        print("=" * W)
        print(f"{sample_label}")
        print("-" * W)
        hdr = f"{'':28}  {'rho':>8}  {'beta':>8}  {'95% CI beta':>20}  {'F-stat':>8}"
        print(hdr)
        print("-" * W)

        # OLS row
        ols = ols_row_from_conley(sample_label)
        if ols:
            ci = f"[{ols['ci_lower']:+.4f}, {ols['ci_upper']:+.4f}]"
            print(f"  {'OLS (Favara-Imbs)':<26}  {'---':>8}  "
                  f"{ols['beta']:>8.4f}  {ci:>20}  {'---':>8}")

        # ML-SAR rows
        for wl in ["W_geo", "W_bank"]:
            ml = ml_sar_row(sample_label, wl)
            if ml:
                ci = f"[{ml['ci_lower']:+.4f}, {ml['ci_upper']:+.4f}]"
                print(f"  {'ML SAR ' + wl:<26}  {ml['rho']:>8.4f}  "
                      f"{ml['beta']:>8.4f}  {ci:>20}  {'---':>8}")

        # IV-SAR rows
        for wl in ["W_geo", "W_bank"]:
            r = ivres.get((sample_label, wl))
            if r is None:
                continue
            ci    = f"[{r['beta_ci_lower']:+.4f}, {r['beta_ci_upper']:+.4f}]"
            f_str = f"{r['first_stage_F']:>8.2f}"
            if r["first_stage_F"] < 10:
                f_str += " (!)"
            print(f"  {'IV-SAR ' + wl:<26}  {r['rho']:>8.4f}  "
                  f"{r['beta']:>8.4f}  {ci:>20}  {f_str}")

        print("-" * W)
        print(f"  Residual Moran I under W_bank:")
        print(f"    OLS residuals (baseline):    {OLS_MORAN_BASELINE:.3f}")
        for wl in ["W_geo", "W_bank"]:
            r = ivres.get((sample_label, wl))
            if r is None:
                continue
            mi    = r["residual_moran_i_mean"]
            delta = mi - OLS_MORAN_BASELINE
            tag   = "REDUCED" if delta < -0.02 else ("NOTE: similar to OLS" if abs(delta) < 0.02 else "INCREASED")
            print(f"    IV-SAR {wl} resid:    {mi:.3f}  (delta={delta:+.3f}, {tag})")
        print("=" * W)

    print()
    print("Note: (*!) = weak instrument (F < 10), IV-SAR estimates may be unreliable.")


# ══════════════════════════════════════════════════════════════════════════════
# Publication-quality comparison plot
# ══════════════════════════════════════════════════════════════════════════════

_EST_ORDER = [
    "OLS",
    "ML SAR W_geo",
    "ML SAR W_bank",
    "IV-SAR W_geo",
    "IV-SAR W_bank",
]
_EST_COLORS = {
    "OLS":           "#2166ac",
    "ML SAR W_geo":  "#4dac26",
    "ML SAR W_bank": "#b8e186",
    "IV-SAR W_geo":  "#d01c8b",
    "IV-SAR W_bank": "#e66101",
}
_EST_MARKERS = {
    "OLS": "D", "ML SAR W_geo": "o", "ML SAR W_bank": "s",
    "IV-SAR W_geo": "^", "IV-SAR W_bank": "v",
}


def make_comparison_plot(results_by_sample, sar_df, conley_df_csv, out_path):
    z_crit = st.norm.ppf(0.975)

    def _row(sample_label, est_label):
        """Return (beta, lo, hi) or None."""
        sl = "Full" if "Full" in sample_label else "Border"

        if est_label == "OLS":
            r = conley_df_csv[
                (conley_df_csv["sample"] == sl) &
                (conley_df_csv["estimator"] == "State clustering (Favara-Imbs)")
            ]
            if r.empty:
                return None
            b, lo, hi = float(r["beta"].iloc[0]), float(r["ci_lower"].iloc[0]), float(r["ci_upper"].iloc[0])
            return b, lo, hi

        if est_label.startswith("ML SAR"):
            wl = est_label.replace("ML SAR ", "")
            row = sar_df[(sar_df["sample"] == sl) & (sar_df["w_matrix"] == wl)]
            if row.empty:
                return None
            b  = float(row["beta_D"].iloc[0])
            se = float(row["beta_D_se"].iloc[0])
            return b, b - z_crit * se, b + z_crit * se

        if est_label.startswith("IV-SAR"):
            wl  = est_label.replace("IV-SAR ", "")
            key = (sl + (" sample" if "Non" in sl else " sample"), wl)
            # handle both key formats
            for k, v in results_by_sample.items():
                ivr = v.get((k, wl)) or v.get((sl, wl))
                if ivr is None:
                    # try all keys
                    for kk, vv in v.items():
                        if kk[1] == wl and sl in kk[0]:
                            ivr = vv
                            break
                if ivr:
                    return ivr["beta"], ivr["beta_ci_lower"], ivr["beta_ci_upper"]
            return None

        return None

    sample_labels = list(results_by_sample.keys())
    fig, axes = plt.subplots(1, len(sample_labels), figsize=(12, 5), sharey=True)
    if len(sample_labels) == 1:
        axes = [axes]

    xs = np.arange(len(_EST_ORDER))
    for ax, sample_label in zip(axes, sample_labels):
        ivres = results_by_sample[sample_label]

        # OLS reference line
        ols_r = _row(sample_label, "OLS")
        if ols_r:
            ax.axhline(ols_r[0], color=_EST_COLORS["OLS"],
                       linewidth=1.0, linestyle="--", alpha=0.5, zorder=1)

        for i, est in enumerate(_EST_ORDER):
            # direct lookup
            if est == "OLS":
                sl = "Full" if "Full" in sample_label else "Border"
                r = conley_df_csv[
                    (conley_df_csv["sample"] == sl) &
                    (conley_df_csv["estimator"] == "State clustering (Favara-Imbs)")
                ]
                if r.empty:
                    continue
                b  = float(r["beta"].iloc[0])
                lo = float(r["ci_lower"].iloc[0])
                hi = float(r["ci_upper"].iloc[0])
            elif est.startswith("ML SAR"):
                wl  = est.replace("ML SAR ", "")
                sl  = "Full" if "Full" in sample_label else "Border"
                row = sar_df[(sar_df["sample"] == sl) & (sar_df["w_matrix"] == wl)]
                if row.empty:
                    continue
                b  = float(row["beta_D"].iloc[0])
                se = float(row["beta_D_se"].iloc[0])
                lo = b - z_crit * se
                hi = b + z_crit * se
            elif est.startswith("IV-SAR"):
                wl  = est.replace("IV-SAR ", "")
                sl  = "Full" if "Full" in sample_label else "Border"
                ivr = ivres.get((sl, wl))
                if ivr is None:
                    continue
                b  = ivr["beta"]
                lo = ivr["beta_ci_lower"]
                hi = ivr["beta_ci_upper"]
            else:
                continue

            color  = _EST_COLORS[est]
            marker = _EST_MARKERS[est]
            ax.vlines(xs[i], lo, hi, color=color, linewidth=2.2, zorder=3)
            ax.hlines([lo, hi], xs[i] - 0.15, xs[i] + 0.15,
                      color=color, linewidth=1.4, zorder=3)
            ax.scatter(xs[i], b, color=color, s=55, zorder=5,
                       marker=marker, edgecolors="white", linewidths=0.8)

            # Flag weak instruments
            if est.startswith("IV-SAR"):
                ivr = ivres.get(("Full" if "Full" in sample_label else "Border",
                                 wl))
                if ivr and ivr["first_stage_F"] < 10:
                    ax.annotate("!", xy=(xs[i], hi), fontsize=9,
                                color=color, ha="center", va="bottom")

        ax.axhline(0, color="0.65", linewidth=0.7, linestyle=":", zorder=1)
        ax.set_xticks(xs)
        ax.set_xticklabels(
            ["OLS", "ML SAR\n$W_{\\rm geo}$", "ML SAR\n$W_{\\rm bank}$",
             "IV-SAR\n$W_{\\rm geo}$", "IV-SAR\n$W_{\\rm bank}$"],
            fontsize=8.5,
        )
        ax.set_title(
            f"{sample_label}",
            fontsize=10, pad=6
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(-0.65, len(_EST_ORDER) - 0.35)
        ax.tick_params(axis="y", labelsize=9)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.3f"))

    axes[0].set_ylabel(r"$\hat{\beta}_1$ on Linter$\_$bra", fontsize=10)
    fig.suptitle(
        "Credit Growth IV-SAR vs OLS and ML-SAR\n"
        r"$\Delta\ln L_{it} = \rho (W\Delta\ln L)_{it} + \beta_1 \,\mathrm{Linter\_bra}_{it}$"
        " + county FE + year FE",
        fontsize=10, y=1.03,
    )

    legend_handles = [
        Line2D([0], [0], color=_EST_COLORS[e], linewidth=2.2,
               marker=_EST_MARKERS[e], markersize=6, label=e)
        for e in _EST_ORDER
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.12), ncol=5, fontsize=8.5,
               frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main run
# ══════════════════════════════════════════════════════════════════════════════

def run(output_dir=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load shared inputs ────────────────────────────────────────────────────
    co_df        = pd.read_csv(COUNTY_PATH, dtype={"fips5": str})
    county_order = co_df["fips5"].str.zfill(5).tolist()

    W_geo_all, gal_order = load_w_geo(county_order)
    assert gal_order == county_order, "GAL order mismatch"

    bank_vars  = load_bank_variants(county_order, W_geo_all=W_geo_all)
    W_bank_all = bank_vars["W_bank"]

    panel = load_panel_with_credit()
    panel["fips5"] = panel["fips5"].astype(str).str.zfill(5)
    YEARS    = sorted(panel["year"].unique())
    year_pos = {yr: i for i, yr in enumerate(YEARS)}
    panel_border = panel[panel["border"] == 1].copy()

    samples = [("Full", panel), ("Border", panel_border)]

    # ── Load existing results for comparison table ────────────────────────────
    sar_df = pd.read_csv(SAR_CSV) if SAR_CSV.exists() else pd.DataFrame()
    conley_df_csv = pd.read_csv(CONLEY_CSV) if CONLEY_CSV.exists() else pd.DataFrame()

    # ── Main estimation loop ──────────────────────────────────────────────────
    all_iv_results = {}    # {sample_label: {(sample_label, W): result_dict}}
    csv_rows       = []

    for sample_label, panel_sub in samples:
        print(f"\n{'='*70}")
        print(f"Sample: {sample_label}")
        print(f"{'='*70}")

        s = build_sample(panel_sub, county_order, W_geo_all, W_bank_all,
                         YEARS, year_pos, sample_label)
        print(f"N={s['N']} counties | T={s['T']} | NT={s['NT']:,} obs | "
              f"G={len(np.unique(s['state_idx_flat']))} states")

        all_iv_results[sample_label] = {}

        for W_label in ["W_geo", "W_bank"]:
            r = run_iv_sar(s, W_label, sample_label, verbose=True)
            all_iv_results[sample_label][(sample_label, W_label)] = r

            csv_rows.append(dict(
                sample               = r["sample"],
                W                    = r["W"],
                estimator            = "IV-SAR",
                rho                  = r["rho"],
                rho_se               = r["rho_se"],
                rho_ci_lower         = r["rho_ci_lower"],
                rho_ci_upper         = r["rho_ci_upper"],
                beta                 = r["beta"],
                beta_se              = r["beta_se"],
                beta_ci_lower        = r["beta_ci_lower"],
                beta_ci_upper        = r["beta_ci_upper"],
                first_stage_F        = r["first_stage_F"],
                residual_moran_i_mean = r["residual_moran_i_mean"],
                N_counties           = r["N_counties"],
                N_obs                = r["N_obs"],
            ))

    # ── Print formatted comparison table ──────────────────────────────────────
    print_comparison_table(all_iv_results, sar_df, conley_df_csv)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if output_dir is not None:
        cols = ["sample", "W", "estimator",
                "rho", "rho_se", "rho_ci_lower", "rho_ci_upper",
                "beta", "beta_se", "beta_ci_lower", "beta_ci_upper",
                "first_stage_F", "residual_moran_i_mean", "N_counties", "N_obs"]
        pd.DataFrame(csv_rows)[cols].to_csv(
            output_dir / "sar_iv_results.csv", index=False)
        print(f"\nSaved sar_iv_results.csv to {output_dir}")

    # ── Comparison plot ────────────────────────────────────────────────────────
    if output_dir is not None:
        make_comparison_plot(
            all_iv_results, sar_df, conley_df_csv,
            output_dir / "sar_iv_comparison.png"
        )

    return all_iv_results, csv_rows


if __name__ == "__main__":
    run(Path(__file__).parents[2] / "output")
