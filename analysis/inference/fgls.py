"""
fgls_comparison.py
==================
Feasible GLS for the credit growth equation, comparing five estimators using
state-clustered standard errors.

Model: Dl_nloans_b_it = beta * Linter_bra_it + gamma * X_ct
       + county FE + year FE + eps_it

Estimators (run for full and border samples):
  1. OLS             -- Favara & Imbs (2015) baseline (no spatial correction)
  2. FGLS W_geo      -- filter A = I - lambda_geo * W_geo
  3. FGLS W_bank     -- filter A = I - lambda_bank * W_bank
  4. FGLS W_bank_knn3-- filter A = I - lambda_knn3 * W_bank_knn3
  5. FGLS W_bank_knn4-- filter A = I - lambda_knn4 * W_bank_knn4

Lambda values are read from saved ML-SEM results:
  output/panel_fe_credit_results.csv    -- lambda_geo, lambda_bank, lambda_knn3, lambda_knn4

FGLS transformation (applied period-by-period via sparse matmul):
  y_tilde_A[t] = A @ y_tilde[t]    where A = I - lambda * W
  x_tilde_A[t] = A @ x_tilde[t]

Point estimate:
  beta_fgls = (x_tilde_A' x_tilde_A)^{-1} (x_tilde_A' y_tilde_A)  [scalar k=1]

State-clustered SEs (UNFILTERED residuals, FILTERED regressor as bread):
  xi_hat  = y_tilde - x_tilde * beta_fgls    [unfiltered residuals]
  bread   = sum_{c,t} (x_tilde_A_ct)^2
  score_s = sum_{c in s, t} x_tilde_A_ct * xi_hat_ct
  meat    = sum_s score_s^2
  Var     = meat / bread^2 * G/(G-1)

Tests:
  Hausman chi2(1) : OLS vs FGLS W_bank
  Hausman t-test  : fallback if denom <= 0 (reports next to chi2)
  Moran's I       : year-by-year on FGLS W_bank residuals, W_bank weights

Spectral radius check: |lambda| < 1 / rho(W) required before applying filter.

Outputs:
  output/fgls_comparison.csv   -- beta, SE, CI, t-stat, p-value for each estimator
  output/fgls_comparison.png   -- coefficient plot (5 estimators × 3 samples)
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
from scipy.sparse import eye as speye
from scipy.sparse.linalg import eigs as sp_eigs

sys.path.insert(0, str(Path(__file__).parents[1]))
import utils  # noqa: applies spreg patch
from utils import row_standardize, sparse_to_pysal_w
from panel_data import CREDIT_CONTROLS, load_panel_with_credit
from w_variants import load_w_geo, load_bank_variants

ROOT        = Path(__file__).parents[2]
COUNTY_PATH = ROOT / "data" / "county_order_Wgeo.csv"
DV          = "Dl_nloans_b"
X_VARS      = ["Linter_bra"] + CREDIT_CONTROLS

ESTIMATOR_LABELS = [
    "OLS (Favara-Imbs)",
    "FGLS $W_{\\rm geo}$",
    "FGLS $W_{\\rm bank}$",
    "FGLS $W_{\\rm bank,knn3}$",
    "FGLS $W_{\\rm bank,knn4}$",
]

_COLORS = {
    "OLS (Favara-Imbs)":      "#2166ac",
    "FGLS W_geo":             "#4dac26",
    "FGLS W_bank":            "#d01c8b",
    "FGLS W_bank_knn3":       "#a6cee3",
    "FGLS W_bank_knn4":       "#b2df8a",
}


# ══════════════════════════════════════════════════════════════════════════════
# Within transformation (two-way FE demeaning)
# ══════════════════════════════════════════════════════════════════════════════

def two_way_within(arr_TN):
    """
    Two-way (county + year) within transformation for a (T, N) panel array.

    Computes arr_it - arr_bar_i - arr_bar_t + arr_bar using three passes:
      1. Subtract county mean (over T)
      2. Add grand mean back
      3. Subtract year mean (over N after step 2)
    """
    county_mean = arr_TN.mean(axis=0)           # (N,) mean over time
    grand_mean  = float(arr_TN.mean())           # scalar
    z = arr_TN - county_mean[None, :]
    z = z + grand_mean
    return z - z.mean(axis=1, keepdims=True)     # (T, N)


# ══════════════════════════════════════════════════════════════════════════════
# Spectral radius
# ══════════════════════════════════════════════════════════════════════════════

def spectral_radius(W_sp):
    """
    Estimate the spectral radius rho(W) = max|eigenvalue| via ARPACK.

    Falls back to max(row_sum) (upper bound) on any convergence failure.
    For row-standardised W with non-negative entries, the Perron root equals
    the maximum row sum = 1; the result is therefore almost always 1.0.
    """
    N = W_sp.shape[0]
    try:
        ncv = min(N - 1, max(10, 50))
        v = sp_eigs(
            W_sp.astype(np.float64), k=1, which="LM",
            return_eigenvectors=False, ncv=ncv,
            maxiter=5000, tol=1e-8,
        )
        return float(np.abs(v[0]))
    except Exception:
        # Safe upper bound via Gershgorin / max row sum
        return float(np.array(W_sp.sum(axis=1)).max())


# ══════════════════════════════════════════════════════════════════════════════
# FGLS spatial filter
# ══════════════════════════════════════════════════════════════════════════════

def apply_filter(arr_TN, A_sp):
    """
    Apply sparse filter A = I - lambda*W to a (T, N) array period-by-period.

    Returns (T, N) array where arr_A[t] = A @ arr_TN[t].
    Uses scipy sparse mat-vec for efficiency: O(T * nnz(A)).
    """
    if arr_TN.ndim == 3:
        T, N, K = arr_TN.shape
        out = np.empty_like(arr_TN)
        for t in range(T):
            out[t] = A_sp @ arr_TN[t]
        return out

    T, N = arr_TN.shape
    out  = np.empty_like(arr_TN)
    for t in range(T):
        out[t] = A_sp @ arr_TN[t]
    return out


def build_filter(W_sp, lam):
    """
    Build spatial filter matrix A = I - lambda * W (sparse CSR).

    Raises ValueError if |lambda| >= 1 / rho(W) (instability condition).
    """
    rho = spectral_radius(W_sp)
    threshold = 1.0 / rho if rho > 0 else np.inf
    if abs(lam) >= threshold:
        raise ValueError(
            f"|lambda|={abs(lam):.4f} >= 1/rho(W)={threshold:.4f} -- "
            "filter is not a contraction; results would be unstable."
        )
    N   = W_sp.shape[0]
    A   = speye(N, format="csr") - lam * W_sp.tocsr()
    return A, rho


# ══════════════════════════════════════════════════════════════════════════════
# OLS scalar
# ══════════════════════════════════════════════════════════════════════════════

def _fit_matrix(y_TN, X_TNK):
    """OLS for a controlled within-transformed design matrix."""
    T, N, K = X_TNK.shape
    X_f = X_TNK.reshape(T * N, K)
    y_f = y_TN.reshape(T * N)
    XtX_inv = np.linalg.inv(X_f.T @ X_f)
    beta = XtX_inv @ (X_f.T @ y_f)
    return beta, XtX_inv


def ols_matrix(y_TN, X_TNK):
    beta, _ = _fit_matrix(y_TN, X_TNK)
    xi = y_TN - np.einsum("tnk,k->tn", X_TNK, beta)
    return beta, xi


def fgls_matrix(y_TN, X_TNK, A_sp):
    """FGLS with the spatial filter applied to every regressor column."""
    y_A = apply_filter(y_TN, A_sp)
    X_A = apply_filter(X_TNK, A_sp)
    beta, _ = _fit_matrix(y_A, X_A)
    xi = y_TN - np.einsum("tnk,k->tn", X_TNK, beta)
    return beta, xi, X_A


# ══════════════════════════════════════════════════════════════════════════════
# State-clustered SE (scalar k=1, mixed bread/meat)
# ══════════════════════════════════════════════════════════════════════════════

def cluster_se_matrix(xi_TN, X_A_TNK, county_states, param_idx=0):
    """
    State-clustered sandwich SE for one coefficient in the controlled model.

    Var(beta) = meat / bread^2 * G/(G-1)
    where:
      bread   = sum_{c,t} x_A_ct^2                       [filtered regressor]
      score_s = sum_{c in s, t} x_A_ct * xi_ct           [mixed score]
      meat    = sum_s score_s^2

    Parameters
    ----------
    xi_TN        : (T, N) UNFILTERED residuals xi = y_tilde - x_tilde * beta
    x_A_TN       : (T, N) FILTERED regressor x_tilde_A (= x_tilde for OLS)
    county_states: (N,)   integer state id for each county

    Returns
    -------
    se : float
    G  : int  (number of unique states)
    """
    T, N, K = X_A_TNK.shape
    X_f = X_A_TNK.reshape(T * N, K)
    XtX_inv = np.linalg.inv(X_f.T @ X_f)

    # Mixed score: filtered regressor × unfiltered residual
    scores  = (X_A_TNK * xi_TN[:, :, None]).sum(axis=0)     # (N,K)

    states  = np.unique(county_states)
    G       = len(states)
    meat    = np.zeros((K, K))
    for s in states:
        mask = (county_states == s)
        score_s = scores[mask].sum(axis=0)
        meat += np.outer(score_s, score_s)

    df_corr = G / (G - 1)
    vcv     = XtX_inv @ meat @ XtX_inv * df_corr
    se      = float(np.sqrt(max(vcv[param_idx, param_idx], 0.0)))
    return se, G


def build_result_row(beta, se, sample, estimator, W_label, lam_used, rho_W, N, T, G):
    """Pack a single result into a dict for the output CSV."""
    z_crit = st.norm.ppf(0.975)
    lo     = beta - z_crit * se
    hi     = beta + z_crit * se
    t_val  = beta / se if se > 0 else np.inf
    p_val  = float(2 * st.norm.sf(abs(t_val)))
    return dict(
        sample=sample, estimator=estimator, W_label=W_label,
        beta=beta, se=se,
        ci_lower=lo, ci_upper=hi, ci_width=hi - lo,
        t_stat=t_val, p_value=p_val,
        lam_used=lam_used, rho_W=rho_W,
        N=N, T=T, G=G,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Hausman test
# ══════════════════════════════════════════════════════════════════════════════

def hausman_test(beta_ols, beta_fgls, se_ols, se_fgls):
    """
    Hausman (1978) efficiency test: H0 = OLS is consistent + efficient.

    H_stat = (beta_ols - beta_fgls)^2 / (se_ols^2 - se_fgls^2)

    If denominator <= 0 (non-positive-definite), the chi2 test is unreliable.
    Always report a t-test alternative:

    t_diff = (beta_ols - beta_fgls) / sqrt(se_ols^2 + se_fgls^2)

    Returns
    -------
    dict with keys: diff, se_diff_sum, t_diff, p_diff,
                    denom, H_stat (None if denom<=0),
                    p_hausman (None if denom<=0),
                    note (str)
    """
    diff      = float(beta_ols - beta_fgls)
    se_sum_sq = float(se_ols ** 2 + se_fgls ** 2)
    se_diff   = float(np.sqrt(max(se_sum_sq, 0.0)))

    t_diff  = diff / se_diff if se_diff > 0 else np.inf
    p_diff  = float(2 * st.norm.sf(abs(t_diff)))

    denom   = float(se_ols ** 2 - se_fgls ** 2)

    if denom > 1e-16:
        H_stat   = float(diff ** 2 / denom)
        p_hausman= float(1.0 - st.chi2.cdf(H_stat, df=1))
        note     = "chi2(1)"
    else:
        H_stat   = None
        p_hausman= None
        note     = "non-PD denom -> t-test only"

    return dict(
        diff=diff, denom=denom,
        H_stat=H_stat, p_hausman=p_hausman,
        t_diff=t_diff, p_diff=p_diff,
        se_ols=se_ols, se_fgls=se_fgls,
        note=note,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Moran's I on FGLS residuals (year-by-year)
# ══════════════════════════════════════════════════════════════════════════════

def moran_by_year(xi_TN, W_bank_sub, YEARS):
    """
    Compute Moran's I year-by-year on the provided residual array.

    xi_TN      : (T, N) residuals
    W_bank_sub : (N, N) scipy sparse spatial weights for W_bank
    YEARS      : list of year labels (length T)

    Returns list of dicts: year, I, E_I, z_norm, p_sim, p_norm
    """
    from esda.moran import Moran

    w_pysal = sparse_to_pysal_w(W_bank_sub)
    rows = []
    T    = xi_TN.shape[0]
    for t in range(T):
        xi_t = xi_TN[t].copy()
        mi   = Moran(xi_t, w_pysal, permutations=999)
        rows.append(dict(
            year  = int(YEARS[t]),
            I     = float(mi.I),
            E_I   = float(mi.EI),
            z_norm= float(mi.z_norm),
            p_sim = float(mi.p_sim),
            p_norm= float(mi.p_norm),
        ))
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Sample builder
# ══════════════════════════════════════════════════════════════════════════════

def build_sample(panel_sub, county_order, W_geo_all, W_bank_all,
                 W_knn3_all, W_knn4_all, YEARS, year_pos):
    """
    Build all inputs for FGLS comparison for one sample.

    County filter: same as panel_fe_credit.build_arrays
      - County present in panel_sub
      - No any-NaN in Dl_nloans_b or controlled regressors

    W submatrices are re-row-standardised after subsetting (necessary for
    correct spectral radius = 1 and proper FGLS filter properties).

    Returns dict with:
      N, T, NT, YEARS
      y_TN, x_TN                    -- (T, N) DV and (T, N, K) controlled X
      county_states                  -- (N,) integer state ID
      W_geo_sub, W_bank_sub,
      W_knn3_sub, W_knn4_sub         -- (N, N) sparse, row-standardised
      usable                         -- list of fips5 in column order
    """
    T = len(YEARS)

    # -- County filter ---------------------------------------------------------
    any_nan    = panel_sub.groupby("fips5")[[DV] + X_VARS].apply(
        lambda g: g.isna().any().any()
    )
    sub_co     = set(panel_sub["fips5"].unique())
    island_sets = []
    for W_all in (W_geo_all, W_bank_all, W_knn3_all, W_knn4_all):
        full_rs = np.array(W_all.sum(axis=1)).flatten()
        island_sets.append({county_order[i] for i, r in enumerate(full_rs) if r == 0})
    islands = set().union(*island_sets)
    usable     = [c for c in county_order
                  if c in sub_co and not any_nan.get(c, True) and c not in islands]
    N          = len(usable)
    usable_pos = {c: i for i, c in enumerate(usable)}

    df = (panel_sub[panel_sub["fips5"].isin(set(usable))]
          .assign(t_idx=lambda d: d["year"].map(year_pos),
                  c_idx=lambda d: d["fips5"].map(usable_pos))
          .sort_values(["t_idx", "c_idx"]))

    y_flat = df[DV].values.astype(np.float64)
    X_flat = df[X_VARS].values.astype(np.float64)

    assert not np.isnan(y_flat).any(), f"NaN in {DV}"
    assert not np.isnan(X_flat).any(), "NaN in controlled X"
    assert len(y_flat) == N * T

    # -- State mapping ---------------------------------------------------------
    county_states = (
        df.groupby("c_idx")["state_n"]
          .first()
          .sort_index()
          .values.astype(int)
    )

    # -- Two-way within transformation ----------------------------------------
    K = len(X_VARS)
    y_TN = two_way_within(y_flat.reshape(T, N))   # (T, N)
    X_raw = X_flat.reshape(T, N, K)
    x_TN = np.stack([two_way_within(X_raw[:, :, j]) for j in range(K)], axis=2)

    # -- W submatrices (re-row-standardised after subsetting) -----------------
    idx        = np.array([county_order.index(c) for c in usable])
    W_geo_sub  = row_standardize(W_geo_all[idx, :][:, idx])
    W_bank_sub = row_standardize(W_bank_all[idx, :][:, idx])
    W_knn3_sub = row_standardize(W_knn3_all[idx, :][:, idx])
    W_knn4_sub = row_standardize(W_knn4_all[idx, :][:, idx])

    return dict(
        N=N, T=T, NT=N * T, YEARS=YEARS,
        y_TN=y_TN, x_TN=x_TN,
        county_states=county_states,
        W_geo_sub=W_geo_sub, W_bank_sub=W_bank_sub,
        W_knn3_sub=W_knn3_sub, W_knn4_sub=W_knn4_sub,
        usable=usable,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Run all four estimators for one sample
# ══════════════════════════════════════════════════════════════════════════════

def run_sample(s, sample_label, lam_geo, lam_bank, lam_knn3, lam_knn4,
               verbose=True):
    """
    Run OLS + four FGLS estimators for one sample dict.

    Parameters
    ----------
    s            : dict from build_sample()
    sample_label : str
    lam_geo      : float  lambda for W_geo filter
    lam_bank     : float  lambda for W_bank filter
    lam_knn3     : float or None  lambda for W_bank_knn3 filter
    lam_knn4     : float or None  lambda for W_bank_knn4 filter
    verbose      : bool

    Returns
    -------
    rows    : list of result dicts (one per estimator)
    hausman : dict from hausman_test()
    moran   : list of per-year Moran's I dicts (under W_bank)
    """
    N  = s["N"];  T  = s["T"];  YEARS = s["YEARS"]
    y  = s["y_TN"]
    x  = s["x_TN"]
    cs = s["county_states"]
    W_geo  = s["W_geo_sub"]
    W_bank = s["W_bank_sub"]
    W_knn3 = s["W_knn3_sub"]
    W_knn4 = s["W_knn4_sub"]

    rows   = []

    # ── 1. OLS (no filter, x_A = x_tilde) ──────────────────────────────────
    beta_ols_vec, xi_ols = ols_matrix(y, x)
    beta_ols = float(beta_ols_vec[0])
    se_ols, G = cluster_se_matrix(xi_ols, x, cs)
    rows.append(build_result_row(
        beta_ols, se_ols, sample_label, "OLS (Favara-Imbs)", "none",
        0.0, 0.0, N, T, G,
    ))
    if verbose:
        print(f"  OLS:        beta={beta_ols:.6f}  SE={se_ols:.6f}  G={G}")

    # -- Sanity check ----------------------------------------------------------
    if not (0.010 < abs(beta_ols) < 0.060):
        print(f"  WARNING: OLS beta={beta_ols:.4f} outside expected range [0.010, 0.060]"
              " -- verify within transformation")

    # ── 2. FGLS W_geo ───────────────────────────────────────────────────────
    A_geo, rho_geo = build_filter(W_geo, lam_geo)
    beta_geo_vec, xi_geo, x_A_geo = fgls_matrix(y, x, A_geo)
    beta_geo = float(beta_geo_vec[0])
    se_geo, _ = cluster_se_matrix(xi_geo, x_A_geo, cs)
    rows.append(build_result_row(
        beta_geo, se_geo, sample_label, "FGLS W_geo", "W_geo",
        lam_geo, rho_geo, N, T, G,
    ))
    if verbose:
        print(f"  FGLS W_geo: beta={beta_geo:.6f}  SE={se_geo:.6f}  "
              f"lambda={lam_geo:.4f}  rho(W)={rho_geo:.4f}")

    # ── 3. FGLS W_bank ──────────────────────────────────────────────────────
    A_bank, rho_bank = build_filter(W_bank, lam_bank)
    beta_bank_vec, xi_bank, x_A_bank = fgls_matrix(y, x, A_bank)
    beta_bank = float(beta_bank_vec[0])
    se_bank, _ = cluster_se_matrix(xi_bank, x_A_bank, cs)
    rows.append(build_result_row(
        beta_bank, se_bank, sample_label, "FGLS W_bank", "W_bank",
        lam_bank, rho_bank, N, T, G,
    ))
    if verbose:
        print(f"  FGLS W_bank:beta={beta_bank:.6f}  SE={se_bank:.6f}  "
              f"lambda={lam_bank:.4f}  rho(W)={rho_bank:.4f}")

    # ── 3b. FGLS W_bank_knn3 ────────────────────────────────────────────────
    if lam_knn3 is not None:
        A_knn3, rho_knn3 = build_filter(W_knn3, lam_knn3)
        beta_knn3_vec, xi_knn3, x_A_knn3 = fgls_matrix(y, x, A_knn3)
        beta_knn3 = float(beta_knn3_vec[0])
        se_knn3, _ = cluster_se_matrix(xi_knn3, x_A_knn3, cs)
        rows.append(build_result_row(
            beta_knn3, se_knn3, sample_label, "FGLS W_bank_knn3", "W_bank_knn3",
            lam_knn3, rho_knn3, N, T, G,
        ))
        if verbose:
            print(f"  FGLS knn3:  beta={beta_knn3:.6f}  SE={se_knn3:.6f}  "
                  f"lambda={lam_knn3:.4f}  rho(W)={rho_knn3:.4f}")

    # ── 3c. FGLS W_bank_knn4 ────────────────────────────────────────────────
    if lam_knn4 is not None:
        A_knn4, rho_knn4 = build_filter(W_knn4, lam_knn4)
        beta_knn4_vec, xi_knn4, x_A_knn4 = fgls_matrix(y, x, A_knn4)
        beta_knn4 = float(beta_knn4_vec[0])
        se_knn4, _ = cluster_se_matrix(xi_knn4, x_A_knn4, cs)
        rows.append(build_result_row(
            beta_knn4, se_knn4, sample_label, "FGLS W_bank_knn4", "W_bank_knn4",
            lam_knn4, rho_knn4, N, T, G,
        ))
        if verbose:
            print(f"  FGLS knn4:  beta={beta_knn4:.6f}  SE={se_knn4:.6f}  "
                  f"lambda={lam_knn4:.4f}  rho(W)={rho_knn4:.4f}")

    # ── Hausman test: OLS vs FGLS W_bank ────────────────────────────────────
    hausman = hausman_test(beta_ols, beta_bank, se_ols, se_bank)
    if verbose:
        print(f"  Hausman (OLS vs FGLS W_bank): diff={hausman['diff']:.4f}  "
              f"denom={hausman['denom']:.6f}  ({hausman['note']})")
        if hausman["H_stat"] is not None:
            print(f"    chi2(1)={hausman['H_stat']:.3f}  p={hausman['p_hausman']:.4f}")
        print(f"    t_diff={hausman['t_diff']:.3f}  p_diff={hausman['p_diff']:.4f}")

    # ── Moran's I on FGLS W_bank residuals year-by-year ─────────────────────
    if verbose:
        print(f"  Computing Moran's I on FGLS W_bank residuals (T={T} years × 999 perms)...",
              flush=True)
    moran = moran_by_year(xi_bank, W_bank, YEARS)
    mean_I = np.mean([r["I"] for r in moran])
    mean_p = np.mean([r["p_sim"] for r in moran])
    if verbose:
        print(f"  Moran W_bank residuals: mean I={mean_I:.4f}  mean p_sim={mean_p:.4f}")

    return rows, hausman, moran


# ══════════════════════════════════════════════════════════════════════════════
# Print formatted table
# ══════════════════════════════════════════════════════════════════════════════

def stars(p):
    return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else "  "))


def print_table(rows, hausman, moran, sample_label):
    W = 82
    print()
    print("=" * W)
    r0 = rows[0]
    print(f"FGLS comparison -- {sample_label} sample  "
          f"(N={r0['N']}, T={r0['T']}, {r0['N']*r0['T']:,} obs, G={r0['G']} states)")
    print("-" * W)
    print(f"{'Estimator':<26} {'lambda':>7} {'beta':>9} {'SE':>7}  "
          f"{'95% CI':>20}  {'t':>6}  {'p':>6}")
    print("-" * W)
    for r in rows:
        lam_s = f"{r['lam_used']:.4f}" if r["lam_used"] > 0 else "—"
        ci    = f"[{r['ci_lower']:+.4f},{r['ci_upper']:+.4f}]"
        sig   = stars(r["p_value"])
        print(f"  {r['estimator']:<24} {lam_s:>7} "
              f"{r['beta']:>9.5f}{sig} {r['se']:>7.5f}  "
              f"{ci:>22}  {r['t_stat']:>6.2f}  {r['p_value']:>6.4f}")
    print("-" * W)

    # Hausman
    h = hausman
    print(f"\n  Hausman test (OLS vs FGLS W_bank): diff={h['diff']:.5f}  denom={h['denom']:.6f}")
    if h["H_stat"] is not None:
        sig_h = stars(h["p_hausman"])
        print(f"    chi2(1) = {h['H_stat']:.3f}  p = {h['p_hausman']:.4f} {sig_h}")
    else:
        print(f"    {h['note']}")
    print(f"    t-test fallback: t = {h['t_diff']:.3f}  p = {h['p_diff']:.4f} "
          f"{stars(h['p_diff'])}")

    # Moran
    mean_I = np.mean([r["I"] for r in moran])
    mean_p = np.mean([r["p_sim"] for r in moran])
    rng_I  = (min(r["I"] for r in moran), max(r["I"] for r in moran))
    print(f"\n  Moran's I on FGLS W_bank residuals (999 perms):")
    print(f"    mean I = {mean_I:.4f}  [{rng_I[0]:.4f}, {rng_I[1]:.4f}]  "
          f"mean p_sim = {mean_p:.4f}")
    print("=" * W)


# ══════════════════════════════════════════════════════════════════════════════
# Publication-quality coefficient plot
# ══════════════════════════════════════════════════════════════════════════════

_PLOT_COLORS = ["#2166ac", "#4dac26", "#d01c8b", "#a6cee3", "#b2df8a"]
_PLOT_ESTIMATORS = list(_COLORS.keys())
_SHORT_LABELS = [
    "OLS",
    "FGLS\n$W_{\\rm geo}$",
    "FGLS\n$W_{\\rm bank}$",
    "FGLS\n$W_{\\rm knn3}$",
    "FGLS\n$W_{\\rm knn4}$",
]


def make_plot(all_rows):
    """
    Coefficient plot with 95% CI bars for all four estimators × two samples.

    all_rows: dict  {sample_label: list_of_result_dicts}
    """
    sample_labels = list(all_rows.keys())
    n_samples = len(sample_labels)

    fig, axes = plt.subplots(1, n_samples, figsize=(10, 5), sharey=True)
    if n_samples == 1:
        axes = [axes]

    xs = np.arange(len(_SHORT_LABELS))
    slot_index = {estimator: idx for idx, estimator in enumerate(_PLOT_ESTIMATORS)}
    color_lookup = {
        estimator: _PLOT_COLORS[idx]
        for idx, estimator in enumerate(_PLOT_ESTIMATORS)
    }
    for ax, slabel in zip(axes, sample_labels):
        rlist = all_rows[slabel]
        for r in rlist:
            estimator = r["estimator"]
            if estimator not in slot_index:
                continue
            x = xs[slot_index[estimator]]
            col = color_lookup[estimator]
            ax.vlines(x, r["ci_lower"], r["ci_upper"],
                      color=col, linewidth=2.5, zorder=3)
            ax.hlines([r["ci_lower"], r["ci_upper"]],
                      x - 0.13, x + 0.13,
                      color=col, linewidth=1.5, zorder=3)
            ax.scatter(x, r["beta"],
                       color=col, s=55, zorder=5,
                       edgecolors="white", linewidths=0.8)

        ax.axhline(0, color="0.6", linewidth=0.8, linestyle=":", zorder=1)
        ax.set_xticks(xs)
        ax.set_xticklabels(_SHORT_LABELS, fontsize=9, ha="center")
        ax.tick_params(axis="x", pad=4)
        r0 = rlist[0]
        ax.set_title(
            f"{slabel}\n$N={r0['N']:,}$ counties, ${r0['N']*r0['T']:,}$ obs",
            fontsize=10, pad=8,
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(-0.60, len(_SHORT_LABELS) - 0.40)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.3f"))
        ax.tick_params(axis="y", labelsize=9)

    axes[0].set_ylabel(r"$\hat{\beta}$ on Linter\_bra", fontsize=10)

    legend_handles = [
        Line2D([0], [0], color=_PLOT_COLORS[i], linewidth=2.5,
               marker="o", markersize=6, label=ESTIMATOR_LABELS[i])
        for i in range(len(_PLOT_ESTIMATORS))
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", bbox_to_anchor=(0.5, -0.08),
        ncol=2, fontsize=8.5, frameon=False,
    )
    fig.suptitle(
        "Feasible GLS vs OLS: Credit Growth Equation\n"
        r"$\Delta\ln(\mathrm{loans}_{b,it}) = \beta\,\mathrm{Linter\_bra}_{it}"
        r" + \gamma^\top X_{it}$"
        " + county FE + year FE  |  State-clustered SEs",
        fontsize=10, y=1.03,
    )
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Lambda loader
# ══════════════════════════════════════════════════════════════════════════════

def load_lambdas(output_dir):
    """
    Load lambda values from saved ML-SEM results.

    Returns dict:
      {
        'full':      {'geo': float, 'bank': float, 'knn3': float, 'knn4': float},
        'contig':    {...},
        'noncontig': {...},
      }
    knn3/knn4 lambdas fall back to bank lambda if their rows are absent
    (e.g., when sem_credit.py has not yet been run with those variants).
    """
    sem_path = output_dir / "panel_fe_credit_results.csv"
    sem      = pd.read_csv(sem_path)

    def get_sem_row(model_str):
        row = sem[sem["model"] == model_str]
        assert len(row) == 1, f"model '{model_str}' not found in {sem_path}"
        return row.iloc[0]

    def get_sem_lam(model_str):
        return float(get_sem_row(model_str)["lam"])

    def get_paired_geo_lam(sample_tag):
        model_str = f"FE W_bank ({sample_tag})"
        row = get_sem_row(model_str)
        if "lam_geo" in row.index and not pd.isna(row["lam_geo"]):
            return float(row["lam_geo"])
        return get_sem_lam(f"FE W_geo ({sample_tag})")

    def get_sem_lam_soft(model_str, fallback_model=None):
        """Return lambda; fall back to fallback_model if not found."""
        row = sem[sem["model"] == model_str]
        if len(row) == 0:
            if fallback_model is not None:
                print(f"  [WARN] '{model_str}' not in SEM results; "
                      f"using '{fallback_model}' lambda as fallback.")
                return get_sem_lam(fallback_model)
            return None
        return float(row["lam"].iloc[0])

    return {
        "full": {
            "geo" : get_paired_geo_lam("full"),
            "bank": get_sem_lam("FE W_bank (full)"),
            "knn3": get_sem_lam_soft("FE W_bank_knn3 (full)"),
            "knn4": get_sem_lam_soft("FE W_bank_knn4 (full)"),
        },
        "contig": {
            "geo" : get_paired_geo_lam("contig"),
            "bank": get_sem_lam("FE W_bank (contig)"),
            "knn3": get_sem_lam_soft("FE W_bank_knn3 (contig)"),
            "knn4": get_sem_lam_soft("FE W_bank_knn4 (contig)"),
        },
        "noncontig": {
            "geo" : get_paired_geo_lam("noncontig"),
            "bank": get_sem_lam("FE W_bank (noncontig)"),
            "knn3": get_sem_lam_soft("FE W_bank_knn3 (noncontig)"),
            "knn4": get_sem_lam_soft("FE W_bank_knn4 (noncontig)"),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# Master entry point
# ══════════════════════════════════════════════════════════════════════════════

def run(output_dir=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # -- Load lambda values from saved results --------------------------------
    lams = load_lambdas(output_dir)
    print("\nLambda values loaded from saved ML-SEM results:")
    for sample_key, d in lams.items():
        print(f"  {sample_key}: lambda_geo={d['geo']:.4f}  "
              f"lambda_bank={d['bank']:.4f}  lambda_knn3={d['knn3']}  "
              f"lambda_knn4={d['knn4']}")

    # -- Load panel and spatial weight matrices --------------------------------
    co_df        = pd.read_csv(COUNTY_PATH, dtype={"fips5": str})
    county_order = co_df["fips5"].str.zfill(5).tolist()

    W_geo_all, gal_order = load_w_geo(county_order)
    assert gal_order == county_order, "GAL county order mismatch"

    bank_vars        = load_bank_variants(county_order, W_geo_all=W_geo_all)
    W_bank_all       = bank_vars["W_bank"]
    W_bank_knn3_all  = bank_vars["W_bank_knn3"]
    W_bank_knn4_all  = bank_vars["W_bank_knn4"]

    panel = load_panel_with_credit()
    panel["fips5"] = panel["fips5"].astype(str).str.zfill(5)
    YEARS    = sorted(panel["year"].unique())
    year_pos = {yr: i for i, yr in enumerate(YEARS)}
    panel_contig    = panel[panel["border"] == 1].copy()
    panel_noncontig = panel[panel["border"] == 0].copy()

    sample_specs = [
        ("Full",      panel,             "full"),
        ("Contig",    panel_contig,      "contig"),
        ("NonContig", panel_noncontig,   "noncontig"),
    ]

    # -- Main estimation loop --------------------------------------------------
    all_csv_rows  = []
    all_rows_plot = {}
    all_moran     = {}
    all_hausman   = {}

    for slabel, panel_sub, lam_key in sample_specs:
        print(f"\n{'='*60}")
        print(f"Sample: {slabel}")
        print(f"{'='*60}")
        print("  Building within-transformed arrays ...", flush=True)

        s = build_sample(panel_sub, county_order, W_geo_all, W_bank_all,
                         W_bank_knn3_all, W_bank_knn4_all, YEARS, year_pos)
        print(f"  N={s['N']}  T={s['T']}  NT={s['NT']:,}")

        d = lams[lam_key]
        rows, hausman, moran = run_sample(
            s, slabel,
            lam_geo=d["geo"], lam_bank=d["bank"],
            lam_knn3=d["knn3"], lam_knn4=d["knn4"],
            verbose=True,
        )

        print_table(rows, hausman, moran, slabel)

        all_rows_plot[slabel] = rows
        all_hausman[slabel]   = hausman
        all_moran[slabel]     = moran
        all_csv_rows.extend(rows)

    # -- Save CSV -------------------------------------------------------------
    if output_dir is not None:
        cols = [
            "sample", "estimator", "W_label", "lam_used", "rho_W",
            "beta", "se", "ci_lower", "ci_upper", "ci_width",
            "t_stat", "p_value", "N", "T", "G",
        ]
        pd.DataFrame(all_csv_rows)[cols].to_csv(
            output_dir / "fgls_comparison.csv", index=False
        )
        print(f"\nSaved fgls_comparison.csv  to {output_dir}")

        # Hausman + Moran summary rows
        hm_rows = []
        for slabel in all_rows_plot:
            h = all_hausman[slabel]
            m = all_moran[slabel]
            hm_rows.append(dict(
                sample       = slabel,
                hausman_diff = h["diff"],
                hausman_denom= h["denom"],
                H_stat       = h["H_stat"] if h["H_stat"] is not None else np.nan,
                p_hausman    = h["p_hausman"] if h["p_hausman"] is not None else np.nan,
                t_diff       = h["t_diff"],
                p_diff       = h["p_diff"],
                hausman_note = h["note"],
                moran_mean_I = float(np.mean([r["I"] for r in m])),
                moran_mean_p = float(np.mean([r["p_sim"] for r in m])),
            ))
        pd.DataFrame(hm_rows).to_csv(
            output_dir / "fgls_hausman_moran.csv", index=False
        )
        print(f"Saved fgls_hausman_moran.csv to {output_dir}")

    # -- Publication-quality plot ---------------------------------------------
    fig = make_plot(all_rows_plot)
    if output_dir is not None:
        fig.savefig(output_dir / "fgls_comparison.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved fgls_comparison.png  to {output_dir}")

    return dict(rows=all_csv_rows, hausman=all_hausman, moran=all_moran)


if __name__ == "__main__":
    run(Path(__file__).parents[2] / "output")
