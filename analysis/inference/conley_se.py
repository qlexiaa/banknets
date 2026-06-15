"""
conley_se_comparison.py
=======================
Compares standard-error estimators for the Favara & Imbs (2015) credit
first-stage regression under the same panel as sem_credit.py:

  Dl_nloans_b_it = beta * Linter_bra_it + gamma * X_ct
                    + county FE + year FE + u_it

Estimators (identical point estimate):
  1. State-clustered              -- Favara & Imbs (2015) baseline (state_n)
  2. Spatial HAC bartlett_*km     -- Conley (1999) triangular kernel on haversine
                                      distances; sweep over 100, 250, 500 km
  3. Spatial HAC binary_band      -- Colella et al. (2019) binary connectivity
                                      kernel built from W_bank_binary
  4. Spatial HAC binary_band_knnK -- binary connectivity on W_bank_knn3/knn4
  5. Spatial HAC cosine           -- raw cosine-similarity kernel on W_bank_avg
  6. State + Spatial binary_band  -- Colella et al. (2019) two-way combination

All spatial meats use kernels with unit diagonal (K[c,c]=1), so the
own-observation term (Σ_t u_ct² x̃_ct x̃_{ct}') is correctly included.
Row-standardised weight matrices are NOT used as HAC kernels; they are
designed for the SEM/SAR model operator and structurally shrink the
off-diagonal meat and omit the own-observation contribution.

References: Conley (1999); Colella, Lalive, Sakalli & Thoenig (2019).

The sandwich variance is:
  Var(β̂) = (X'X)⁻¹ B̂ (X'X)⁻¹ × df_correction
  B = Σ_t S_t' K S_t,  where S_t = u_t ∘ x̃_t  (element-wise, shape N×K)

Outputs
-------
  output/conley_se_comparison.csv   -- beta, SE, CI, t-stat, p-value,
                                       kernel, cutoff_km, meat_psd_clip_pct
  output/conley_se_comparison.png   -- publication-quality coefficient plot
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
import utils  # noqa: applies spreg patch
from panel_data import CREDIT_CONTROLS, load_panel_with_credit
from w_variants import load_w_geo, load_bank_variants
from sar.multiplier_decomposition import load_centroids, haversine_matrix

ROOT        = Path(__file__).parents[2]
DATA        = ROOT / "data"
COUNTY_PATH = DATA / "county_order_Wgeo.csv"
DV    = "Dl_nloans_b"
X_VARS = ["Linter_bra"] + CREDIT_CONTROLS

# Bartlett kernel distance cutoff sweep (km)  -- Conley (1999)
CUTOFFS_KM = [100, 250, 500]

ESTIMATOR_LABELS = [
    "State clustering (Favara-Imbs)",
    "Spatial HAC bartlett_100km",
    "Spatial HAC bartlett_250km",
    "Spatial HAC bartlett_500km",
    "Spatial HAC binary_band",
    "Spatial HAC binary_band_knn3",
    "Spatial HAC binary_band_knn4",
    "Spatial HAC cosine",
    "State + Spatial HAC binary_band",
    "State + Spatial HAC binary_band_knn3",
    "State + Spatial HAC binary_band_knn4",
]


# ══════════════════════════════════════════════════════════════════════════════
# Within transformation
# ══════════════════════════════════════════════════════════════════════════════

def two_way_within(arr_TN):
    """
    Two-way (county + year) within transformation for a (T, N) panel array.

    Steps (equivalent to arr - county_mean - year_mean + grand_mean):
      1. Demean by county mean (mean over T): z = arr - arr.mean(axis=0)
      2. Add grand mean back:                 z = z  + arr.mean()
      3. Demean by year mean (mean over N):   z = z  - z.mean(axis=1, keepdims=True)

    The result satisfies: z_it = arr_it - arr_bar_i - arr_bar_t + arr_bar
    which is the standard two-way FE within estimator.
    """
    county_mean = arr_TN.mean(axis=0)
    grand_mean  = float(arr_TN.mean())
    z = arr_TN - county_mean[None, :]
    z = z + grand_mean
    year_mean = z.mean(axis=1, keepdims=True)
    return z - year_mean


def ols_matrix(y_flat, X_flat):
    """OLS after within-transformation for the controlled X matrix."""
    XtX = X_flat.T @ X_flat
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ (X_flat.T @ y_flat)
    u = y_flat - X_flat @ beta
    return beta, u, XtX_inv


# ══════════════════════════════════════════════════════════════════════════════
# Spatial kernel constructors
# ══════════════════════════════════════════════════════════════════════════════

def bartlett_distance_kernel(lats, lons, cutoff_km):
    """
    Conley (1999) triangular kernel on haversine distances.
    K(d) = max(0, 1 − d/cutoff_km).  Diagonal=1 (own-observation term).
    Returns dense (N, N) ndarray.
    """
    D = haversine_matrix(lats, lons)   # (N, N) great-circle distances in km
    K = np.maximum(0.0, 1.0 - D / cutoff_km)
    np.fill_diagonal(K, 1.0)
    return K


def binary_band_kernel(W_raw):
    """
    Colella et al. (2019) binary connectivity kernel.
    (W_raw > 0) → 0/1, symmetrised via ((B + B') > 0), diagonal=1.
    Returns dense (N, N) ndarray.
    """
    if scipy.sparse.issparse(W_raw):
        W = W_raw.toarray().astype(np.float64)
    else:
        W = np.asarray(W_raw, dtype=np.float64).copy()
    B = (W > 0).astype(np.float64)
    B = ((B + B.T) > 0).astype(np.float64)
    np.fill_diagonal(B, 1.0)
    return B


def cosine_kernel(W_cosine_raw, threshold=0.0):
    """
    Raw cosine-similarity kernel: threshold near-zero pairs, symmetrize,
    set diagonal=1.  Continuous weighting following Colella et al. (2019).
    Returns dense (N, N) ndarray.
    """
    if scipy.sparse.issparse(W_cosine_raw):
        W = W_cosine_raw.toarray().astype(np.float64)
    else:
        W = np.asarray(W_cosine_raw, dtype=np.float64).copy()
    np.fill_diagonal(W, 0.0)
    W = np.where(W > threshold, W, 0.0)
    K = (W + W.T) / 2.0
    np.fill_diagonal(K, 1.0)
    return K


def make_psd(B, label=""):
    """
    Symmetrize B and clip negative eigenvalues to enforce PSD.
    Warns if the beta-variance element B[0,0] changes by more than 1%.
    Returns (B_psd, min_eigenvalue, clip_pct).
    """
    B = (B + B.T) / 2.0
    vals, vecs = np.linalg.eigh(B)
    min_val = float(vals.min())
    clip_pct = 0.0
    if min_val < 0:
        orig_var = float(B[0, 0])
        B = (vecs * np.maximum(vals, 0.0)) @ vecs.T
        new_var = float(B[0, 0])
        if abs(orig_var) > 1e-15:
            clip_pct = abs(new_var - orig_var) / abs(orig_var) * 100.0
        if clip_pct > 1.0:
            print(f"  [WARN] PSD clip {label}: beta-var changed {clip_pct:.2f}%")
    print(f"  [{label}] min_eigval={min_val:.4e}  psd_clip={clip_pct:.4f}%")
    return B, min_val, clip_pct


# ══════════════════════════════════════════════════════════════════════════════
# Sandwich meat estimators
# ══════════════════════════════════════════════════════════════════════════════

def meat_spatial(u_TN, X_TNK, K):
    """
    Conley (1999) spatial HAC meat.

    B = Σ_t S_t' K S_t,  where S_t = u_t ∘ x̃_t  (element-wise, shape N×K).

    K must include diagonal=1 (K[c,c]=1) so the own-observation term is
    included.  The diagonal contributes B_het = Σ_t Σ_c s_ct s_ct' (the
    White heteroskedastic meat); off-diagonal entries add cross-county
    spatial covariance with the kernel weight K[c,c'].

    K can be a dense (N, N) ndarray or a scipy sparse matrix.

    Parameters
    ----------
    u_TN  : (T, N) within-transformed residuals
    X_TNK : (T, N, K) within-transformed regressors
    K     : (N, N) kernel with diagonal=1 (dense ndarray or sparse)

    Returns
    -------
    B : (K, K) meat matrix
    """
    nk = X_TNK.shape[2]
    B = np.zeros((nk, nk))
    for t in range(u_TN.shape[0]):
        S_t = u_TN[t, :, None] * X_TNK[t]   # (N, K)
        B += S_t.T @ (K @ S_t)
    return B


def meat_cluster_state(u_TN, X_TNK, county_states):
    """
    State-clustered sandwich meat.

    B_state = Σ_s score_s score_s',
    score_s = Σ_{c∈s} Σ_t u_ct x̃_ct  (K-vector, sum over counties and time).

    Parameters
    ----------
    u_TN          : (T, N) residuals
    X_TNK         : (T, N, K) within-transformed regressors
    county_states : (N,) integer state id for each county in the sample

    Returns
    -------
    B_state : (K, K) meat matrix
    G       : int, number of unique states in the sample
    """
    county_scores = (u_TN[:, :, None] * X_TNK).sum(axis=0)  # (N, K)
    states = np.unique(county_states)
    G = len(states)
    nk = X_TNK.shape[2]
    B = np.zeros((nk, nk))
    for s in states:
        mask        = (county_states == s)
        state_score = county_scores[mask].sum(axis=0)   # (K,)
        B          += np.outer(state_score, state_score)
    return B, G


def meat_twoway_overlap(u_TN, X_TNK, K_band, county_states):
    """
    Overlap meat for the Colella et al. (2019) two-way combination.

    B_overlap = Σ_t S_t' (K_band ∘ SS) S_t,
    where SS_{cc'} = 1{state(c)==state(c')} including the diagonal (c=c').

    K_band should include diagonal=1; the diagonal contribution is in both
    B_state and B_spatial and cancels in B_two = B_state + B_spatial − B_overlap.
    B_overlap captures the within-state AND band-linked pairs that would
    otherwise be double-counted.

    Parameters
    ----------
    u_TN          : (T, N) residuals
    X_TNK         : (T, N, K) within-transformed regressors
    K_band        : (N, N) binary band kernel with diagonal=1 (dense or sparse)
    county_states : (N,) integer state ids

    Returns
    -------
    B : (K, K) overlap meat matrix
    """
    ss = (county_states[:, None] == county_states[None, :]).astype(np.float64)
    if scipy.sparse.issparse(K_band):
        K_arr = K_band.toarray().astype(np.float64)
    else:
        K_arr = np.asarray(K_band, dtype=np.float64)
    K_ss = K_arr * ss

    nk = X_TNK.shape[2]
    B = np.zeros((nk, nk))
    for t in range(u_TN.shape[0]):
        S_t = u_TN[t, :, None] * X_TNK[t]
        B += S_t.T @ (K_ss @ S_t)
    return B


def meat_twoway(B_state, B_spatial_band, B_overlap):
    """
    Colella et al. (2019) two-way combination.
    B_twoway = B_state + B_spatial(band) − B_overlap(band ∩ same-state).
    The unit diagonal in both B_spatial_band and B_overlap cancels.
    """
    return B_state + B_spatial_band - B_overlap


# ══════════════════════════════════════════════════════════════════════════════
# Variance and SE helpers
# ══════════════════════════════════════════════════════════════════════════════

def sandwich_se(XtX_inv, meat, df_corr, param_idx=0):
    """Sandwich SE for one coefficient in the controlled OLS estimator."""
    vcv = XtX_inv @ meat @ XtX_inv * df_corr
    return float(np.sqrt(max(vcv[param_idx, param_idx], 0.0)))


def df_cluster(G):
    """Standard cluster DF adjustment: G / (G - 1)."""
    return G / (G - 1)


def df_conley(NT, k=1):
    """
    User-specified Conley HAC finite-sample DF correction.
    NT / (NT - k - 1)  where k = number of regressors excl. fixed effects.
    """
    return NT / (NT - k - 1)


def stars(p):
    return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))


# ══════════════════════════════════════════════════════════════════════════════
# Sample construction
# ══════════════════════════════════════════════════════════════════════════════

def build_sample(panel_sub, county_order,
                 W_bank_binary_raw_all, W_bank_knn3_raw_all, W_bank_knn4_raw_all,
                 W_bank_cosine_raw_all,
                 centroid_lats, centroid_lons,
                 YEARS, year_pos):
    """
    Build within-transformed arrays for one sample (full or border sub-sample).

    County filter matches panel_fe_credit.build_arrays:
      - Counties present in panel_sub with no NaN in DV or controls.

    HAC kernels are built locally inside compute_all_ses from the raw
    (pre-standardisation) submatrices returned here.  Bartlett kernels
    use haversine distances from centroid_lats/lons (subsetted to usable).

    Returns a dict with all ingredients for sandwich SE computation.
    """
    T = len(YEARS)

    # -- County filter (matches panel_fe_credit.build_arrays exactly) ---------
    any_nan = panel_sub.groupby("fips5")[[DV] + X_VARS].apply(
        lambda g: g.isna().any().any()
    )
    sub_co  = set(panel_sub["fips5"].unique())
    usable  = [c for c in county_order
               if c in sub_co and not any_nan.get(c, True)]
    N       = len(usable)
    usable_pos = {c: i for i, c in enumerate(usable)}

    df = (panel_sub[panel_sub["fips5"].isin(set(usable))]
          .assign(t_idx=lambda d: d["year"].map(year_pos),
                  c_idx=lambda d: d["fips5"].map(usable_pos))
          .sort_values(["t_idx", "c_idx"]))

    y_flat = df[DV].values.astype(np.float64)
    X_flat = df[X_VARS].values.astype(np.float64)

    assert not np.isnan(y_flat).any(), f"NaN in {DV}"
    assert not np.isnan(X_flat).any(), "NaN in controlled X"
    assert len(y_flat) == N * T, f"length mismatch: {len(y_flat)} != {N*T}"

    # -- State mapping ---------------------------------------------------------
    county_states = (
        df.groupby("c_idx")["state_n"]
          .first()
          .sort_index()
          .values.astype(int)
    )

    # -- Two-way within transformation ----------------------------------------
    K = len(X_VARS)
    y_TN  = y_flat.reshape(T, N)
    X_TNK = X_flat.reshape(T, N, K)

    y_tilde = two_way_within(y_TN)
    X_tilde = np.stack(
        [two_way_within(X_TNK[:, :, j]) for j in range(K)],
        axis=2,
    )

    # -- OLS estimate and within residuals ------------------------------------
    beta_vec, u_flat, XtX_inv = ols_matrix(
        y_tilde.flatten(),
        X_tilde.reshape(N * T, K),
    )
    u_TN = u_flat.reshape(T, N)

    # -- Raw submatrices (no row-standardisation) for local kernel building ---
    idx             = np.array([county_order.index(c) for c in usable])
    W_binary_sub    = W_bank_binary_raw_all[idx, :][:, idx]
    W_knn3_sub      = W_bank_knn3_raw_all[idx, :][:, idx]
    W_knn4_sub      = W_bank_knn4_raw_all[idx, :][:, idx]
    W_cosine_sub    = W_bank_cosine_raw_all[idx, :][:, idx]

    lats_sub = centroid_lats[idx] if centroid_lats is not None else None
    lons_sub = centroid_lons[idx] if centroid_lons is not None else None

    return dict(
        N=N, T=T, NT=N * T,
        beta=float(beta_vec[0]), beta_vec=beta_vec, XtX_inv=XtX_inv,
        x_tilde=X_tilde, u_TN=u_TN,
        county_states=county_states,
        W_binary=W_binary_sub,
        W_knn3=W_knn3_sub,
        W_knn4=W_knn4_sub,
        W_cosine=W_cosine_sub,
        lats=lats_sub, lons=lons_sub,
        usable=usable,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SE comparison
# ══════════════════════════════════════════════════════════════════════════════

def compute_all_ses(s):
    """
    Compute all SEs from a build_sample() output dict.

    Kernels are built locally here from the raw submatrices in s:
      - bartlett_*km  : Conley (1999) triangular kernel on haversine distances
      - binary_band   : Colella et al. (2019) 0/1 connectivity, symmetrized
      - cosine        : raw cosine similarity, symmetrized and thresholded
    All kernels have diagonal=1 (own-observation term included).

    Returns dict mapping estimator label -> result row dict, plus '_meta'.
    """
    NT      = s["NT"]
    beta    = s["beta"]
    XtX_inv = s["XtX_inv"]
    u       = s["u_TN"]      # (T, N)
    x       = s["x_tilde"]   # (T, N, K)
    k       = len(X_VARS)

    # -- White / HC own-observation meat (kernel = identity) ------------------
    # B_het = Σ_t Σ_c s_ct s_ct'.  Equivalent to meat_spatial with K=I.
    scores  = u[:, :, None] * x           # (T, N, K)
    B_het   = np.einsum("tnk,tnl->kl", scores, scores)

    # -- iid SE (for ordering acceptance check) --------------------------------
    u_flat  = u.flatten()
    sigma2  = float(np.dot(u_flat, u_flat)) / max(NT - k, 1)
    se_iid  = float(np.sqrt(max(float(XtX_inv[0, 0]) * sigma2, 0.0)))

    # -- DF corrections --------------------------------------------------------
    B_state, G = meat_cluster_state(u, x, s["county_states"])
    dfc = df_cluster(G)
    dfh = df_conley(NT, k)

    # -- Build binary band kernel (two-way uses this) --------------------------
    K_band = binary_band_kernel(s["W_binary"])   # (N, N) dense, diagonal=1

    # -- Two-way: B_overlap uses binary band ∩ same-state ---------------------
    B_spatial_band  = meat_spatial(u, x, K_band)
    B_overlap_band  = meat_twoway_overlap(u, x, K_band, s["county_states"])
    B_two           = meat_twoway(B_state, B_spatial_band, B_overlap_band)

    # Verify that the unit diagonal cancels in B_two (tol 1e-8):
    #   B_two == B_state + (B_spatial_band − B_het) − (B_overlap_band − B_het)
    B_offdiag_band    = B_spatial_band - B_het
    B_offdiag_band_ss = B_overlap_band - B_het
    assert np.allclose(
        B_two,
        B_state + B_offdiag_band - B_offdiag_band_ss,
        atol=1e-8,
    ), "Two-way diagonal-cancellation identity violated"

    # Acceptance: two-way SE invariant to including/excluding the diagonal.
    # Compute with zero-diagonal kernel to confirm.
    K_band_nodiag = K_band.copy()
    np.fill_diagonal(K_band_nodiag, 0.0)
    B_spatial_band_nd = meat_spatial(u, x, K_band_nodiag)
    B_overlap_band_nd = meat_twoway_overlap(u, x, K_band_nodiag, s["county_states"])
    B_two_nd          = B_state + B_spatial_band_nd - B_overlap_band_nd
    se_two_withdiag   = sandwich_se(XtX_inv, B_two, dfc)
    se_two_nodiag     = sandwich_se(XtX_inv, B_two_nd, dfc)
    assert abs(se_two_withdiag - se_two_nodiag) < 1e-8, (
        f"Two-way SE not invariant to diagonal: "
        f"{se_two_withdiag:.10f} vs {se_two_nodiag:.10f}"
    )

    # Apply PSD guard to two-way meat
    B_two_psd, two_min_eig, two_clip_pct = make_psd(B_two, "two_way_binary_band")
    se_two = sandwich_se(XtX_inv, B_two_psd, dfc)

    # -- State-cluster SE (no kernel / PSD guard needed) ----------------------
    se_state = sandwich_se(XtX_inv, B_state, dfc)

    # -- Diagnostic: fraction of W_bank_binary nonzero pairs that are same-state
    # The value-weighted B_overlap/B_spatial ratio is expected near 1 even when
    # the count-based frac_same_state is well below 1: within-state county pairs
    # tend to have correlated residuals, so same-state pairs dominate the
    # quadratic form B_overlap even when they are a minority of total linked pairs.
    _W_arr = s["W_binary"].toarray() if scipy.sparse.issparse(s["W_binary"]) else s["W_binary"]
    _ss    = (s["county_states"][:, None] == s["county_states"][None, :])
    np.fill_diagonal(_ss, False)
    _nnz_band = int((_W_arr > 0).sum())
    _nnz_same = int(((_W_arr > 0) & _ss).sum())
    frac_same_state_band = (_nnz_same / _nnz_band) if _nnz_band > 0 else 0.0

    # -- Helper for result rows -----------------------------------------------
    def build_row(se, kernel, cutoff_km=float("nan"), clip_pct=0.0):
        lo   = beta - 1.96 * se
        hi   = beta + 1.96 * se
        tval = beta / se if se > 0 else np.inf
        pval = float(2 * st.norm.sf(abs(tval)))
        return dict(
            beta=beta, se=se,
            ci_lower=lo, ci_upper=hi, ci_width=hi - lo,
            t_stat=tval, p_value=pval,
            kernel=kernel,
            cutoff_km=cutoff_km,
            meat_psd_clip_pct=clip_pct,
        )

    result = {}
    result["State clustering (Favara-Imbs)"] = build_row(
        se_state, kernel="cluster", clip_pct=0.0)

    # -- Bartlett (geographic Conley) kernels ----------------------------------
    has_centroids = (s["lats"] is not None)
    for cutoff_km in CUTOFFS_KM:
        label = f"Spatial HAC bartlett_{cutoff_km}km"
        if has_centroids:
            K_bart       = bartlett_distance_kernel(s["lats"], s["lons"], cutoff_km)
            B_bart_raw   = meat_spatial(u, x, K_bart)
            B_bart, min_eig, clip_pct = make_psd(B_bart_raw, f"bartlett_{cutoff_km}km")
            se_bart      = sandwich_se(XtX_inv, B_bart, dfh)
        else:
            se_bart, min_eig, clip_pct = float("nan"), float("nan"), 0.0
        result[label] = build_row(se_bart,
                                  kernel="bartlett",
                                  cutoff_km=float(cutoff_km),
                                  clip_pct=clip_pct)

    # -- Binary band kernel (W_bank_binary) ------------------------------------
    B_band_psd, band_min_eig, band_clip_pct = make_psd(
        B_spatial_band, "binary_band")
    se_band = sandwich_se(XtX_inv, B_band_psd, dfh)
    result["Spatial HAC binary_band"] = build_row(
        se_band, kernel="binary_band", clip_pct=band_clip_pct)

    # -- Binary band kernels (KNN variants): standalone spatial + two-way ------
    for knn_label, W_knn_sub in [("knn3", s["W_knn3"]), ("knn4", s["W_knn4"])]:
        K_knn          = binary_band_kernel(W_knn_sub)
        B_knn_spatial  = meat_spatial(u, x, K_knn)
        B_knn_overlap  = meat_twoway_overlap(u, x, K_knn, s["county_states"])
        B_knn_two      = meat_twoway(B_state, B_knn_spatial, B_knn_overlap)

        # Diagonal-cancellation identity for this knn variant
        B_knn_offdiag    = B_knn_spatial - B_het
        B_knn_offdiag_ss = B_knn_overlap  - B_het
        assert np.allclose(
            B_knn_two,
            B_state + B_knn_offdiag - B_knn_offdiag_ss,
            atol=1e-8,
        ), f"Two-way diagonal-cancellation identity violated ({knn_label})"

        # Standalone spatial SE
        B_knn_psd, _, knn_clip_pct = make_psd(B_knn_spatial, f"binary_band_{knn_label}")
        se_knn = sandwich_se(XtX_inv, B_knn_psd, dfh)
        result[f"Spatial HAC binary_band_{knn_label}"] = build_row(
            se_knn, kernel="binary_band", clip_pct=knn_clip_pct)

        # Two-way SE
        B_knn_two_psd, _, knn_two_clip = make_psd(B_knn_two, f"two_way_{knn_label}")
        se_knn_two = sandwich_se(XtX_inv, B_knn_two_psd, dfc)
        result[f"State + Spatial HAC binary_band_{knn_label}"] = build_row(
            se_knn_two, kernel=f"two_way_binary_band_{knn_label}", clip_pct=knn_two_clip)

    # -- Cosine kernel (raw pre-standardisation cosine similarity) -------------
    K_cos       = cosine_kernel(s["W_cosine"], threshold=0.0)
    B_cos_raw   = meat_spatial(u, x, K_cos)
    B_cos, cos_min_eig, cos_clip_pct = make_psd(B_cos_raw, "cosine")
    se_cos      = sandwich_se(XtX_inv, B_cos, dfh)
    result["Spatial HAC cosine"] = build_row(
        se_cos, kernel="cosine", clip_pct=cos_clip_pct)

    # -- Two-way (state + binary band) ----------------------------------------
    result["State + Spatial HAC binary_band"] = build_row(
        se_two, kernel="two_way_binary_band", clip_pct=two_clip_pct)

    # -- Acceptance checks ----------------------------------------------------
    print(f"  Two-way diagonal invariance: |SE_with - SE_without| = "
          f"{abs(se_two_withdiag - se_two_nodiag):.2e}  (tol 1e-8) OK")
    print(f"  iid SE = {se_iid:.4f}  |  state-cluster SE = {se_state:.4f}")

    # Flag spatial SEs that are suspiciously small or yield very large t-stats
    _two_way_labels = {lb for lb in result if lb.startswith("State + Spatial")}
    for label, row in result.items():
        if label in ({"State clustering (Favara-Imbs)"} | _two_way_labels):
            continue
        tval = row["t_stat"]
        if not np.isfinite(tval):
            continue
        if abs(tval) > 10:
            print(f"  [WARN] {label}: t-stat={tval:.2f} — suspect, check SE ordering")
        if row["se"] < se_iid * 0.5:
            print(f"  [WARN] {label}: SE={row['se']:.4f} < 0.5×iid SE={se_iid:.4f} — "
                  "own-term likely missing")

    # -- Two-way overlap diagnostic -------------------------------------------
    _ratio = (float(B_spatial_band[0, 0]) / float(B_band_psd[0, 0])
              if float(B_band_psd[0, 0]) != 0 else float("nan"))
    _ratio_overlap = (float(B_overlap_band[0, 0]) / float(B_spatial_band[0, 0])
                      if float(B_spatial_band[0, 0]) != 0 else float("nan"))
    print(f"  Two-way diagnostic (binary_band):")
    print(f"    B_state[0,0]   = {float(B_state[0, 0]):>12.6f}")
    print(f"    B_spatial[0,0] = {float(B_spatial_band[0, 0]):>12.6f}  (binary band, diag=1)")
    print(f"    B_overlap[0,0] = {float(B_overlap_band[0, 0]):>12.6f}  (same-state AND band-linked)")
    print(f"    B_overlap / B_spatial = {_ratio_overlap:.4f}"
          "  [note: near 1 expected under state-year treatment --")
    print(f"       within-state residual covariance dominates the quadratic form")
    print(f"       even when count-based frac_same_state_band << 1]")
    print(f"    Band nonzero pairs: {_nnz_band:,} total | {_nnz_same:,} same-state"
          f" | {frac_same_state_band*100:.2f}% same-state (count-based)")

    result["_meta"] = dict(
        N=s["N"], T=s["T"], G=G,
        se_iid=se_iid,
        B_het=float(B_het[0, 0]),
        B_state=float(B_state[0, 0]),
        B_spatial_band=float(B_spatial_band[0, 0]),
        B_overlap_band=float(B_overlap_band[0, 0]),
        B_two=float(B_two[0, 0]),
        dfc=dfc, dfh=dfh,
        frac_same_state_band=frac_same_state_band,
        nnz_band=_nnz_band, nnz_band_same=_nnz_same,
        two_min_eig=two_min_eig,
        two_clip_pct=two_clip_pct,
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Publication-quality coefficient plot
# ══════════════════════════════════════════════════════════════════════════════

_COLORS = {
    "State clustering (Favara-Imbs)":           "#2166ac",
    "Spatial HAC bartlett_100km":                "#4dac26",
    "Spatial HAC bartlett_250km":                "#1a9641",
    "Spatial HAC bartlett_500km":                "#a6d96a",
    "Spatial HAC binary_band":                   "#d01c8b",
    "Spatial HAC binary_band_knn3":              "#f1b6da",
    "Spatial HAC binary_band_knn4":              "#c51b7d",
    "Spatial HAC cosine":                        "#fc8d59",
    "State + Spatial HAC binary_band":           "#e66101",
    "State + Spatial HAC binary_band_knn3":      "#b35806",
    "State + Spatial HAC binary_band_knn4":      "#fdb863",
}
_SHORT = [
    "State\nclustering",
    "Bartlett\n100km",
    "Bartlett\n250km",
    "Bartlett\n500km",
    "Binary\nband",
    "Binary\nknn3",
    "Binary\nknn4",
    "Cosine",
    "Two-way\nband",
    "Two-way\nknn3",
    "Two-way\nknn4",
]


def _make_coef_plot(all_results):
    sample_labels = list(all_results.keys())
    n = len(sample_labels)
    fig, axes = plt.subplots(1, n, figsize=(14, 5), sharey=True)
    if n == 1:
        axes = [axes]

    xs = np.arange(len(ESTIMATOR_LABELS))
    for ax, slabel in zip(axes, sample_labels):
        res  = all_results[slabel]
        meta = res["_meta"]
        for i, (label, short) in enumerate(zip(ESTIMATOR_LABELS, _SHORT)):
            r     = res[label]
            color = _COLORS[label]
            ax.vlines(xs[i], r["ci_lower"], r["ci_upper"],
                      color=color, linewidth=2.5, zorder=3)
            ax.hlines([r["ci_lower"], r["ci_upper"]],
                      xs[i] - 0.14, xs[i] + 0.14,
                      color=color, linewidth=1.5, zorder=3)
            ax.scatter(xs[i], r["beta"],
                       color=color, s=55, zorder=5,
                       edgecolors="white", linewidths=0.8)

        ax.axhline(0, color="0.6", linewidth=0.8, linestyle=":", zorder=1)
        ax.set_xticks(xs)
        ax.set_xticklabels(_SHORT, fontsize=7.5, ha="center", va="top")
        ax.tick_params(axis="x", pad=4)
        ax.set_title(
            f"{slabel}\n$N={meta['N']:,}$ counties, ${meta['N']*meta['T']:,}$ obs",
            fontsize=10, pad=8,
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(-0.65, len(ESTIMATOR_LABELS) - 0.35)
        ax.tick_params(axis="y", labelsize=9)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.3f"))

    axes[0].set_ylabel(r"$\hat{\beta}$ on Linter$\_$bra", fontsize=10)

    fig.suptitle(
        "Conley / Colella Spatial HAC vs State-Clustered Standard Errors\n"
        r"OLS: $\Delta\ln(\mathrm{loans}_{b,it}) = "
        r"\beta\,\mathrm{Linter\_bra}_{it}$ + county FE + year FE",
        fontsize=10, y=1.03,
    )

    legend_handles = [
        Line2D([0], [0], color=_COLORS[lb], linewidth=2.5,
               marker="o", markersize=6, label=lb.replace("\n", " "))
        for lb in ESTIMATOR_LABELS
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", bbox_to_anchor=(0.5, -0.12),
        ncol=3, fontsize=8.0, frameon=False,
    )
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Main run
# ══════════════════════════════════════════════════════════════════════════════

def run(output_dir=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # -- Load county order and W_geo for gal-order assertion ------------------
    co_df        = pd.read_csv(COUNTY_PATH, dtype={"fips5": str})
    county_order = co_df["fips5"].str.zfill(5).tolist()

    W_geo_all, gal_order = load_w_geo(county_order)
    assert gal_order == county_order, "GAL order mismatch"

    # Trigger build of knn3/knn4/binary caches (idempotent after first run).
    # load_bank_variants returns row-standardised variants, which we do NOT use
    # as HAC kernels; we load the raw npz files directly below.
    print("Ensuring bank-variant caches are built ...", flush=True)
    _ = load_bank_variants(county_order, W_geo_all=W_geo_all)

    # -- Load raw (pre-standardisation) bank matrices -------------------------
    W_bank_binary_raw = scipy.sparse.load_npz(DATA / "W_bank_binary.npz")
    W_bank_knn3_raw   = scipy.sparse.load_npz(DATA / "W_bank_knn3.npz")
    W_bank_knn4_raw   = scipy.sparse.load_npz(DATA / "W_bank_knn4.npz")
    W_bank_cosine_raw = scipy.sparse.load_npz(DATA / "W_bank_avg.npz")

    # -- Load county centroids for bartlett kernel ----------------------------
    print("Loading county centroids for Bartlett distance kernel ...", flush=True)
    centroid_result = load_centroids(county_order)
    if centroid_result is not None:
        c_lats, c_lons = centroid_result
        print(f"  Centroids ready for {len(c_lats)} counties.")
    else:
        c_lats, c_lons = None, None
        print("  [WARN] Centroids unavailable; bartlett rows will be NaN.")

    # -- Panel ----------------------------------------------------------------
    panel = load_panel_with_credit()
    panel["fips5"] = panel["fips5"].astype(str).str.zfill(5)
    YEARS    = sorted(panel["year"].unique())
    year_pos = {yr: i for i, yr in enumerate(YEARS)}

    panel_contig    = panel[panel["border"] == 1].copy()
    panel_noncontig = panel[panel["border"] == 0].copy()
    samples = [("Full", panel), ("Contig", panel_contig), ("NonContig", panel_noncontig)]

    # -- Estimate + compute SEs for each sample -------------------------------
    all_results = {}
    for slabel, panel_sub in samples:
        print(f"\nBuilding {slabel} sample within arrays ...", flush=True)
        s = build_sample(
            panel_sub, county_order,
            W_bank_binary_raw, W_bank_knn3_raw, W_bank_knn4_raw,
            W_bank_cosine_raw,
            c_lats, c_lons,
            YEARS, year_pos,
        )
        print(f"  N={s['N']} counties | T={s['T']} | NT={s['NT']:,} obs")
        print(f"  beta_hat = {s['beta']:.6f}  (two-way FWL)")
        print(f"  Computing spatial meats ...", flush=True)
        res = compute_all_ses(s)
        all_results[slabel] = res

    # -- Print formatted tables -----------------------------------------------
    W = 82
    csv_rows = []
    for slabel, res in all_results.items():
        m    = res["_meta"]
        N, T = m["N"], m["T"]
        print()
        print("=" * W)
        print(f"{slabel} sample  (N={N} counties, {N*T:,} obs, G={m['G']} states)")
        print("-" * W)
        print(f"{'Estimator':<40} {'beta':>8}  {'SE':>7}  "
              f"{'95% CI':>20}  {'CI width':>8}")
        print("-" * W)

        for label in ESTIMATOR_LABELS:
            r  = res[label]
            ci = f"[{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]"
            print(f"  {label:<38} {r['beta']:8.4f}  "
                  f"{r['se']:7.4f}  {ci:>22}  {r['ci_width']:8.4f}")
            csv_rows.append(dict(
                sample             = slabel,
                estimator          = label,
                kernel             = r["kernel"],
                cutoff_km          = r["cutoff_km"],
                meat_psd_clip_pct  = r["meat_psd_clip_pct"],
                beta               = r["beta"],
                se                 = r["se"],
                ci_lower           = r["ci_lower"],
                ci_upper           = r["ci_upper"],
                ci_width           = r["ci_width"],
                t_stat             = r["t_stat"],
                p_value            = r["p_value"],
                B_state            = m["B_state"],
                B_spatial_band     = m["B_spatial_band"],
                B_overlap_band     = m["B_overlap_band"],
                frac_same_state_band = m["frac_same_state_band"],
            ))

        print("-" * W)
        ref_width = res["State clustering (Favara-Imbs)"]["ci_width"]
        print("CI width vs state clustering:")
        for label in ESTIMATOR_LABELS[1:]:
            pct = (res[label]["ci_width"] / ref_width - 1) * 100
            print(f"  {label:<40}  {pct:+.1f}%")
        print("=" * W)

    # -- Acceptance checks (Full sample) --------------------------------------
    if "Full" in all_results:
        res_f = all_results["Full"]
        m_f   = res_f["_meta"]
        fi_se   = res_f["State clustering (Favara-Imbs)"]["se"]
        fi_beta = res_f["State clustering (Favara-Imbs)"]["beta"]
        se_iid  = m_f["se_iid"]
        print()
        print("-- Acceptance checks (Full sample) ---------------------------------")
        print(f"  State-cluster SE : beta={fi_beta:.4f}  SE={fi_se:.4f}"
              f"  [target ~0.011 (F&I Table 2 baseline)]")
        print(f"  iid SE           : {se_iid:.4f}")
        print(f"  Two-way SE (band): {res_f['State + Spatial HAC binary_band']['se']:.4f}")
        print()
        print("  Spatial SE ordering (should be between iid and cluster SE, or above):")
        spatial_labels = [lb for lb in ESTIMATOR_LABELS
                          if lb != "State clustering (Favara-Imbs)"
                          and not lb.startswith("State + Spatial")]
        for lb in spatial_labels:
            r = res_f[lb]
            flag = ""
            if np.isfinite(r["se"]) and r["se"] < se_iid * 0.5:
                flag = " *** OWN TERM MISSING?"
            if np.isfinite(r["t_stat"]) and abs(r["t_stat"]) > 10:
                flag += " *** t > 10, SUSPECT"
            print(f"    {lb:<40}  SE={r['se']:.4f}  t={r['t_stat']:.2f}{flag}")
        print("--------------------------------------------------------------------")

    # -- Save CSV -------------------------------------------------------------
    cols = [
        "sample", "estimator", "kernel", "cutoff_km", "meat_psd_clip_pct",
        "beta", "se", "ci_lower", "ci_upper", "ci_width", "t_stat", "p_value",
        "B_state", "B_spatial_band", "B_overlap_band", "frac_same_state_band",
    ]
    if output_dir is not None:
        pd.DataFrame(csv_rows)[cols].to_csv(
            output_dir / "conley_se_comparison.csv", index=False)
        print(f"\nSaved conley_se_comparison.csv to {output_dir}")

    # -- Plot -----------------------------------------------------------------
    fig = _make_coef_plot(all_results)
    if output_dir is not None:
        fig.savefig(output_dir / "conley_se_comparison.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved conley_se_comparison.png to {output_dir}")

    return all_results


if __name__ == "__main__":
    run(Path(__file__).parents[2] / "output")
