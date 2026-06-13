"""
conley_se_comparison.py
=======================
Compares four standard-error estimators for the Favara & Imbs (2015) credit
first-stage regression under the same panel as sem_credit.py:

  Dl_nloans_b_it = beta * Linter_bra_it + gamma * X_ct
                    + county FE + year FE + u_it

Four estimators, identical point estimate:
  1. State-clustered         -- Favara & Imbs (2015) baseline, clustered by state_n
  2. Spatial HAC W_geo       -- Conley (1999) sandwich under queen contiguity
  3. Spatial HAC W_bank      -- Conley (1999) sandwich under bank-network weights
  4. State + Spatial W_bank  -- Colella et al. (2019) additive two-way combination

The Conley (1999) sandwich estimator is implemented from scratch in numpy,
with the two-way combination following Colella, Lalive, Sakalli & Thoenig
(2019).

The sandwich variance is:

  Var(beta_hat) = (X'X)^{-1}  B_hat  (X'X)^{-1}  *  df_correction

For the spatial meat (estimators 2, 3):

  B_spatial = sum_t  sum_c  sum_{c'}  w_{cc'} u_ct u_{c't} x_tilde_ct x_tilde_{c't}
            = sum_t  v_t' W v_t          (v_t = u_t o x_tilde_t, element-wise)

For the state-cluster meat (estimator 1):

  B_state = sum_s  (sum_{c in s, t}  u_ct x_tilde_ct)^2

For the two-way meat (estimator 4) following Colella et al. (2019):

  B_twoway = B_state + B_spatial(W_bank) - B_OLS
  B_OLS    = sum_{c,t}  u_ct^2 x_tilde_ct^2   (HC overlap term)

Outputs
-------
  output/conley_se_comparison.csv   -- beta, SE, CI, t-stat, p-value
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
from utils import row_standardize
from panel_data import CREDIT_CONTROLS, load_panel_with_credit
from w_variants import load_w_geo, load_bank_variants

ROOT        = Path(__file__).parents[2]
COUNTY_PATH = ROOT / "data" / "county_order_Wgeo.csv"
DV = "Dl_nloans_b"
X_VARS = ["Linter_bra"] + CREDIT_CONTROLS

ESTIMATOR_LABELS = [
    "State clustering (Favara-Imbs)",
    "Spatial HAC W_geo",
    "Spatial HAC W_bank",
    "Spatial HAC W_bank_knn3",
    "Spatial HAC W_bank_knn4",
    "State + Spatial HAC W_bank",
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
    county_mean = arr_TN.mean(axis=0)             # (N,) -- mean over time
    grand_mean  = float(arr_TN.mean())             # scalar
    z = arr_TN - county_mean[None, :]             # step 1: subtract county mean
    z = z + grand_mean                            # step 2: restore grand mean
    year_mean = z.mean(axis=1, keepdims=True)     # (T,1) -- mean over counties
    return z - year_mean                          # step 3: subtract year mean


def ols_matrix(y_flat, X_flat):
    """OLS after within-transformation for the controlled X matrix."""
    XtX = X_flat.T @ X_flat
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ (X_flat.T @ y_flat)
    u = y_flat - X_flat @ beta
    return beta, u, XtX_inv


# ══════════════════════════════════════════════════════════════════════════════
# Sandwich meat estimators
# ══════════════════════════════════════════════════════════════════════════════

def meat_spatial(u_TN, X_TNK, W_sp):
    """
    Conley (1999) spatial HAC meat for a controlled regressor matrix.

    B_spatial = sum_t  sum_c  sum_{c'}  w_{cc'} u_ct u_{c't} x_ct x_{c't}
              = sum_t  v_t' W v_t

    where v_t = u_t o x_tilde_t  (element-wise product, shape N).

    Implementation uses sparse matrix-vector multiply W @ v_t with cost
    O(nnz(W)) per time period -- no explicit county-pair loop.
    Total complexity: O(T * nnz(W)).

    Parameters
    ----------
    u_TN : (T, N) within-transformed OLS residuals (time-major)
    X_TNK : (T, N, K) within-transformed regressors
    W_sp : (N, N) scipy sparse weight matrix, used as-is (no re-standardisation)

    Returns
    -------
    B : (K, K) meat matrix
    """
    K = X_TNK.shape[2]
    B = np.zeros((K, K))
    for t in range(u_TN.shape[0]):
        S_t = u_TN[t, :, None] * X_TNK[t]   # (N, K)
        B += S_t.T @ (W_sp @ S_t)
    return B


def meat_cluster_state(u_TN, X_TNK, county_states):
    """
    State-clustered sandwich meat.

    B_state = sum_s  score_s^2
    score_s = sum_{c in s}  sum_t  u_ct * x_tilde_ct

    The county-level score is first summed over time, then aggregated within
    each state. For k=1, score_s is a scalar and B_state is a sum of squares.

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
    # Sum score over time periods for each county: shape (N, K)
    county_scores = (u_TN[:, :, None] * X_TNK).sum(axis=0)

    states = np.unique(county_states)
    G = len(states)
    K = X_TNK.shape[2]
    B = np.zeros((K, K))
    for s in states:
        mask        = (county_states == s)
        state_score = county_scores[mask].sum(axis=0)
        B          += np.outer(state_score, state_score)
    return B, G


def meat_twoway_overlap(u_TN, X_TNK, W_sp, county_states):
    """
    True overlap term for the Colella et al. (2019) two-way SE correction.

    B_overlap = sum_t sum_{c,c'} w_{cc'} * 1{state(c)==state(c')}
                                      * u_ct u_{c't} X_ct X_{c't}'

    This is the part that BOTH B_state and B_spatial share: within-state
    county pairs that are also spatially linked (w_{cc'} > 0).  Subtracting
    B_overlap (rather than the HC diagonal B_OLS) avoids double-counting
    these cross-county same-state contributions.

    Key distinction from the previous B_OLS approach:
      - B_OLS = sum_{c,t} u_ct^2 x_ct^2  (diagonal of W is zero, so spatial
        meat has NO own-county contribution; B_OLS is the within-county term
        that only B_state counts — not spatial).
      - The actual double-counted region is within-state AND linked pairs,
        i.e. B_overlap above, NOT B_OLS.
      - Using B_OLS as the overlap therefore underestimates the subtraction,
        potentially inflating the two-way SE when many within-state bank links
        exist (dense W_bank for border-MSA counties sharing a state).

    For W matrices with no within-state links (e.g. W_bank_interstate),
    B_overlap = 0 and B_twoway = B_state + B_spatial exactly.
    """
    N = u_TN.shape[1]
    # Build a within-state mask matrix (N x N), True if same state
    same_state = (county_states[:, None] == county_states[None, :])  # (N, N) bool

    # Element-wise mask of W: keep only within-state links
    # W_sp is scipy sparse; convert to dense for element-wise product
    W_dense  = W_sp.toarray()                          # (N, N)
    W_masked = W_dense * same_state.astype(np.float64) # (N, N), within-state links only

    K = X_TNK.shape[2]
    B = np.zeros((K, K))
    for t in range(u_TN.shape[0]):
        S_t = u_TN[t, :, None] * X_TNK[t]
        B += S_t.T @ (W_masked @ S_t)
    return B


def meat_twoway(B_state, B_spatial_bank, B_overlap):
    """
    Colella et al. (2019) additive two-way combination with correct overlap.

    B_twoway = B_state + B_spatial(W_bank) - B_overlap

    B_overlap is the contribution that appears in BOTH B_state (all within-state
    pairs, summed over time) and B_spatial (all w_{cc'}>0 pairs).  Their
    intersection is: pairs that are BOTH in the same state AND spatially linked.
    Subtracting B_overlap once gives the union without double-counting.

    Note: the spatial kernel W has zero diagonal, so own-county pairs are
    in B_state but NOT in B_spatial — the naive subtraction of B_OLS
    (which equals the own-county diagonal sum) is incorrect because it
    removes a term that was never in B_spatial.
    """
    return B_state + B_spatial_bank - B_overlap


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

def build_sample(panel_sub, county_order, W_geo_all, W_bank_all,
                 W_knn3_all, W_knn4_all, YEARS, year_pos):
    """
    Build within-transformed arrays for one sample (full or border).

    County filter is identical to panel_fe_credit.build_arrays:
      - Counties present in panel_sub
      - No county with any NaN in Dl_nloans_b or controlled regressors

    W submatrices are NOT re-standardised (per user spec -- W_*_all is already
    row-standardised; subsetting introduces minor row-sum deviations from 1 but
    the user explicitly requests using them as-is).

    Returns a dict with all ingredients for sandwich SE computation.
    """
    T = len(YEARS)

    # -- County filter (matches panel_fe_credit.build_arrays exactly) ---------
    any_nan    = panel_sub.groupby("fips5")[[DV] + X_VARS].apply(
        lambda g: g.isna().any().any()
    )
    sub_co     = set(panel_sub["fips5"].unique())
    usable     = [c for c in county_order
                  if c in sub_co and not any_nan.get(c, True)]
    N          = len(usable)
    usable_pos = {c: i for i, c in enumerate(usable)}

    df = (panel_sub[panel_sub["fips5"].isin(set(usable))]
          .assign(t_idx=lambda d: d["year"].map(year_pos),
                  c_idx=lambda d: d["fips5"].map(usable_pos))
          .sort_values(["t_idx", "c_idx"]))

    y_flat = df[DV].values.astype(np.float64)            # (N*T,) time-major
    X_flat = df[X_VARS].values.astype(np.float64)         # (N*T,K)

    assert not np.isnan(y_flat).any(), f"NaN in {DV}"
    assert not np.isnan(X_flat).any(), "NaN in controlled X"
    assert len(y_flat) == N * T,       f"length mismatch: {len(y_flat)} != {N*T}"

    # -- State mapping: one state per county, time-invariant ------------------
    county_states = (
        df.groupby("c_idx")["state_n"]
          .first()
          .sort_index()
          .values.astype(int)     # (N,) -- state_n for county i = 0 .. N-1
    )

    # -- Two-way within transformation ----------------------------------------
    # Reshape to (T, N) for vectorised ops; data is sorted time-major
    K = len(X_VARS)
    y_TN = y_flat.reshape(T, N)
    X_TNK = X_flat.reshape(T, N, K)

    y_tilde = two_way_within(y_TN)    # (T, N)
    X_tilde = np.stack(
        [two_way_within(X_TNK[:, :, j]) for j in range(K)],
        axis=2,
    )

    # -- OLS estimate and within residuals ------------------------------------
    beta_vec, u_flat, XtX_inv = ols_matrix(
        y_tilde.flatten(),
        X_tilde.reshape(N * T, K),
    )
    u_TN = u_flat.reshape(T, N)       # (T, N)

    # -- W submatrices (no re-standardisation per spec) -----------------------
    idx        = np.array([county_order.index(c) for c in usable])
    W_geo_sub  = W_geo_all[idx, :][:, idx]    # (N, N) sparse
    W_bank_sub = W_bank_all[idx, :][:, idx]   # (N, N) sparse
    W_knn3_sub = W_knn3_all[idx, :][:, idx]   # (N, N) sparse
    W_knn4_sub = W_knn4_all[idx, :][:, idx]   # (N, N) sparse

    return dict(
        N=N, T=T, NT=N * T,
        beta=float(beta_vec[0]), beta_vec=beta_vec, XtX_inv=XtX_inv,
        x_tilde=X_tilde, u_TN=u_TN,
        county_states=county_states,
        W_geo=W_geo_sub, W_bank=W_bank_sub,
        W_knn3=W_knn3_sub, W_knn4=W_knn4_sub,
        usable=usable,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SE comparison
# ══════════════════════════════════════════════════════════════════════════════

def compute_all_ses(s):
    """
    Compute all four SEs from a build_sample() output dict.

    Returns dict mapping estimator label -> result row dict, plus '_meta'.
    """
    NT   = s["NT"]
    beta = s["beta"]
    XtX_inv = s["XtX_inv"]
    u    = s["u_TN"]      # (T, N)
    x    = s["x_tilde"]   # (T, N, K)
    k    = len(X_VARS)

    # -- Compute raw meats ----------------------------------------------------
    B_geo            = meat_spatial(u, x, s["W_geo"])
    B_bank           = meat_spatial(u, x, s["W_bank"])
    B_knn3           = meat_spatial(u, x, s["W_knn3"])
    B_knn4           = meat_spatial(u, x, s["W_knn4"])
    B_state, G       = meat_cluster_state(u, x, s["county_states"])
    # True overlap: within-state AND spatially linked pairs
    B_overlap        = meat_twoway_overlap(u, x, s["W_bank"], s["county_states"])
    # Old overlap (diagonal / HC): kept for se_twoway_old column
    scores           = u[:, :, None] * x
    B_het            = np.einsum("tnk,tnl->kl", scores, scores)
    B_two            = meat_twoway(B_state, B_bank, B_overlap)
    B_two_old        = B_state + B_bank - B_het  # previous (incorrect) formula

    # -- Diagnostic: fraction of W_bank nonzero pairs that are same-state ----
    # Sanity check for meat_twoway_overlap correctness:
    #   If B_overlap / B_bank is close to 1, the same-state mask is suspect.
    #   Expected: << 1 because W_bank_interstate >> W_bank_intrastate.
    _W_bnk = s["W_bank"].toarray()
    _ss    = (s["county_states"][:, None] == s["county_states"][None, :])
    np.fill_diagonal(_ss, False)               # exclude diagonal
    _nnz_bank = int((_W_bnk > 0).sum())
    _nnz_same = int(((_W_bnk > 0) & _ss).sum())
    frac_same_state_bank = (_nnz_same / _nnz_bank) if _nnz_bank > 0 else 0.0

    # -- DF corrections -------------------------------------------------------
    dfc = df_cluster(G)        # G/(G-1)           -- for cluster estimators
    dfh = df_conley(NT, k)     # NT/(NT-k-1)       -- for Conley HAC

    # -- Standard errors ------------------------------------------------------
    se_state   = sandwich_se(XtX_inv, B_state, dfc)
    se_geo     = sandwich_se(XtX_inv, B_geo,   dfh)
    se_bank    = sandwich_se(XtX_inv, B_bank,  dfh)
    se_knn3    = sandwich_se(XtX_inv, B_knn3,  dfh)
    se_knn4    = sandwich_se(XtX_inv, B_knn4,  dfh)
    se_two     = sandwich_se(XtX_inv, B_two,   dfc)  # cluster DF for twoway
    se_two_old = sandwich_se(XtX_inv, B_two_old, dfc)

    # -- CIs, t-stats, p-values (normal approximation) -----------------------
    z_crit = st.norm.ppf(0.975)   # 1.96

    def build_row(se):
        lo   = beta - z_crit * se
        hi   = beta + z_crit * se
        tval = beta / se if se > 0 else np.inf
        pval = float(2 * st.norm.sf(abs(tval)))
        return dict(beta=beta, se=se,
                    ci_lower=lo, ci_upper=hi,
                    ci_width=hi - lo,
                    t_stat=tval, p_value=pval)

    r_two = build_row(se_two)
    r_two["se_twoway_old"] = se_two_old
    result = {
        "State clustering (Favara-Imbs)": build_row(se_state),
        "Spatial HAC W_geo":              build_row(se_geo),
        "Spatial HAC W_bank":             build_row(se_bank),
        "Spatial HAC W_bank_knn3":        build_row(se_knn3),
        "Spatial HAC W_bank_knn4":        build_row(se_knn4),
        "State + Spatial HAC W_bank":     r_two,
    }
    result["_meta"] = dict(
        N=s["N"], T=s["T"], G=G,
        B_het=float(B_het[0, 0]),
        B_geo=float(B_geo[0, 0]),
        B_bank=float(B_bank[0, 0]),
        B_knn3=float(B_knn3[0, 0]),
        B_knn4=float(B_knn4[0, 0]),
        B_state=float(B_state[0, 0]),
        B_overlap=float(B_overlap[0, 0]),
        B_two=float(B_two[0, 0]),
        B_two_old=float(B_two_old[0, 0]),
        dfc=dfc, dfh=dfh,
        frac_same_state_bank=frac_same_state_bank,
        nnz_bank=_nnz_bank, nnz_bank_same=_nnz_same,
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Publication-quality coefficient plot
# ══════════════════════════════════════════════════════════════════════════════

_COLORS = {
    "State clustering (Favara-Imbs)": "#2166ac",
    "Spatial HAC W_geo":              "#4dac26",
    "Spatial HAC W_bank":             "#d01c8b",
    "Spatial HAC W_bank_knn3":        "#a6cee3",
    "Spatial HAC W_bank_knn4":        "#b2df8a",
    "State + Spatial HAC W_bank":     "#e66101",
}
_SHORT = [
    "State\nclustering",
    "Spatial HAC\n$W_{\\rm geo}$",
    "Spatial HAC\n$W_{\\rm bank}$",
    "Spatial HAC\n$W_{\\rm knn3}$",
    "Spatial HAC\n$W_{\\rm knn4}$",
    "Two-way\n(state + $W_{\\rm bank}$)",
]


def _make_coef_plot(all_results):
    sample_labels = list(all_results.keys())
    n = len(sample_labels)
    fig, axes = plt.subplots(1, n, figsize=(10, 5), sharey=True)
    if n == 1:
        axes = [axes]

    xs = np.arange(len(ESTIMATOR_LABELS))
    for ax, slabel in zip(axes, sample_labels):
        res  = all_results[slabel]
        meta = res["_meta"]
        for i, (label, short) in enumerate(zip(ESTIMATOR_LABELS, _SHORT)):
            r     = res[label]
            color = _COLORS[label]
            # Vertical CI bar
            ax.vlines(xs[i], r["ci_lower"], r["ci_upper"],
                      color=color, linewidth=2.5, zorder=3)
            # Horizontal end-caps
            ax.hlines([r["ci_lower"], r["ci_upper"]],
                      xs[i] - 0.14, xs[i] + 0.14,
                      color=color, linewidth=1.5, zorder=3)
            # Point estimate
            ax.scatter(xs[i], r["beta"],
                       color=color, s=55, zorder=5,
                       edgecolors="white", linewidths=0.8)

        ax.axhline(0, color="0.6", linewidth=0.8, linestyle=":", zorder=1)
        ax.set_xticks(xs)
        ax.set_xticklabels(_SHORT, fontsize=8.5, ha="center", va="top")
        ax.tick_params(axis="x", pad=4)
        ax.set_title(
            f"{slabel}\n"
            f"$N={meta['N']:,}$ counties, ${meta['N']*meta['T']:,}$ obs",
            fontsize=10, pad=8,
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(-0.65, len(ESTIMATOR_LABELS) - 0.35)
        ax.tick_params(axis="y", labelsize=9)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.3f"))

    axes[0].set_ylabel(
        r"$\hat{\beta}$ on Linter$\_$bra", fontsize=10
    )

    fig.suptitle(
        "Conley Spatial HAC vs State-Clustered Standard Errors\n"
        r"OLS: $\Delta\ln(\mathrm{loans}_{b,it}) = "
        r"\beta\,\mathrm{Linter\_bra}_{it}$ + county FE + year FE",
        fontsize=10, y=1.03,
    )

    legend_handles = [
        Line2D([0], [0], color=_COLORS[lb], linewidth=2.5,
               marker="o", markersize=6,
               label=lb.replace("\n", " "))
        for lb in ESTIMATOR_LABELS
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", bbox_to_anchor=(0.5, -0.10),
        ncol=2, fontsize=8.5, frameon=False,
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

    # -- Load shared inputs ---------------------------------------------------
    co_df        = pd.read_csv(COUNTY_PATH, dtype={"fips5": str})
    county_order = co_df["fips5"].str.zfill(5).tolist()

    W_geo_all, gal_order = load_w_geo(county_order)
    assert gal_order == county_order, "GAL order mismatch"

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

    samples = [("Full", panel), ("Contig", panel_contig), ("NonContig", panel_noncontig)]

    # -- Estimate + compute SEs for each sample -------------------------------
    all_results = {}
    for slabel, panel_sub in samples:
        print(f"\nBuilding {slabel} sample within arrays ...", flush=True)
        s = build_sample(panel_sub, county_order, W_geo_all, W_bank_all,
                         W_bank_knn3_all, W_bank_knn4_all, YEARS, year_pos)
        print(f"  N={s['N']} counties | T={s['T']} | NT={s['NT']:,} obs")
        print(f"  beta_hat = {s['beta']:.6f}  (two-way FWL)")
        print(f"  Computing spatial meats (T={s['T']} sparse mat-vecs per W) ...",
              flush=True)
        res = compute_all_ses(s)
        all_results[slabel] = res
        m = res["_meta"]

        # -- Two-way overlap diagnostic ----------------------------------------
        _ratio = (m["B_overlap"] / m["B_bank"]
                  if m["B_bank"] != 0 else float("nan"))
        print(f"  G={m['G']} states")
        print(f"  Two-way overlap diagnostic ({slabel}):")
        print(f"    B_state        = {m['B_state']:>12.6f}")
        print(f"    B_spatial bank = {m['B_bank']:>12.6f}   (all W_bank-linked pairs)")
        print(f"    B_overlap      = {m['B_overlap']:>12.6f}   (within-state AND W_bank-linked)")
        print(f"    B_overlap / B_spatial = {_ratio:.4f}"
              f"  [expect << 1 if cross-state links dominate]")
        print(f"    W_bank nonzero pairs : {m['nnz_bank']:,} total"
              f"  |  {m['nnz_bank_same']:,} same-state"
              f"  |  {m['frac_same_state_bank']*100:.2f}% same-state")
        print(f"    B_two = B_state + B_bank - B_overlap"
              f" = {m['B_two']:.6f}"
              f"  ({'> B_state -- two-way SE > cluster SE' if m['B_two'] > m['B_state'] else '<= B_state -- check overlap'})")
        print(f"  B_geo={m['B_geo']:.4f}  B_knn3={m['B_knn3']:.4f}  B_knn4={m['B_knn4']:.4f}")

    # -- Print formatted tables -----------------------------------------------
    W = 76
    csv_rows = []
    for slabel, res in all_results.items():
        m    = res["_meta"]
        N, T = m["N"], m["T"]
        print()
        print("=" * W)
        print(f"{slabel} sample  (N={N} counties, {N*T:,} obs, G={m['G']} states)")
        print("-" * W)
        print(f"{'Estimator':<36} {'beta':>8}  {'SE':>7}  "
              f"{'95% CI':>18}  {'CI width':>8}")
        print("-" * W)

        ref_width = res["State clustering (Favara-Imbs)"]["ci_width"]
        for label in ESTIMATOR_LABELS:
            r   = res[label]
            s_  = stars(r["p_value"])
            ci  = f"[{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]"
            print(f"  {label:<34} {r['beta']:8.4f}  "
                  f"{r['se']:7.4f}  {ci:>20}  {r['ci_width']:8.4f}")
            csv_rows.append(dict(
                sample        = slabel,
                estimator     = label,
                beta          = r["beta"],
                se            = r["se"],
                se_twoway_old = r.get("se_twoway_old", float("nan")),
                ci_lower      = r["ci_lower"],
                ci_upper      = r["ci_upper"],
                ci_width      = r["ci_width"],
                t_stat        = r["t_stat"],
                p_value       = r["p_value"],
                # Two-way meat components (same for all estimators in a sample)
                B_state       = m["B_state"],
                B_spatial     = m["B_bank"],
                B_overlap     = m["B_overlap"],
                frac_same_state_bank = m["frac_same_state_bank"],
            ))

        print("-" * W)
        print("CI width increase vs state clustering:")
        for label in ESTIMATOR_LABELS[1:]:
            pct = (res[label]["ci_width"] / ref_width - 1) * 100
            print(f"  {label:<36}  {pct:+.1f}%")
        print("=" * W)

    # -- Verify state SE against Favara-Imbs baseline -------------------------
    if "Full" in all_results:
        fi_se   = all_results["Full"]["State clustering (Favara-Imbs)"]["se"]
        fi_beta = all_results["Full"]["State clustering (Favara-Imbs)"]["beta"]
        print()
        print(f"State-cluster SE check (Full): beta={fi_beta:.4f}  SE={fi_se:.4f}")
    print(f"  F&I Table 2 baseline: beta~0.028  SE~0.010")
    print("  Spec includes the F&I controls and lagged DV; beta is column 0 of X.")

    # -- Save CSV -------------------------------------------------------------
    if output_dir is not None:
        cols = [
            "sample", "estimator", "beta", "se", "se_twoway_old",
            "ci_lower", "ci_upper", "ci_width", "t_stat", "p_value",
            "B_state", "B_spatial", "B_overlap", "frac_same_state_bank",
        ]
        pd.DataFrame(csv_rows)[cols].to_csv(
            output_dir / "conley_se_comparison.csv", index=False)
        print(f"\nSaved conley_se_comparison.csv  to {output_dir}")

    # -- Publication-quality plot ---------------------------------------------
    fig = _make_coef_plot(all_results)
    if output_dir is not None:
        fig.savefig(output_dir / "conley_se_comparison.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved conley_se_comparison.png  to {output_dir}")

    return all_results


if __name__ == "__main__":
    run(Path(__file__).parents[2] / "output")
