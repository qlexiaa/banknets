"""
sem_w1994.py -- Look-ahead robustness check: SEM with pre-deregulation W_bank_1994
====================================================================================
Motivation
----------
W_bank_avg is time-averaged over 1994-2005, the same period as the outcome and
treatment variables.  If deregulation itself reshapes the bank network (banks enter
new states, branching patterns shift), W_bank_avg may be partly endogenous to the
treatment.  W_bank_1994 is built from the 1994 FDIC cross-section only -- the year
of IBBEA enactment -- so it predates the post-deregulation reshaping of bank networks
and is exogenous to subsequent deregulation shocks.  Stability of the key lambda
estimate under W_bank_1994 relative to W_bank_avg supports the identification.

Construction
------------
Identical to pipeline/04_build_bank_weights.py `build_binary_year()` applied to the
1994 slice of data/fdic_deposits_1994_2005.csv:

  M[c,h] = 1  if BHC h has any branch in county c (1994)
  w_cc'  = (M @ M.T)[c,c'] / sqrt(diag[c] * diag[c'])   (cosine similarity)

Zero diagonal, row-standardised after sample subsetting.

The raw (pre-row-standardisation) cosine matrix is also saved temporarily to compute
network-persistence correlations against W_bank_avg.

Network-persistence statistics reported:
  corr_rs   : Pearson r between sample-row-standardised W_bank_1994 and
              W_bank_avg on the union of their nonzero off-diagonal entries
  corr_raw  : Pearson r between raw cosine similarity vectors on the union of
              their nonzero off-diagonal entries
  pct_shared: % of W_bank_avg nonzero entries also nonzero in W_bank_1994
              (extent to which the 1994 network spans the same county pairs as
               the time-averaged network)

Outputs
-------
  data/W_bank_1994.npz              -- cached row-standardised W_bank_1994
  output/sem_w1994_results.csv      -- SEM results (same schema as panel_fe_credit_results.csv)
                                       plus network-persistence columns appended
"""
import warnings
warnings.filterwarnings("ignore")

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse
import scipy.stats as stats
import spreg

sys.path.insert(0, str(Path(__file__).parents[1]))
import utils  # noqa: applies spreg Panel_FE_Error compatibility patch
from utils import row_standardize, sparse_to_pysal_w
from panel_data import CREDIT_CONTROLS, load_panel_with_credit, get_samples
from w_variants import load_w_geo, load_bank_variants

ROOT        = Path(__file__).parents[2]
COUNTY_PATH = ROOT / "data" / "county_order_Wgeo.csv"
FDIC_PATH   = ROOT / "data" / "fdic_deposits_1994_2005.csv"
CACHE_PATH  = ROOT / "data" / "W_bank_1994.npz"
CACHE_META_PATH = CACHE_PATH.with_suffix(".meta.json")
WAVG_PATH   = ROOT / "data" / "W_bank_avg.npz"
YEAR_1994   = 1994
X_VARS      = ["Linter_bra"] + CREDIT_CONTROLS


# ── Matrix construction ────────────────────────────────────────────────────────

def _load_fdic_filtered(county_order):
    """Load FDIC branch rows for counties in the spatial matrix order."""
    county_set = set(county_order)

    fdic = pd.read_csv(FDIC_PATH, dtype={"STCNTYBR": str})
    fdic["fips5"] = fdic["STCNTYBR"].str.zfill(5)
    return (fdic[fdic["fips5"].isin(county_set) &
                 fdic["RSSDHCR"].notna() &
                 (fdic["RSSDHCR"] != 0)]
            .copy())


def _build_raw_cosine(df_year, N, county_idx):
    """Build an unstandardised binary BHC cosine matrix for one FDIC year."""
    pairs = df_year[["fips5", "RSSDHCR"]].drop_duplicates().reset_index(drop=True)
    hcs = pairs["RSSDHCR"].unique()
    n_hcs = len(hcs)
    if n_hcs == 0 or pairs.empty:
        return np.zeros((N, N), dtype=np.float64), 0

    hc_map = {hc: i for i, hc in enumerate(hcs)}
    rows_idx = pairs["fips5"].map(county_idx).values
    cols_idx = pairs["RSSDHCR"].map(hc_map).values
    M = scipy.sparse.csr_matrix(
        (np.ones(len(pairs), dtype=np.float64), (rows_idx, cols_idx)),
        shape=(N, n_hcs),
    )

    B = (M @ M.T).toarray().astype(np.float64)
    d = B.diagonal().copy()
    denom = np.sqrt(np.outer(d, d))
    with np.errstate(divide="ignore", invalid="ignore"):
        W_raw = np.where(denom > 0, B / denom, 0.0)
    np.fill_diagonal(W_raw, 0.0)
    return W_raw, n_hcs


def _build_cosine_1994(county_order):
    """
    Build binary cosine-similarity matrix from 1994 FDIC branch data.

    Returns
    -------
    W_cos_raw : ndarray (N, N)
        Raw cosine similarities, zero diagonal, NOT row-standardised.
        Saved separately for persistence-correlation computation.
    W_cos_rs : ndarray (N, N)
        Row-standardised version used for SEM estimation.
    n_hcs : int
        Number of distinct BHCs active in 1994.
    """
    N          = len(county_order)
    county_idx = {fips: i for i, fips in enumerate(county_order)}

    print(f"  Loading FDIC data for {YEAR_1994} ...", flush=True)
    fdic = _load_fdic_filtered(county_order)
    fdic = fdic[fdic["year"] == YEAR_1994].copy()
    print(f"  1994 branch rows: {len(fdic):,}  |  panel counties: {N}", flush=True)

    W_cos_raw, n_hcs = _build_raw_cosine(fdic, N, county_idx)

    # Row-standardise
    rs = W_cos_raw.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        W_cos_rs = np.where(rs > 0, W_cos_raw / rs, 0.0)
    np.fill_diagonal(W_cos_rs, 0.0)

    nz = int((W_cos_rs > 0).sum())
    print(f"  W_bank_1994: n_hcs={n_hcs:,}  nonzero entries={nz:,}  "
          f"density={100.0*nz/(N*(N-1)):.4f}%", flush=True)

    return W_cos_raw, W_cos_rs, n_hcs


def _county_order_fingerprint(county_order):
    h = hashlib.sha256()
    for fips in county_order:
        h.update(str(fips).zfill(5).encode("ascii"))
        h.update(b"\0")
    return h.hexdigest()


def _load_cache_fingerprint():
    try:
        meta = json.loads(CACHE_META_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return meta.get("county_order_sha256")


def _save_cache_metadata(fingerprint, county_count):
    meta = {
        "county_order_sha256": fingerprint,
        "county_count": int(county_count),
    }
    CACHE_META_PATH.write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_avg_raw_cosine(county_order):
    """
    Rebuild the raw time-averaged bank cosine matrix for persistence diagnostics.

    data/W_bank_avg.npz stores the row-standardised matrix used for estimation,
    not the unstandardised cosine values. This helper reconstructs the raw
    annual cosine matrices and averages them without row-standardising.
    """
    N = len(county_order)
    county_idx = {fips: i for i, fips in enumerate(county_order)}

    print("  Rebuilding raw W_bank_avg cosine matrix for persistence stats ...",
          flush=True)
    fdic = _load_fdic_filtered(county_order)
    years = sorted(fdic["year"].dropna().unique())
    if not years:
        return np.zeros((N, N), dtype=np.float64)

    W_sum = np.zeros((N, N), dtype=np.float64)
    for yr in years:
        W_raw, _ = _build_raw_cosine(fdic[fdic["year"] == yr], N, county_idx)
        W_sum += W_raw

    W_avg_raw = W_sum / len(years)
    np.fill_diagonal(W_avg_raw, 0.0)
    return W_avg_raw


def load_w1994(county_order):
    """Load W_bank_1994 from cache; build and cache if not present.

    Returns (W_1994_raw_flat, W_1994_rs_sparse, n_hcs).
    W_1994_raw_flat is the flattened upper-triangle of the raw cosine matrix
    (needed for persistence correlation).
    """
    N = len(county_order)
    fingerprint = _county_order_fingerprint(county_order)

    if CACHE_PATH.exists() and _load_cache_fingerprint() == fingerprint:
        print(f"  Loading W_bank_1994 from cache: {CACHE_PATH.name}", flush=True)
        W_1994_rs_sp = scipy.sparse.load_npz(str(CACHE_PATH))
        # Raw cosine is not cached -- rebuild the raw array for correlation
        W_raw, _, n_hcs = _build_cosine_1994(county_order)
    else:
        if CACHE_PATH.exists():
            print("  W_bank_1994 cache county order mismatch; rebuilding", flush=True)
        W_raw, W_1994_rs, n_hcs = _build_cosine_1994(county_order)
        W_1994_rs_sp = scipy.sparse.csr_matrix(W_1994_rs)
        scipy.sparse.save_npz(str(CACHE_PATH), W_1994_rs_sp)
        _save_cache_metadata(fingerprint, N)
        print(f"  Cached W_bank_1994 -> {CACHE_PATH.name}", flush=True)

    return W_raw, W_1994_rs_sp, n_hcs


# ── Network-persistence statistics ────────────────────────────────────────────

def _pearson_or_nan(x, y):
    """Pearson r with constant-vector guard."""
    if len(x) < 2 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    corr, _ = stats.pearsonr(x, y)
    return corr


def load_wavg_for_persistence(county_order):
    """Load row-standardised W_bank_avg and rebuild its raw cosine analogue."""
    if not WAVG_PATH.exists():
        print(f"  [WARN] {WAVG_PATH} not found -- skipping persistence stats")
        return None, None
    W_avg_rs_sp = scipy.sparse.load_npz(str(WAVG_PATH))
    W_avg_raw = _build_avg_raw_cosine(county_order)
    return W_avg_raw, W_avg_rs_sp


def network_persistence(W_1994_raw, W_1994_rs_sp,
                        W_avg_raw, W_avg_rs_sp,
                        subset_idx=None, label=None):
    """
    Compute persistence statistics between W_bank_1994 and W_bank_avg.

    Parameters
    ----------
    W_1994_raw : ndarray (N, N)
        Raw (un-standardised) cosine similarities for 1994.
    W_1994_rs_sp : scipy.sparse CSR
        Row-standardised W_bank_1994.
    W_avg_raw : ndarray (N, N)
        Raw time-averaged cosine similarities.
    W_avg_rs_sp : scipy.sparse CSR
        Row-standardised W_bank_avg.
    subset_idx : ndarray or None
        Optional county indices used by a sample-specific SEM estimation.
        When supplied, row-standardised correlations are recomputed after
        subsetting, matching the SEM matrix construction.

    Returns
    -------
    dict with keys:
        corr_rs     : Pearson r on row-standardised matrices, union-of-nonzero
        corr_raw    : Pearson r on raw cosine matrices, union-of-nonzero
        pct_shared  : % of W_avg nonzero entries also nonzero in W_1994
        n_pairs_avg : number of nonzero off-diagonal pairs in W_avg
        n_pairs_1994: number of nonzero off-diagonal pairs in W_1994
        n_pairs_union: number of nonzero pairs in union
    """
    if W_avg_raw is None or W_avg_rs_sp is None:
        return {}

    if subset_idx is None:
        W_avg = row_standardize(W_avg_rs_sp).toarray().astype(np.float64)
        W_1994 = row_standardize(W_1994_rs_sp).toarray().astype(np.float64)
        W_avg_raw_sub = W_avg_raw.copy()
        W_1994_raw_sub = W_1994_raw.copy()
    else:
        idx = np.asarray(subset_idx, dtype=int)
        W_avg = row_standardize(W_avg_rs_sp[idx, :][:, idx]).toarray().astype(np.float64)
        W_1994 = row_standardize(W_1994_rs_sp[idx, :][:, idx]).toarray().astype(np.float64)
        W_avg_raw_sub = W_avg_raw[np.ix_(idx, idx)].copy()
        W_1994_raw_sub = W_1994_raw[np.ix_(idx, idx)].copy()

    np.fill_diagonal(W_avg, 0.0)
    np.fill_diagonal(W_1994, 0.0)
    np.fill_diagonal(W_avg_raw_sub, 0.0)
    np.fill_diagonal(W_1994_raw_sub, 0.0)

    # Masks
    mask_avg  = W_avg  > 0
    mask_1994 = W_1994 > 0
    mask_union = mask_avg | mask_1994

    n_avg   = int(mask_avg.sum())
    n_1994  = int(mask_1994.sum())
    n_union = int(mask_union.sum())
    n_shared = int((mask_avg & mask_1994).sum())

    pct_shared = 100.0 * n_shared / n_avg if n_avg > 0 else np.nan

    # Pearson r on row-standardised matrices (union of nonzero)
    corr_rs = _pearson_or_nan(W_1994[mask_union], W_avg[mask_union])

    # Pearson r on raw cosine matrices (union of raw nonzero entries)
    mask_raw_union = (W_avg_raw_sub > 0) | (W_1994_raw_sub > 0)
    corr_raw = _pearson_or_nan(
        W_1994_raw_sub[mask_raw_union],
        W_avg_raw_sub[mask_raw_union],
    )

    scope = f" ({label})" if label else ""
    print(f"  Network persistence stats{scope}:")
    print(f"    W_avg  nonzero pairs : {n_avg:,}")
    print(f"    W_1994 nonzero pairs : {n_1994:,}")
    print(f"    Shared pairs         : {n_shared:,}  ({pct_shared:.1f}% of W_avg)")
    print(f"    corr(W_1994_rs, W_avg_rs) on union  : {corr_rs:.4f}")
    print(f"    corr(W_1994_raw, W_avg_raw) on union: {corr_raw:.4f}")

    return dict(
        corr_rs       = float(corr_rs),
        corr_raw      = float(corr_raw),
        pct_shared    = float(pct_shared),
        n_pairs_avg   = n_avg,
        n_pairs_1994  = n_1994,
        n_pairs_union = n_union,
    )


# ── SEM estimation (mirrors sem_credit._build_arrays_variant) ─────────────────

def _usable_counties(panel_sub, county_order, W_all_sp, dv="Dl_nloans_b"):
    """Counties with complete DV/control data and nonzero full-sample W rows."""
    any_nan = panel_sub.groupby("fips5")[[dv] + X_VARS].apply(
        lambda g: g.isna().any().any()
    )
    sub_co = set(panel_sub["fips5"].unique())

    rs_full = np.array(W_all_sp.sum(axis=1)).flatten()
    islands = {county_order[i] for i, r in enumerate(rs_full) if r == 0}

    return [c for c in county_order
            if c in sub_co and not any_nan.get(c, True) and c not in islands]


def _build_arrays(panel_sub, county_order, W_all_sp, YEARS, year_pos, sample_label):
    """Build balanced long-format arrays; subset and row-standardise W to sample."""
    T   = len(YEARS)
    DV  = "Dl_nloans_b"

    usable = _usable_counties(panel_sub, county_order, W_all_sp, dv=DV)
    N     = len(usable)
    u_pos = {c: i for i, c in enumerate(usable)}

    df = (panel_sub[panel_sub["fips5"].isin(set(usable))]
          .assign(t_idx=lambda d: d["year"].map(year_pos),
                  c_idx=lambda d: d["fips5"].map(u_pos))
          .sort_values(["t_idx", "c_idx"]))

    y_long    = df[DV].values.reshape(-1, 1)
    t_idx_vec = df["t_idx"].values
    year_dummies = np.column_stack([
        (t_idx_vec == year_pos[yr]).astype(np.float64) for yr in YEARS[1:]])
    x_long = np.hstack([df[X_VARS].values.astype(np.float64), year_dummies])

    assert not np.isnan(y_long).any(), f"NaN in y ({sample_label})"
    assert not np.isnan(x_long).any(), f"NaN in X ({sample_label})"
    assert y_long.shape == (N * T, 1)
    assert x_long.shape == (N * T, len(X_VARS) + len(YEARS) - 1), \
        f"x shape {x_long.shape} (expected ({N*T}, {len(X_VARS) + len(YEARS) - 1}))"

    idx   = np.array([county_order.index(c) for c in usable])
    W_sub = row_standardize(W_all_sp[idx, :][:, idx])
    return y_long, x_long, sparse_to_pysal_w(W_sub), N


def _extract(res, N, T):
    beta   = float(res.betas[0, 0])
    lam    = float(res.lam)
    se_b   = float(res.std_err[0])
    se_l   = float(res.std_err[-1])
    z_b, p_b = res.z_stat[0]
    z_l, p_l = res.z_stat[-1]
    return dict(beta=beta, se_beta=se_b, z_beta=float(z_b), p_beta=float(p_b),
                lam=lam,   se_lam=se_l,  z_lam=float(z_l),  p_lam=float(p_l),
                n_co=N,    n_obs=N * T)


# ── Lambda gap test ────────────────────────────────────────────────────────────

def _gap_test(r_geo, r_1994):
    """Descriptive lambda gap; formal p-values need paired inference."""
    if r_geo is None or r_1994 is None:
        return {k: np.nan for k in ("gap_lam", "se_gap", "z_gap", "p_gap_onesided")}
    gap    = r_1994["lam"] - r_geo["lam"]
    return dict(gap_lam=gap, se_gap=np.nan, z_gap=np.nan, p_gap_onesided=np.nan)


# ── Public entry point ─────────────────────────────────────────────────────────

def run(output_dir=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    co_df        = pd.read_csv(COUNTY_PATH, dtype={"fips5": str})
    county_order = co_df["fips5"].str.zfill(5).tolist()

    W_geo_all, gal_order = load_w_geo(county_order)
    assert gal_order == county_order, "GAL order mismatch"

    bank_vars       = load_bank_variants(county_order, W_geo_all=W_geo_all)
    W_knn3_all      = bank_vars["W_bank_knn3"]
    W_knn4_all      = bank_vars["W_bank_knn4"]

    panel = load_panel_with_credit()
    panel["fips5"] = panel["fips5"].astype(str).str.zfill(5)
    YEARS    = sorted(panel["year"].unique())
    T        = len(YEARS)
    year_pos = {yr: i for i, yr in enumerate(YEARS)}

    # ── Build / load W_bank_1994 ──────────────────────────────────────────────
    print("\nBuilding / loading W_bank_1994 ...")
    W_1994_raw, W_1994_rs_sp, n_hcs_1994 = load_w1994(county_order)
    samples = get_samples(panel)

    # ── Network persistence ───────────────────────────────────────────────────
    print("\nComputing network persistence statistics ...")
    W_avg_raw, W_avg_rs_sp = load_wavg_for_persistence(county_order)
    network_persistence(
        W_1994_raw, W_1994_rs_sp, W_avg_raw, W_avg_rs_sp,
        label="full county order",
    )
    persist_by_sample = {}
    county_idx = {c: i for i, c in enumerate(county_order)}
    for sample_label, panel_sub in samples:
        usable = _usable_counties(panel_sub, county_order, W_1994_rs_sp)
        idx = np.array([county_idx[c] for c in usable], dtype=int)
        persist_by_sample[sample_label] = network_persistence(
            W_1994_raw, W_1994_rs_sp, W_avg_raw, W_avg_rs_sp,
            subset_idx=idx,
            label=f"{sample_label} W_bank_1994 SEM sample",
        )

    # ── SEM estimation ────────────────────────────────────────────────────────
    results  = {}

    W_SPECS = [
        ("W_geo",        W_geo_all),
        ("W_bank_1994",  W_1994_rs_sp),
        ("W_bank_knn3",  W_knn3_all),
        ("W_bank_knn4",  W_knn4_all),
    ]

    W_COL = 86
    print()
    print("=" * W_COL)
    print("Panel_FE_Error -- DV: Dl_nloans_b | W_bank_1994 robustness check")
    print("Regressors: " + ", ".join(X_VARS) + " | Two-way FE (ML)")
    print("=" * W_COL)
    print(f"{'Sample':<12} {'W matrix':<18} {'N':>5}  "
          f"{'beta':>8} {'SE':>6}  "
          f"{'lambda':>8} {'SE':>6}  {'p_lam':>8}")
    print("-" * W_COL)

    for sample_label, panel_sub in samples:
        for w_name, W_all in W_SPECS:
            tag = f"{sample_label} x {w_name}"
            print(f"  Estimating {tag} ...", flush=True)
            try:
                y, x, w_pysal, N = _build_arrays(
                    panel_sub, county_order, W_all, YEARS, year_pos, sample_label)
                nx = X_VARS + [f"yr{yr}" for yr in YEARS[1:]]
                res = spreg.Panel_FE_Error(
                    y, x, w_pysal,
                    name_y="Dl_nloans_b", name_x=nx,
                    name_w=w_name, name_ds=sample_label,
                )
                r = _extract(res, N, T)
                results[(sample_label, w_name)] = r
                def _st(p):
                    if np.isnan(p): return ""
                    return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
                print(f"  {sample_label:<12} {w_name:<18} {r['n_co']:>5}  "
                      f"{r['beta']:>8.4f}{_st(r['p_beta'])} {r['se_beta']:>6.4f}  "
                      f"{r['lam']:>8.4f}{_st(r['p_lam'])} {r['se_lam']:>6.4f}  "
                      f"{r['p_lam']:>8.4f}")
            except Exception as exc:
                print(f"  [SKIP] {tag}: {exc}")
                results[(sample_label, w_name)] = None

        print()

    # ── Lambda gap tests ──────────────────────────────────────────────────────
    print()
    for cmp_w in ("W_bank_1994", "W_bank_knn3", "W_bank_knn4"):
        print(f"Lambda gap ({cmp_w} vs W_geo) -- descriptive only; paired SE/p-value not reported:")
        print(f"{'Sample':<12} {'gap':>8}")
        print("-" * 24)

        for sample_label, _ in samples:
            r_geo = results.get((sample_label, "W_geo"))
            r_cmp = results.get((sample_label, cmp_w))
            g = _gap_test(r_geo, r_cmp)
            print(f"  {sample_label:<12} {g['gap_lam']:>8.4f}")
        print()

    # ── Print persistence ─────────────────────────────────────────────────────
    if persist_by_sample:
        print()
        print("Network persistence by W_bank_1994 estimation sample:")
        print(f"  BHCs active in 1994     : {n_hcs_1994:,}")
        print(f"  {'Sample':<12} {'shared%':>9} {'corr_rs':>9} {'corr_raw':>9}")
        print(f"  {'-' * 42}")
        for sample_label, _ in samples:
            p = persist_by_sample.get(sample_label, {})
            print(f"  {sample_label:<12} "
                  f"{p.get('pct_shared', float('nan')):>9.1f} "
                  f"{p.get('corr_rs', float('nan')):>9.4f} "
                  f"{p.get('corr_raw', float('nan')):>9.4f}")

    print("=" * W_COL)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if output_dir is not None:
        rows = []
        for sample_label, _ in samples:
            for w_name, _ in W_SPECS:
                r = results.get((sample_label, w_name))
                if r is None:
                    continue
                # Gap test: compare this W vs W_geo (only for non-geo rows)
                if w_name in ("W_bank_1994", "W_bank_knn3", "W_bank_knn4"):
                    g = _gap_test(
                        results.get((sample_label, "W_geo")),
                        results.get((sample_label, w_name))
                    )
                else:
                    g = {k: np.nan for k in
                         ("gap_lam", "se_gap", "z_gap", "p_gap_onesided")}
                # Network-persistence stats apply only to W_bank_1994
                p = (persist_by_sample.get(sample_label, {})
                     if w_name == "W_bank_1994" else {})
                row = dict(
                    model    = f"FE {w_name} ({sample_label.lower()})",
                    sample   = sample_label,
                    w_matrix = w_name,
                    **{k: r[k] for k in
                       ["beta", "se_beta", "z_beta", "p_beta",
                        "lam",  "se_lam",  "z_lam",  "p_lam",
                        "n_co", "n_obs"]},
                    **{k: g[k] for k in
                       ["gap_lam", "se_gap", "z_gap", "p_gap_onesided"]},
                    # Network persistence applies only to W_bank_1994 rows.
                    n_hcs_1994        = n_hcs_1994 if w_name == "W_bank_1994" else np.nan,
                    corr_1994_avg_rs  = p.get("corr_rs",      np.nan),
                    corr_1994_avg_raw = p.get("corr_raw",     np.nan),
                    pct_pairs_shared  = p.get("pct_shared",   np.nan),
                    n_pairs_avg       = p.get("n_pairs_avg",  np.nan),
                    n_pairs_1994      = p.get("n_pairs_1994", np.nan),
                )
                rows.append(row)

        out_path = output_dir / "sem_w1994_results.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"\nSaved sem_w1994_results.csv to {output_dir}")

    return results, persist_by_sample


if __name__ == "__main__":
    run(ROOT / "output")
