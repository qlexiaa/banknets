"""
w_variants.py -- Spatial weight matrix catalogue for bank-network analysis.

Centralises construction and caching of all W matrices used in the pipeline.
All returned matrices are row-standardised scipy CSR arrays.

Variants
--------
  W_bank          : time-averaged cosine similarity of BHC branch presence vectors
  W_bank_count    : count-weighted version of W_bank
  W_bank_binary   : nonzero W_bank entries mapped to 1 (binary connectivity)
  W_bank_knn3     : top-3 nearest bank-network neighbours (from W_bank)
  W_bank_knn4     : top-4 nearest bank-network neighbours (from W_bank)
  W_bank_count_knn3  : W_bank_knn3 neighbor set, re-weighted by branch count
  W_bank_count_knn4  : W_bank_knn4 neighbor set, re-weighted by branch count
  W_bank_binary_knn3 : W_bank_knn3 neighbor set, binary (0/1) weights
  W_bank_binary_knn4 : W_bank_knn4 neighbor set, binary (0/1) weights
  W_bank_nonGeo   : W_bank with geographic-contiguity neighbours zeroed out
  W_bank_interstate : W_bank restricted to cross-state (i,j) pairs only
  W_bank_intrastate : W_bank restricted to within-state (i,j) pairs only

W_geo (queen contiguity) is returned by load_w_geo() and is used as the
geographic benchmark throughout.

Cache files live in data/; matrices are built once and reused from .npz files
on subsequent calls.
"""

import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import scipy.sparse

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from utils import gal_to_W, row_standardize  # noqa: E402 (sys.path set above)

ROOT        = _HERE.parent
DATA        = ROOT / "data"
GAL_PATH    = DATA / "W_geo_queen.gal"
COUNTY_PATH = DATA / "county_order_Wgeo.csv"

PATHS = {
    "W_bank"          : DATA / "W_bank_avg.npz",
    "W_bank_count"    : DATA / "W_bank_count_avg.npz",
    "W_bank_binary"   : DATA / "W_bank_binary.npz",
    "W_bank_knn3"     : DATA / "W_bank_knn3.npz",
    "W_bank_knn4"     : DATA / "W_bank_knn4.npz",
    "W_bank_count_knn3"  : DATA / "W_bank_count_knn3.npz",
    "W_bank_count_knn4"  : DATA / "W_bank_count_knn4.npz",
    "W_bank_binary_knn3" : DATA / "W_bank_binary_knn3.npz",
    "W_bank_binary_knn4" : DATA / "W_bank_binary_knn4.npz",
    "W_bank_nonGeo"   : DATA / "W_bank_nonGeo.npz",
    "W_bank_interstate"  : DATA / "W_bank_interstate.npz",
    "W_bank_intrastate"  : DATA / "W_bank_intrastate.npz",
}

ALL_VARIANTS = [
    "W_bank", "W_bank_count", "W_bank_binary", "W_bank_knn3", "W_bank_knn4",
    "W_bank_count_knn3", "W_bank_count_knn4",
    "W_bank_binary_knn3", "W_bank_binary_knn4",
    "W_bank_nonGeo", "W_bank_interstate", "W_bank_intrastate",
]
VARIANT_ORDER = ALL_VARIANTS


# ── Builder functions ──────────────────────────────────────────────────────────

def _build_binary(W_bank_sp):
    W = W_bank_sp.toarray().astype(np.float64)
    np.fill_diagonal(W, 0.0)
    W_bin = (W > 0).astype(np.float64)
    np.fill_diagonal(W_bin, 0.0)
    return scipy.sparse.csr_matrix(W_bin)


def _build_knn(W_bank_sp, k=4):
    """Keep top-k weights per row, zero the rest."""
    W = W_bank_sp.toarray().astype(np.float64)
    np.fill_diagonal(W, 0.0)
    N = W.shape[0]
    W_knn = np.zeros_like(W)
    for i in range(N):
        row = W[i]
        nz  = np.count_nonzero(row)
        if nz == 0:
            continue
        if nz <= k:
            W_knn[i] = row
        else:
            top_k = np.argpartition(row, -k)[-k:]
            W_knn[i, top_k] = row[top_k]
    np.fill_diagonal(W_knn, 0.0)
    return scipy.sparse.csr_matrix(W_knn)


def _build_reweighted_from_knn(W_knn_sp, W_full_weighted_sp):
    """
    Take the neighbor topology already selected by cosine-ranked
    KNN truncation (W_knn_sp: W_bank_knn3 or W_bank_knn4), and
    re-weight those same fixed positions using an alternative
    weighting scheme (W_full_weighted_sp: the full, untruncated
    binary or count matrix). Row-standardizes the result.

    This ensures W_bank_binary_knn{k} and W_bank_count_knn{k} share
    the EXACT SAME neighbor set as W_bank_knn{k} (cosine), differing
    only in the weight assigned to each of the k selected links --
    rather than independently re-selecting k neighbors from a
    differently-ranked (or, for binary, tie-degenerate) matrix.
    """
    mask = (W_knn_sp.toarray() > 0)
    W_full = W_full_weighted_sp.toarray()
    W_reweighted = W_full * mask
    np.fill_diagonal(W_reweighted, 0.0)
    return row_standardize(scipy.sparse.csr_matrix(W_reweighted))


def _build_nongeo(W_bank_sp, W_geo_sp):
    """Zero out entries shared with W_geo (Favara & Imbs 2015 non-geographic links)."""
    W_ng  = W_bank_sp.toarray().astype(np.float64)
    W_geo = W_geo_sp.toarray().astype(np.float64)
    np.fill_diagonal(W_ng, 0.0)
    np.fill_diagonal(W_geo, 0.0)
    W_ng[W_geo > 0] = 0.0
    np.fill_diagonal(W_ng, 0.0)
    return scipy.sparse.csr_matrix(W_ng)


def _build_interstate(W_bank_sp, county_order):
    """Keep only cross-state links (different first-two FIPS digits)."""
    states     = np.array([c[:2] for c in county_order])
    W          = W_bank_sp.toarray().astype(np.float64)
    np.fill_diagonal(W, 0.0)
    same_state = states[:, None] == states[None, :]
    np.fill_diagonal(same_state, False)
    W[same_state] = 0.0
    np.fill_diagonal(W, 0.0)
    return scipy.sparse.csr_matrix(W)


def _build_intrastate(W_bank_sp, county_order):
    """Keep only within-state links (identical first-two FIPS digits)."""
    states     = np.array([c[:2] for c in county_order])
    W          = W_bank_sp.toarray().astype(np.float64)
    np.fill_diagonal(W, 0.0)
    diff_state = states[:, None] != states[None, :]
    np.fill_diagonal(diff_state, False)
    W[diff_state] = 0.0
    np.fill_diagonal(W, 0.0)
    return scipy.sparse.csr_matrix(W)


# ── Cache helper ───────────────────────────────────────────────────────────────

def _load_or_build(name, build_fn):
    path = PATHS[name]
    if path.exists():
        return scipy.sparse.load_npz(path)
    print(f"  Building {name} ...", flush=True)
    W = build_fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    scipy.sparse.save_npz(str(path), W)
    print(f"  Cached {name} -> {path}")
    return W


# ── Public API ─────────────────────────────────────────────────────────────────

def load_w_geo(county_order):
    """Return (W_geo_sparse_rs, gal_county_order) from the queen-contiguity GAL file."""
    import pandas as pd
    if county_order is None:
        co_df        = pd.read_csv(COUNTY_PATH, dtype={"fips5": str})
        county_order = co_df["fips5"].str.zfill(5).tolist()
    W_geo, order = gal_to_W(GAL_PATH, county_order)
    return row_standardize(W_geo), order


def load_bank_variants(county_order, W_geo_all=None):
    """Return OrderedDict[str, csr_matrix] of row-standardised bank-network variants.

    Parameters
    ----------
    county_order : list[str]
        Ordered FIPS5 codes matching the row/column index of all matrices.
    W_geo_all : scipy.sparse.csr_matrix or None
        Row-standardised W_geo for the full county_order.  If None it is loaded
        from the GAL file automatically (needed only for W_bank_nonGeo).

    Returns
    -------
    OrderedDict mapping variant name -> row-standardised scipy CSR matrix.
    """
    # Load base matrices
    W_bank_raw  = scipy.sparse.load_npz(PATHS["W_bank"])
    W_count_raw = scipy.sparse.load_npz(PATHS["W_bank_count"])

    if W_geo_all is None:
        W_geo_all, _ = load_w_geo(county_order)

    variants = OrderedDict()

    # 1. W_bank (cosine similarity, time-averaged)
    variants["W_bank"] = row_standardize(W_bank_raw)

    # 2. W_bank_count (count-weighted)
    variants["W_bank_count"] = row_standardize(W_count_raw)

    # 3. W_bank_binary (nonzero → 1)
    W_bin_raw = _load_or_build("W_bank_binary", lambda: _build_binary(W_bank_raw))
    variants["W_bank_binary"] = row_standardize(W_bin_raw)

    # 4. W_bank_knn3 (top-3 neighbours from W_bank)
    W_knn3_raw = _load_or_build("W_bank_knn3", lambda: _build_knn(W_bank_raw, k=3))
    variants["W_bank_knn3"] = row_standardize(W_knn3_raw)

    # 5. W_bank_knn4 (top-4 neighbours from W_bank)
    W_knn4_raw = _load_or_build("W_bank_knn4", lambda: _build_knn(W_bank_raw, k=4))
    variants["W_bank_knn4"] = row_standardize(W_knn4_raw)

    # 6. W_bank_count_knn3/knn4 (W_bank_knn{k} neighbor set, re-weighted by
    #    branch count -- same neighbors as the cosine KNN, different weights)
    W_count_knn3_raw = _load_or_build(
        "W_bank_count_knn3",
        lambda: _build_reweighted_from_knn(W_knn3_raw, W_count_raw),
    )
    variants["W_bank_count_knn3"] = row_standardize(W_count_knn3_raw)

    W_count_knn4_raw = _load_or_build(
        "W_bank_count_knn4",
        lambda: _build_reweighted_from_knn(W_knn4_raw, W_count_raw),
    )
    variants["W_bank_count_knn4"] = row_standardize(W_count_knn4_raw)

    # 7. W_bank_binary_knn3/knn4 (W_bank_knn{k} neighbor set, binary weights --
    #    same neighbors as the cosine KNN, just 0/1 instead of continuous)
    W_bin_knn3_raw = _load_or_build(
        "W_bank_binary_knn3",
        lambda: _build_reweighted_from_knn(W_knn3_raw, W_bin_raw),
    )
    variants["W_bank_binary_knn3"] = row_standardize(W_bin_knn3_raw)

    W_bin_knn4_raw = _load_or_build(
        "W_bank_binary_knn4",
        lambda: _build_reweighted_from_knn(W_knn4_raw, W_bin_raw),
    )
    variants["W_bank_binary_knn4"] = row_standardize(W_bin_knn4_raw)

    # 8. W_bank_nonGeo (geographic neighbours zeroed out)
    W_ng_raw = _load_or_build(
        "W_bank_nonGeo",
        lambda: _build_nongeo(W_bank_raw, W_geo_all),
    )
    variants["W_bank_nonGeo"] = row_standardize(W_ng_raw)

    # 9. W_bank_interstate (cross-state links only)
    W_is_raw = _load_or_build(
        "W_bank_interstate",
        lambda: _build_interstate(W_bank_raw, county_order),
    )
    variants["W_bank_interstate"] = row_standardize(W_is_raw)

    # 10. W_bank_intrastate (within-state links only)
    W_ina_raw = _load_or_build(
        "W_bank_intrastate",
        lambda: _build_intrastate(W_bank_raw, county_order),
    )
    variants["W_bank_intrastate"] = row_standardize(W_ina_raw)

    return variants


def variant_stats(variants):
    """Print sparsity summary for each variant."""
    print(f"\n{'Variant':<22}  {'N_links':>10}  {'Density':>10}  {'Avg nbrs':>10}")
    print("-" * 58)
    for name, W in variants.items():
        W_arr = W.toarray()
        np.fill_diagonal(W_arr, 0.0)
        nz  = int(np.count_nonzero(W_arr))
        N   = W_arr.shape[0]
        den = nz / (N * (N - 1))
        print(f"  {name:<20}  {nz:>10,}  {den:>10.6f}  {nz/N:>10.2f}")


if __name__ == "__main__":
    import pandas as pd
    co_df        = pd.read_csv(COUNTY_PATH, dtype={"fips5": str})
    county_order = co_df["fips5"].str.zfill(5).tolist()
    print(f"Loading/building {len(VARIANT_ORDER)} bank-network W variants ...")
    W_geo, _ = load_w_geo(county_order)
    variants  = load_bank_variants(county_order, W_geo_all=W_geo)
    variant_stats(variants)
    print("\nDone.")
