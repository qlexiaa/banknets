"""
Shared utilities for spatial weight matrix construction and estimation.

Importing this module also applies a compatibility patch for spreg
Panel_FE_Error: in some numpy versions sig2 is a 1-element array and
the internal summary formatter raises TypeError on float(sig2).  The
patch wraps that formatter so the error is silently ignored — all
computational attributes (betas, lam, std_err, z_stat) are unaffected.
"""
import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix
from libpysal.weights import WSP

# ── spreg numpy >= 2.0 compatibility patch ────────────────────────────────────
# Several spreg formatters call float(reg.sig2) or use sig2 directly in %
# format strings.  In numpy >= 2.0 sig2 is stored as a 1-element array and
# these calls raise TypeError.  All computational attributes (betas, lam/rho,
# std_err, z_stat) are written before the formatters run, so swallowing the
# errors loses nothing.
#
# Two locations need patching:
#   1. spreg.summary_output.{Panel_FE_Error, Panel_FE_Lag} — called as the
#      very last line of each panel model's __init__
#   2. spreg.output._nonspat_top — called inside ML_Error / ML_Lag __init__
#      before the final output() call; returns a string (self.other_top)
try:
    import sys as _sys
    import spreg  # ensure submodules are registered in sys.modules

    # Patch 1: summary_output formatters
    import spreg.summary_output as _so
    for _fn_name in ("Panel_FE_Error", "Panel_FE_Lag", "ML_Error", "ML_Lag"):
        if hasattr(_so, _fn_name):
            def _make_safe(orig):
                def _safe(*args, **kwargs):
                    try:
                        orig(*args, **kwargs)
                    except TypeError:
                        pass
                return _safe
            setattr(_so, _fn_name, _make_safe(getattr(_so, _fn_name)))

    # Patch 2: spreg.output._nonspat_top (used by ML_Error, ML_Lag)
    # ml_error.py and others do `from .output import _nonspat_top`, so each
    # submodule holds its own reference.  Must patch every one of them.
    import spreg.ml_error, spreg.ml_lag  # ensure they are loaded

    _out_mod = _sys.modules.get('spreg.output')
    if _out_mod is not None and hasattr(_out_mod, '_nonspat_top'):
        _orig_top = _out_mod._nonspat_top

        def _safe_nonspat_top(reg, *args, **kwargs):
            # Coerce 1-element array sig2 to scalar before %-formatting
            for _attr in ('sig2', 'sig2ML', 'sig2n', 'sig2n_k'):
                if hasattr(reg, _attr):
                    try:
                        setattr(reg, _attr,
                                float(np.asarray(getattr(reg, _attr)).flat[0]))
                    except Exception:
                        pass
            try:
                return _orig_top(reg, *args, **kwargs)
            except TypeError:
                return ''

        # Patch the source module and every submodule that imported it
        for _mod_name, _mod in list(_sys.modules.items()):
            if _mod_name.startswith('spreg') and hasattr(_mod, '_nonspat_top'):
                setattr(_mod, '_nonspat_top', _safe_nonspat_top)

except Exception:
    pass  # spreg not installed or already patched


def row_standardize(W):
    """Row-standardise a scipy sparse matrix (zero rows left as zero)."""
    rs = np.array(W.sum(axis=1)).flatten()
    rs[rs == 0] = 1.0
    return (scipy.sparse.diags(1.0 / rs) @ W).tocsr()


def parse_gal(path):
    """
    Parse a PySAL GAL file.
    Returns (ordered_fips5_list, {fips5: [neighbour_fips5, ...]}).
    """
    with open(path) as f:
        lines = f.readlines()
    order, adj = [], {}
    i = 1
    while i < len(lines):
        parts = lines[i].strip().split()
        cid  = parts[0].zfill(5)
        nnbr = int(parts[1])
        nbrs = [nb.zfill(5) for nb in (lines[i + 1].strip().split() if nnbr > 0 else [])]
        order.append(cid)
        adj[cid] = nbrs
        i += 2
    return order, adj


def adj_to_sparse(order, adj):
    """Convert a neighbour-adjacency dict to a binary sparse CSR matrix."""
    idx = {c: i for i, c in enumerate(order)}
    N   = len(order)
    rows, cols = [], []
    for c, nbrs in adj.items():
        for nb in nbrs:
            if nb in idx:
                rows.append(idx[c])
                cols.append(idx[nb])
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))


def gal_to_W(path, county_order=None):
    """
    Parse a GAL file and return a row-standardised sparse CSR matrix.

    Parameters
    ----------
    path          : path to .gal file
    county_order  : if supplied, asserts the GAL order matches exactly

    Returns
    -------
    W_sparse      : row-standardised CSR matrix
    order         : list of fips5 strings in GAL row order
    """
    order, adj = parse_gal(path)
    if county_order is not None:
        assert order == list(county_order), \
            "GAL county order does not match supplied county_order"
    return row_standardize(adj_to_sparse(order, adj)), order


def sparse_to_pysal_w(W_sparse):
    """Convert a row-standardised sparse CSR to a libpysal W object."""
    return WSP(W_sparse.tocsr()).to_W()
