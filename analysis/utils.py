"""
Shared utilities for spatial weight matrix construction and estimation.

Imported by: build_W_bank.py, build_W_bank_count.py,
             run_sem_rq1.py, run_panel_fe_error.py
"""
import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix
from libpysal.weights import WSP


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
