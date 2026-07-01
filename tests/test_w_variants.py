"""
Tests for analysis/w_variants.py

Covers the pure helper functions relevant to the KNN-3/KNN-4 truncated
bank-network weight matrix bug fix:
  - _build_knn                   (cosine-ranked topology selection)
  - _build_binary                (all-ones at nonzero W_bank positions)
  - _build_reweighted_from_knn   (re-weights a fixed cosine-selected topology)

Background
----------
W_bank_binary_knn3/4 and W_bank_count_knn3/4 used to be built by
independently top-k-truncating the full binary / count matrices with
np.argpartition. For the binary case this meant picking an ARBITRARY k of
typically hundreds of *tied* (all == 1.0) entries per row -- not a
meaningful "k strongest links" in any sense. For the count case, "top k"
was well defined but ranked by raw branch-count scale rather than
similarity, systematically favoring large banking markets as neighbors for
nearly every county (hub-and-spoke topology).

The fix reuses the neighbor *topology* already selected by cosine-ranked
KNN truncation (W_bank_knn3/4) and only swaps the *weight* assigned to
those fixed positions -- see _build_reweighted_from_knn.
"""
import sys
import os
import importlib

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))

from conftest import FakeSparse

_module = None


def _get_module():
    global _module
    if _module is None:
        # conftest pre-installs a MagicMock under "w_variants" for modules
        # that merely import it as a dependency. Remove that mock so this
        # file exercises the real w_variants source.
        sys.modules.pop("w_variants", None)
        _module = importlib.import_module("w_variants")
    return _module


def _make_sparse(arr):
    return FakeSparse(np.array(arr, dtype=float))


# ---------------------------------------------------------------------------
# Shared fixtures: a small synthetic county network with an unambiguous
# cosine top-3 ranking, plus a count matrix sharing its support.
# ---------------------------------------------------------------------------

def _cosine_matrix():
    """5x5 matrix; row 0's top-3 by cosine weight are cols 1, 2, 3."""
    return np.array([
        [0.0, 0.9, 0.8, 0.7, 0.1],
        [0.9, 0.0, 0.5, 0.4, 0.3],
        [0.8, 0.5, 0.0, 0.6, 0.2],
        [0.7, 0.4, 0.6, 0.0, 0.3],
        [0.1, 0.3, 0.2, 0.3, 0.0],
    ])


def _count_matrix():
    """Same support as _cosine_matrix, but non-proportional weights."""
    return np.array([
        [0, 50, 5, 2, 40],
        [50, 0, 3, 9, 1],
        [5, 3, 0, 7, 2],
        [2, 9, 7, 0, 6],
        [40, 1, 2, 6, 0],
    ], dtype=float)


# ---------------------------------------------------------------------------
# W_bank_binary_knn3 must select the SAME neighbors as W_bank_knn3 (cosine)
# ---------------------------------------------------------------------------

class TestBinaryKnnSharesNeighborSet:

    def test_identical_sparsity_pattern(self):
        mod = _get_module()
        W_bank = _make_sparse(_cosine_matrix())
        W_knn3 = mod._build_knn(W_bank, k=3)
        W_bin_full = mod._build_binary(W_bank)
        W_bin_knn3 = mod._build_reweighted_from_knn(W_knn3, W_bin_full)

        mask_cosine = W_knn3.toarray() > 0
        mask_binary = W_bin_knn3.toarray() > 0
        np.testing.assert_array_equal(mask_cosine, mask_binary)

    def test_row0_neighbors_are_cols_1_2_3(self):
        mod = _get_module()
        W_bank = _make_sparse(_cosine_matrix())
        W_knn3 = mod._build_knn(W_bank, k=3)
        W_bin_full = mod._build_binary(W_bank)
        W_bin_knn3 = mod._build_reweighted_from_knn(W_knn3, W_bin_full)
        row0 = W_bin_knn3.toarray()[0]
        nonzero_cols = set(np.flatnonzero(row0))
        assert nonzero_cols == {1, 2, 3}


# ---------------------------------------------------------------------------
# Each of a row's k neighbors gets equal weight 1/k; row sums to 1
# ---------------------------------------------------------------------------

class TestBinaryKnnEqualWeights:

    def test_each_neighbor_weight_is_one_over_k(self):
        mod = _get_module()
        W_bank = _make_sparse(_cosine_matrix())
        k = 3
        W_knn3 = mod._build_knn(W_bank, k=k)
        W_bin_full = mod._build_binary(W_bank)
        W_bin_knn3 = mod._build_reweighted_from_knn(W_knn3, W_bin_full)
        row0 = W_bin_knn3.toarray()[0]
        for col in (1, 2, 3):
            assert abs(row0[col] - 1.0 / k) < 1e-9

    def test_row_sums_to_one(self):
        mod = _get_module()
        W_bank = _make_sparse(_cosine_matrix())
        W_knn3 = mod._build_knn(W_bank, k=3)
        W_bin_full = mod._build_binary(W_bank)
        W_bin_knn3 = mod._build_reweighted_from_knn(W_knn3, W_bin_full)
        out = W_bin_knn3.toarray()
        for i in range(out.shape[0]):
            row_sum = out[i].sum()
            if row_sum > 0:
                assert abs(row_sum - 1.0) < 1e-9, f"Row {i} sum={row_sum}"


# ---------------------------------------------------------------------------
# W_bank_count_knn3 shares W_bank_knn3's neighbor set but differs in weight
# ---------------------------------------------------------------------------

class TestCountKnnSharesNeighborSetButDifferentWeights:

    def test_identical_sparsity_pattern(self):
        mod = _get_module()
        W_bank = _make_sparse(_cosine_matrix())
        W_count = _make_sparse(_count_matrix())
        W_knn3 = mod._build_knn(W_bank, k=3)
        W_count_knn3 = mod._build_reweighted_from_knn(W_knn3, W_count)

        mask_cosine = W_knn3.toarray() > 0
        mask_count = W_count_knn3.toarray() > 0
        np.testing.assert_array_equal(mask_cosine, mask_count)

    def test_weights_differ_from_cosine(self):
        """At the shared neighbor positions, count-derived (row-standardised)
        weights differ from cosine-derived (row-standardised) weights."""
        mod = _get_module()
        W_bank = _make_sparse(_cosine_matrix())
        W_count = _make_sparse(_count_matrix())
        W_knn3 = mod._build_knn(W_bank, k=3)
        W_count_knn3 = mod._build_reweighted_from_knn(W_knn3, W_count)

        row_standardize = mod.row_standardize
        cosine_rs = row_standardize(W_knn3).toarray()
        count_rs = W_count_knn3.toarray()

        assert not np.allclose(cosine_rs, count_rs)

    def test_weights_are_not_uniform_within_row(self):
        """Unlike the binary variant, count weights are not uniform 1/k --
        they retain the relative scale of the underlying count values."""
        mod = _get_module()
        W_bank = _make_sparse(_cosine_matrix())
        W_count = _make_sparse(_count_matrix())
        W_knn3 = mod._build_knn(W_bank, k=3)
        W_count_knn3 = mod._build_reweighted_from_knn(W_knn3, W_count)
        row0 = W_count_knn3.toarray()[0]
        nonzero_vals = row0[row0 > 0]
        # count row 0 at cols 1,2,3 = [50, 5, 2] -> not all equal
        assert len(set(np.round(nonzero_vals, 8))) > 1


# ---------------------------------------------------------------------------
# Regression test: old behavior (independent truncation of the full
# binary/count matrix) is NOT what the new functions do.
# ---------------------------------------------------------------------------

class TestRegressionAgainstNaiveIndependentTruncation:
    """
    The original bug applied top-k truncation directly to the full binary
    matrix (or the full count matrix). For binary, this meant
    np.argpartition selected an ARBITRARY k of many tied (all == 1.0)
    entries per row -- a selection that need not match, and generally will
    not match, the cosine-ranked KNN topology. The fixed
    _build_reweighted_from_knn approach always matches the cosine topology
    by construction; this test demonstrates the two approaches diverge on a
    tie-heavy row, which is exactly the scenario that produced the bug.
    """

    def _naive_full_binary_knn(self, W_full_binary, k):
        """Reimplementation of the OLD (buggy) approach: independently
        top-k-truncate the full binary matrix directly, with no reference
        to any cosine ranking. Ties are broken arbitrarily by argpartition."""
        W = W_full_binary.copy()
        np.fill_diagonal(W, 0.0)
        N = W.shape[0]
        W_knn = np.zeros_like(W)
        for i in range(N):
            row = W[i]
            nz = np.count_nonzero(row)
            if nz == 0:
                continue
            if nz <= k:
                W_knn[i] = row
            else:
                top_k = np.argpartition(row, -k)[-k:]
                W_knn[i, top_k] = row[top_k]
        np.fill_diagonal(W_knn, 0.0)
        return W_knn

    def test_naive_binary_truncation_diverges_from_cosine_topology(self):
        mod = _get_module()

        # Row 0 has 9 tied (== 1.0) nonzero entries in the full binary
        # matrix (cols 1..9), but the cosine matrix picks a clear, distinct
        # top-3 (cols 1, 2, 3) among them.
        n = 10
        cosine = np.zeros((n, n))
        cosine[0, 1] = 0.9
        cosine[0, 2] = 0.8
        cosine[0, 3] = 0.7
        for j in range(4, n):
            cosine[0, j] = 0.1
        cosine[:, 0] = cosine[0, :]  # keep matrix well-formed (symmetric row/col 0)

        full_binary = (cosine > 0).astype(float)  # all nonzero entries tied at 1.0

        W_bank = _make_sparse(cosine)
        W_knn3 = mod._build_knn(W_bank, k=3)
        cosine_neighbors = set(np.flatnonzero(W_knn3.toarray()[0]))
        assert cosine_neighbors == {1, 2, 3}

        naive = self._naive_full_binary_knn(full_binary, k=3)
        naive_neighbors = set(np.flatnonzero(naive[0]))

        # The naive independent-truncation approach, applied directly to the
        # tie-degenerate binary matrix, selects a DIFFERENT set of neighbors
        # than the cosine ranking -- this divergence is the bug.
        assert naive_neighbors != cosine_neighbors

        # The FIXED approach, by construction, always matches the cosine
        # topology instead of the naive (tie-arbitrary) one.
        W_bin_full = mod._build_binary(W_bank)
        fixed = mod._build_reweighted_from_knn(W_knn3, W_bin_full)
        fixed_neighbors = set(np.flatnonzero(fixed.toarray()[0]))
        assert fixed_neighbors == cosine_neighbors
        assert fixed_neighbors != naive_neighbors
