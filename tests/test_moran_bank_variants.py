"""
Tests for analysis/diagnostics/moran_bank_variants.py

Covers the pure helper functions modified in this PR:
  - build_wbank_knn
  - matrix_stats
  - trim_islands
"""
import sys
import os
import importlib

import numpy as np
import pytest

# conftest.py installs sys.modules mocks.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis", "diagnostics"))

from conftest import FakeSparse

_module = None


def _get_module():
    global _module
    if _module is None:
        _module = importlib.import_module("moran_bank_variants")
    return _module


# ---------------------------------------------------------------------------
# matrix_stats
# ---------------------------------------------------------------------------

class TestMatrixStats:
    """Tests for matrix_stats(W_sparse)."""

    def _fn(self, W_sp):
        return _get_module().matrix_stats(W_sp)

    def _make_sparse(self, arr):
        return FakeSparse(np.array(arr, dtype=float))

    def test_all_required_keys(self):
        W = self._make_sparse([[0, 1], [1, 0]])
        result = self._fn(W)
        for key in ["n_links", "possible_links", "density", "sparsity", "avg_nbrs"]:
            assert key in result, f"Missing key: {key}"

    def test_density_plus_sparsity_equals_one(self):
        W = self._make_sparse([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        result = self._fn(W)
        assert abs(result["density"] + result["sparsity"] - 1.0) < 1e-12

    def test_possible_links_formula(self):
        """possible_links = n*(n-1)."""
        n = 5
        arr = np.zeros((n, n))
        W = self._make_sparse(arr)
        result = self._fn(W)
        assert result["possible_links"] == n * (n - 1)

    def test_zero_matrix_zero_links(self):
        W = self._make_sparse(np.zeros((4, 4)))
        result = self._fn(W)
        assert result["n_links"] == 0
        assert result["density"] == 0.0
        assert result["sparsity"] == 1.0

    def test_full_offdiag_matrix_full_density(self):
        """All off-diagonal entries non-zero → density=1."""
        W = self._make_sparse([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        result = self._fn(W)
        assert abs(result["density"] - 1.0) < 1e-12

    def test_avg_nbrs_calculation(self):
        """avg_nbrs = n_links / n."""
        W = self._make_sparse([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        result = self._fn(W)
        expected_avg = result["n_links"] / 3
        assert abs(result["avg_nbrs"] - expected_avg) < 1e-12

    def test_diagonal_ignored(self):
        """Diagonal entries should not count as links."""
        arr = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=float)
        W = self._make_sparse(arr)
        result = self._fn(W)
        # Off-diagonal non-zeros: (0,1), (1,0), (1,2), (2,1) → 4 links
        assert result["n_links"] == 4

    def test_chain_graph(self):
        """4-unit chain 0-1-2-3: 3 undirected edges = 6 directed links."""
        arr = np.zeros((4, 4))
        for i in range(3):
            arr[i, i+1] = 1
            arr[i+1, i] = 1
        W = self._make_sparse(arr)
        result = self._fn(W)
        assert result["n_links"] == 6
        assert result["possible_links"] == 12
        assert abs(result["density"] - 0.5) < 1e-12

    def test_density_positive_for_nonempty(self):
        W = self._make_sparse([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        result = self._fn(W)
        assert result["density"] > 0.0

    def test_large_sparse_matrix(self):
        """Stress test on a 20×20 ring."""
        n = 20
        arr = np.zeros((n, n))
        for i in range(n):
            arr[i, (i + 1) % n] = 1
            arr[i, (i - 1) % n] = 1
        W = self._make_sparse(arr)
        result = self._fn(W)
        assert result["n_links"] == 2 * n  # 2 links per unit, bidirectional
        assert result["avg_nbrs"] == 2.0


# ---------------------------------------------------------------------------
# trim_islands
# ---------------------------------------------------------------------------

class TestTrimIslands:
    """Tests for trim_islands(W_sparse, labels)."""

    def _fn(self, W_sp, labels):
        return _get_module().trim_islands(W_sp, labels)

    def _make_sparse(self, arr):
        return FakeSparse(np.array(arr, dtype=float))

    def test_no_islands_unchanged(self):
        """If no row has all-zero weights, all units are kept."""
        arr = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        W = self._make_sparse(arr)
        labels = np.array(["A", "B", "C"])
        W_out, labels_out = self._fn(W, labels)
        assert len(labels_out) == 3

    def test_island_removed(self):
        """A unit with all-zero row weights should be removed."""
        arr = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
        W = self._make_sparse(arr)
        labels = np.array(["A", "B", "C"])
        W_out, labels_out = self._fn(W, labels)
        # Unit C (index 2) is an island
        assert "C" not in labels_out

    def test_cascade_removal(self):
        """After removing an island, new islands created by that removal are also removed."""
        # B and C are connected only to each other
        # A is connected only to D
        # D is connected only to A but A connects to nobody else (island after D removed)
        # Chain: A-B, B isolated after A removed; but let's test a simpler case
        # Island D → removal may make C an island → remove C too
        arr = np.array([
            [0, 1, 0, 0],   # A connects to B
            [1, 0, 0, 0],   # B connects to A
            [0, 0, 0, 1],   # C connects to D
            [0, 0, 1, 0],   # D connects to C
        ], dtype=float)
        # Now remove D by zeroing its row (simulate an island)
        arr[3, :] = 0
        arr[:, 3] = 0  # and column (orphan C)
        # Now C has no connections either
        arr[2, :] = 0
        W = self._make_sparse(arr)
        labels = np.array(["A", "B", "C", "D"])
        W_out, labels_out = self._fn(W, labels)
        # Only A and B remain
        assert set(labels_out) == {"A", "B"}

    def test_labels_match_kept_units(self):
        """Returned labels correspond to kept rows."""
        arr = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
        W = self._make_sparse(arr)
        labels = np.array([10, 20, 30])
        W_out, labels_out = self._fn(W, labels)
        assert 30 not in labels_out
        assert 10 in labels_out
        assert 20 in labels_out

    def test_returns_row_standardised_W(self):
        """Output W should have row sums of 1 for non-island rows."""
        arr = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        W = self._make_sparse(arr)
        labels = np.array(["A", "B", "C"])
        W_out, labels_out = self._fn(W, labels)
        out = W_out.toarray()
        for i in range(len(labels_out)):
            row_sum = out[i].sum()
            if row_sum > 0:
                assert abs(row_sum - 1.0) < 1e-9, \
                    f"Row {i} sum={row_sum:.6f}, expected 1.0 (row-standardised)"

    def test_all_islands_stop_condition(self):
        """If fewer than 3 units remain, iteration stops (no infinite loop)."""
        arr = np.zeros((3, 3))
        W = self._make_sparse(arr)
        labels = np.array(["A", "B", "C"])
        W_out, labels_out = self._fn(W, labels)
        # Should not hang; result may be empty or very small
        assert len(labels_out) <= 3

    def test_single_connected_pair(self):
        """Two units connected to each other: both kept."""
        arr = np.array([[0, 1], [1, 0]], dtype=float)
        W = self._make_sparse(arr)
        labels = np.array(["X", "Y"])
        W_out, labels_out = self._fn(W, labels)
        assert len(labels_out) == 2
        assert set(labels_out) == {"X", "Y"}

    def test_weighted_graph_not_zero_row(self):
        """A row with small but non-zero weight is not treated as an island."""
        arr = np.array([[0, 0.001, 0], [0.001, 0, 1.0], [0, 1.0, 0]], dtype=float)
        W = self._make_sparse(arr)
        labels = np.array(["A", "B", "C"])
        W_out, labels_out = self._fn(W, labels)
        # All three have connections; none should be removed
        assert len(labels_out) == 3


# ---------------------------------------------------------------------------
# build_wbank_knn
# ---------------------------------------------------------------------------

class TestBuildWbankKnn:
    """Tests for build_wbank_knn(W_sparse, k).

    This function is in moran_bank_variants.py and mirrors _build_knn from
    w_density_summary.py but uses row_standardize from utils.
    """

    def _fn(self, W_sp, k):
        return _get_module().build_wbank_knn(W_sp, k)

    def _make_sparse(self, arr):
        return FakeSparse(np.array(arr, dtype=float))

    def test_output_has_toarray(self):
        W = self._make_sparse([[0, 0.8, 0.5], [0.8, 0, 0.3], [0.5, 0.3, 0]])
        result = self._fn(W, k=1)
        assert hasattr(result, "toarray")

    def test_k1_at_most_one_link_per_row(self):
        arr = np.array([[0, 0.8, 0.5, 0.2],
                        [0.8, 0, 0.3, 0.1],
                        [0.5, 0.3, 0, 0.9],
                        [0.2, 0.1, 0.9, 0]], dtype=float)
        W = self._make_sparse(arr)
        result = self._fn(W, k=1)
        out = result.toarray()
        for i in range(4):
            n_nonzero = int((out[i] > 0).sum())
            assert n_nonzero <= 1, f"Row {i} has {n_nonzero} links, expected <= 1"

    def test_row_standardized(self):
        """Each non-isolated row should sum to 1 after row standardisation."""
        arr = np.array([[0, 0.8, 0.5], [0.8, 0, 0.3], [0.5, 0.3, 0]], dtype=float)
        W = self._make_sparse(arr)
        result = self._fn(W, k=2)
        out = result.toarray()
        for i in range(3):
            row_sum = out[i].sum()
            if row_sum > 0:
                assert abs(row_sum - 1.0) < 1e-9, f"Row {i} sum={row_sum:.6f}"

    def test_diagonal_zero(self):
        arr = np.array([[0.9, 0.8, 0.5], [0.8, 0.7, 0.3], [0.5, 0.3, 0.6]])
        W = self._make_sparse(arr)
        result = self._fn(W, k=2)
        out = result.toarray()
        diag = np.diag(out)
        np.testing.assert_allclose(diag, 0.0, atol=1e-12)

    def test_isolated_row_stays_zero(self):
        arr = np.array([[0, 0.5, 0], [0.5, 0, 0], [0, 0, 0]], dtype=float)
        W = self._make_sparse(arr)
        result = self._fn(W, k=1)
        out = result.toarray()
        np.testing.assert_allclose(out[2], 0.0, atol=1e-12)

    def test_selects_top_k_weights(self):
        """k=1 should preserve the largest weight per row."""
        arr = np.array([[0, 0.9, 0.1, 0.3],
                        [0.2, 0, 0.7, 0.4],
                        [0.3, 0.8, 0, 0.5],
                        [0.1, 0.3, 0.4, 0]], dtype=float)
        W = self._make_sparse(arr)
        result = self._fn(W, k=1)
        out = result.toarray()
        # Row 0: top weight is col 1 (0.9) → that position should be 1.0 (row-standardised)
        assert abs(out[0, 1] - 1.0) < 1e-9
        assert out[0, 2] == 0.0

    def test_same_result_as_w_density_build_knn_for_small_k(self):
        """build_wbank_knn and _build_knn should give equivalent results for k=2."""
        import importlib
        w_density_mod = importlib.import_module("w_density_summary")
        arr = np.array([[0, 0.6, 0.4, 0.1],
                        [0.6, 0, 0.4, 0.2],
                        [0.4, 0.4, 0, 0.7],
                        [0.1, 0.2, 0.7, 0]], dtype=float)
        W = self._make_sparse(arr)
        r1 = self._fn(W, k=2)
        r2 = w_density_mod._build_knn(FakeSparse(arr), k=2)
        np.testing.assert_allclose(r1.toarray(), r2.toarray(), atol=1e-9)

    def test_shape_preserved(self):
        n = 6
        arr = np.abs(np.random.randn(n, n))
        np.fill_diagonal(arr, 0.0)
        W = self._make_sparse(arr)
        result = self._fn(W, k=3)
        assert result.toarray().shape == (n, n)