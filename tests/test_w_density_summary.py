"""
Tests for analysis/diagnostics/w_density_summary.py

Covers the pure helper functions introduced in this PR:
  - _row_stats  (including new mean_nonzero_weight / median_nonzero_weight fields)
  - _build_knn
  - _state_split_rows  (new function in this PR)
"""
import sys
import os
import importlib

import numpy as np
import pytest

# conftest.py installs sys.modules mocks.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis", "diagnostics"))

_module = None


def _get_module():
    global _module
    if _module is None:
        _module = importlib.import_module("w_density_summary")
    return _module


# ---------------------------------------------------------------------------
# FakeSparse helper (from conftest.FakeSparse but locally defined for clarity)
# ---------------------------------------------------------------------------

from conftest import FakeSparse


# ---------------------------------------------------------------------------
# _row_stats
# ---------------------------------------------------------------------------

class TestRowStats:
    """Tests for _row_stats(W_sp, label)."""

    def _fn(self, W_sp, label="test"):
        return _get_module()._row_stats(W_sp, label)

    def _make_sparse(self, arr):
        return FakeSparse(np.array(arr, dtype=float))

    def test_label_preserved(self):
        W = self._make_sparse([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        result = self._fn(W, "MyMatrix")
        assert result["matrix"] == "MyMatrix"

    def test_n_equals_nrows(self):
        W = self._make_sparse([[0, 1], [1, 0]])
        result = self._fn(W)
        assert result["N"] == 2

    def test_nnz_excludes_diagonal(self):
        """Diagonal should be zeroed before counting nnz."""
        # 3×3 identity (all on diagonal) → after setdiag(0) → nnz=0
        W = self._make_sparse(np.eye(3))
        result = self._fn(W)
        assert result["nnz"] == 0

    def test_density_pct_simple(self):
        """2×2 off-diagonal matrix: 2 links out of 2 possible → 100%."""
        W = self._make_sparse([[0, 1], [1, 0]])
        result = self._fn(W)
        assert abs(result["density_pct"] - 100.0) < 1e-6

    def test_density_pct_zero(self):
        """Empty W → density 0%."""
        W = self._make_sparse(np.zeros((3, 3)))
        result = self._fn(W)
        assert result["density_pct"] == 0.0

    def test_mean_nbrs(self):
        """Chain graph: 0-1-2; county 0 has 1 nbr, 1 has 2, 2 has 1 → mean=4/3."""
        W = self._make_sparse([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        result = self._fn(W)
        expected_mean = 4.0 / 3.0
        assert abs(result["mean_nbrs"] - expected_mean) < 1e-9

    def test_median_nbrs(self):
        """Chain graph: nbrs = [1, 2, 1] → median = 1."""
        W = self._make_sparse([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        result = self._fn(W)
        assert result["median_nbrs"] == 1.0

    def test_pct_isolated_all_isolated(self):
        """All-zero matrix → 100% isolated."""
        W = self._make_sparse(np.zeros((4, 4)))
        result = self._fn(W)
        assert result["pct_isolated"] == 100.0

    def test_pct_isolated_none_isolated(self):
        """Fully connected (off-diag) → 0% isolated."""
        W = self._make_sparse([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        result = self._fn(W)
        assert result["pct_isolated"] == 0.0

    def test_pct_isolated_partial(self):
        """3-county matrix where 1 county has no connections → ~33% isolated."""
        W = self._make_sparse([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        result = self._fn(W)
        expected = 100.0 / 3.0
        assert abs(result["pct_isolated"] - expected) < 0.01

    def test_weighted_matrix_mean_nbrs(self):
        """Function counts nonzero entries, not sum of weights."""
        W = self._make_sparse([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
        result = self._fn(W)
        # Every county has 2 non-zero neighbors
        assert result["mean_nbrs"] == 2.0

    def test_all_fields_present(self):
        """Result dict must contain all required keys including new weight stats."""
        W = self._make_sparse([[0, 1], [1, 0]])
        result = self._fn(W)
        for key in ["matrix", "N", "nnz", "density_pct", "mean_nbrs",
                    "median_nbrs", "pct_isolated",
                    "mean_nonzero_weight", "median_nonzero_weight"]:
            assert key in result, f"Missing key: {key}"

    def test_mean_nonzero_weight_uniform(self):
        """All nonzero weights equal 1 → mean_nonzero_weight = 1."""
        W = self._make_sparse([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        result = self._fn(W)
        assert abs(result["mean_nonzero_weight"] - 1.0) < 1e-9

    def test_median_nonzero_weight_uniform(self):
        """All nonzero weights equal 1 → median_nonzero_weight = 1."""
        W = self._make_sparse([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        result = self._fn(W)
        assert abs(result["median_nonzero_weight"] - 1.0) < 1e-9

    def test_mean_nonzero_weight_weighted(self):
        """Mean of nonzero off-diagonal weights computed correctly."""
        # Off-diagonal: 0.5, 0.5, 0.5, 0.5 → mean = 0.5
        W = self._make_sparse([[0, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0]])
        result = self._fn(W)
        assert abs(result["mean_nonzero_weight"] - 0.5) < 1e-9

    def test_mean_nonzero_weight_mixed(self):
        """Mixed weights: mean computed over all nonzero entries."""
        # Off-diag entries: 0.2 and 0.8 → mean = 0.5
        arr = [[0, 0.2], [0.8, 0]]
        W = self._make_sparse(arr)
        result = self._fn(W)
        expected_mean = (0.2 + 0.8) / 2.0
        assert abs(result["mean_nonzero_weight"] - expected_mean) < 1e-9

    def test_nonzero_weight_nan_when_no_links(self):
        """All-zero matrix → mean/median weight should be NaN (no entries)."""
        import math
        W = self._make_sparse(np.zeros((3, 3)))
        result = self._fn(W)
        assert math.isnan(result["mean_nonzero_weight"])
        assert math.isnan(result["median_nonzero_weight"])

    def test_large_matrix(self):
        """Spot-check on a 10×10 ring graph."""
        N = 10
        arr = np.zeros((N, N))
        for i in range(N):
            arr[i, (i + 1) % N] = 1
            arr[i, (i - 1) % N] = 1
        W = self._make_sparse(arr)
        result = self._fn(W)
        assert result["N"] == 10
        assert result["mean_nbrs"] == 2.0
        assert result["pct_isolated"] == 0.0


# ---------------------------------------------------------------------------
# _build_knn
# ---------------------------------------------------------------------------

class TestBuildKnn:
    """Tests for _build_knn(W_sp, k)."""

    def _fn(self, W_sp, k):
        return _get_module()._build_knn(W_sp, k)

    def _make_sparse(self, arr):
        return FakeSparse(np.array(arr, dtype=float))

    def test_output_is_sparse(self):
        """Return value should be a FakeSparse (or CSR-like) object."""
        W = self._make_sparse([[0, 0.8, 0.5], [0.8, 0, 0.3], [0.5, 0.3, 0]])
        result = self._fn(W, k=1)
        # Should have a toarray method
        assert hasattr(result, "toarray")

    def test_k1_each_row_has_at_most_one_link(self):
        """With k=1, each row should have at most 1 non-zero entry."""
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
        """After _build_knn, each non-isolated row should sum to 1."""
        arr = np.array([[0, 0.8, 0.5], [0.8, 0, 0.3], [0.5, 0.3, 0]], dtype=float)
        W = self._make_sparse(arr)
        result = self._fn(W, k=2)
        out = result.toarray()
        for i in range(3):
            row_sum = out[i].sum()
            if row_sum > 0:
                assert abs(row_sum - 1.0) < 1e-9, f"Row {i} sum={row_sum:.6f}, expected 1.0"

    def test_diagonal_is_zero(self):
        """Diagonal should remain 0 (self-links excluded)."""
        arr = np.array([[0.9, 0.8, 0.5], [0.8, 0.7, 0.3], [0.5, 0.3, 0.6]])
        W = self._make_sparse(arr)
        result = self._fn(W, k=2)
        out = result.toarray()
        diag = np.diag(out)
        np.testing.assert_allclose(diag, 0.0, atol=1e-12)

    def test_isolated_row_stays_zero(self):
        """A row with all zeros (island) stays zero after KNN."""
        arr = np.array([[0, 0.5, 0], [0.5, 0, 0], [0, 0, 0]], dtype=float)
        W = self._make_sparse(arr)
        result = self._fn(W, k=1)
        out = result.toarray()
        np.testing.assert_allclose(out[2], 0.0, atol=1e-12)

    def test_k_geq_nonzero_preserves_all_links(self):
        """When k >= number of non-zero links per row, all links are kept."""
        arr = np.array([[0, 0.6, 0.4], [0.6, 0, 0.4], [0.4, 0.4, 0]], dtype=float)
        W = self._make_sparse(arr)
        result_k10 = self._fn(W, k=10)
        result_k2 = self._fn(W, k=2)
        out_k10 = result_k10.toarray()
        out_k2 = result_k2.toarray()
        # Both should produce the same row-standardised result since each row has 2 links
        np.testing.assert_allclose(out_k10, out_k2, atol=1e-9)

    def test_selects_top_k_by_weight(self):
        """k=1 should keep only the largest weight per row."""
        arr = np.array([[0, 0.9, 0.1, 0.3],
                        [0.2, 0, 0.7, 0.4],
                        [0.3, 0.8, 0, 0.5],
                        [0.1, 0.3, 0.4, 0]], dtype=float)
        W = self._make_sparse(arr)
        result = self._fn(W, k=1)
        out = result.toarray()
        # Row 0: largest is col 1 (0.9)
        assert out[0, 1] > 0.0
        assert out[0, 2] == 0.0
        assert out[0, 3] == 0.0
        # Row 1: largest is col 2 (0.7)
        assert out[1, 2] > 0.0

    def test_output_shape(self):
        """Output shape should match input shape."""
        n = 5
        arr = np.abs(np.random.randn(n, n))
        np.fill_diagonal(arr, 0.0)
        W = self._make_sparse(arr)
        result = self._fn(W, k=2)
        assert result.toarray().shape == (n, n)

    def test_symmetric_input_produces_row_standardised_output(self):
        """A symmetric weight matrix with k=2 should be row-standardised."""
        arr = np.array([[0, 0.5, 0.3, 0.1],
                        [0.5, 0, 0.4, 0.2],
                        [0.3, 0.4, 0, 0.6],
                        [0.1, 0.2, 0.6, 0]], dtype=float)
        W = self._make_sparse(arr)
        result = self._fn(W, k=2)
        out = result.toarray()
        for i in range(4):
            row_sum = out[i].sum()
            if row_sum > 0:
                assert abs(row_sum - 1.0) < 1e-9, f"Row {i} sum={row_sum:.6f}"


# ---------------------------------------------------------------------------
# _state_split_rows  (new function in this PR)
# ---------------------------------------------------------------------------

class TestStateSplitRows:
    """Tests for _state_split_rows(W_raw, county_order)."""

    def _fn(self, W_raw, county_order):
        return _get_module()._state_split_rows(W_raw, county_order)

    def test_returns_two_rows(self):
        """Function always returns a list of exactly two dicts."""
        W = np.zeros((4, 4))
        county_order = ["01001", "01002", "02001", "02002"]
        result = self._fn(W, county_order)
        assert len(result) == 2

    def test_first_row_is_same_state(self):
        W = np.zeros((4, 4))
        county_order = ["01001", "01002", "02001", "02002"]
        result = self._fn(W, county_order)
        assert result[0]["matrix"] == "W_bank same-state entries"

    def test_second_row_is_cross_state(self):
        W = np.zeros((4, 4))
        county_order = ["01001", "01002", "02001", "02002"]
        result = self._fn(W, county_order)
        assert result[1]["matrix"] == "W_bank cross-state entries"

    def test_N_correct(self):
        W = np.zeros((6, 6))
        county_order = ["01001", "01002", "01003", "02001", "02002", "02003"]
        result = self._fn(W, county_order)
        assert result[0]["N"] == 6
        assert result[1]["N"] == 6

    def test_zero_W_no_nonzero_entries(self):
        """All-zero W_raw → nnz=0 for both rows."""
        N = 6
        W = np.zeros((N, N))
        county_order = [f"0{i}001" for i in range(1, N + 1)]
        result = self._fn(W, county_order)
        for row in result:
            assert row["nnz"] == 0

    def test_all_same_state_no_cross_state_links(self):
        """All counties in same state → cross-state row has nnz=0."""
        N = 4
        W = np.array([
            [0.0, 0.5, 0.3, 0.2],
            [0.5, 0.0, 0.4, 0.1],
            [0.3, 0.4, 0.0, 0.3],
            [0.2, 0.1, 0.3, 0.0],
        ])
        county_order = ["01001", "01002", "01003", "01004"]  # all state "01"
        result = self._fn(W, county_order)
        same_row = result[0]
        cross_row = result[1]
        # All pairs are same-state → same-state nnz = all off-diagonal nonzeros
        expected_nnz = int((W > 0).sum())
        assert same_row["nnz"] == expected_nnz
        assert cross_row["nnz"] == 0

    def test_all_different_states_no_same_state_links(self):
        """All counties in different states → same-state row has nnz=0."""
        N = 4
        W = np.array([
            [0.0, 0.3, 0.4, 0.5],
            [0.3, 0.0, 0.2, 0.6],
            [0.4, 0.2, 0.0, 0.1],
            [0.5, 0.6, 0.1, 0.0],
        ])
        county_order = ["01001", "02001", "03001", "04001"]  # different states
        result = self._fn(W, county_order)
        same_row = result[0]
        assert same_row["nnz"] == 0
        # Cross-state should have all off-diagonal entries
        cross_row = result[1]
        assert cross_row["nnz"] == int((W > 0).sum())

    def test_density_pct_range(self):
        """density_pct should be between 0 and 100."""
        N = 6
        rng = np.random.default_rng(42)
        W = rng.uniform(0, 1, (N, N))
        np.fill_diagonal(W, 0)
        county_order = ["01001", "01002", "01003", "02001", "02002", "02003"]
        result = self._fn(W, county_order)
        for row in result:
            assert 0.0 <= row["density_pct"] <= 100.0

    def test_mean_nonzero_weight_same_state(self):
        """mean_nonzero_weight should reflect actual same-state nonzero values."""
        # 4 counties: 2 in state 01, 2 in state 02
        # Only same-state links: (0,1) = 0.6, (1,0) = 0.6; (2,3) = 0.4, (3,2) = 0.4
        W = np.array([
            [0.0, 0.6, 0.0, 0.0],
            [0.6, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.4],
            [0.0, 0.0, 0.4, 0.0],
        ])
        county_order = ["01001", "01002", "02001", "02002"]
        result = self._fn(W, county_order)
        same_row = result[0]
        # All same-state nonzero values: 0.6, 0.6, 0.4, 0.4 → mean = 0.5
        assert abs(same_row["mean_nonzero_weight"] - 0.5) < 1e-9

    def test_median_nonzero_weight_cross_state(self):
        """median_nonzero_weight should reflect cross-state nonzero values."""
        # 4 counties: 01001, 01002, 02001, 02002
        # Cross-state links: (0,2)=0.3, (0,3)=0.7, (1,2)=0.5, (1,3)=0.2, and symmetric
        W = np.array([
            [0.0, 0.0, 0.3, 0.7],
            [0.0, 0.0, 0.5, 0.2],
            [0.3, 0.5, 0.0, 0.0],
            [0.7, 0.2, 0.0, 0.0],
        ])
        county_order = ["01001", "01002", "02001", "02002"]
        result = self._fn(W, county_order)
        cross_row = result[1]
        # Cross-state values: 0.3, 0.7, 0.5, 0.2, 0.3, 0.5, 0.7, 0.2
        nz_vals = np.array([0.3, 0.7, 0.5, 0.2, 0.3, 0.5, 0.7, 0.2])
        expected_median = float(np.median(nz_vals))
        assert abs(cross_row["median_nonzero_weight"] - expected_median) < 1e-9

    def test_pct_isolated_no_same_state_links(self):
        """If no same-state links exist, pct_isolated for same-state = 100%."""
        # All counties have different states
        N = 4
        W = np.array([
            [0.0, 0.5, 0.3, 0.0],
            [0.5, 0.0, 0.0, 0.4],
            [0.3, 0.0, 0.0, 0.2],
            [0.0, 0.4, 0.2, 0.0],
        ])
        county_order = ["01001", "02001", "03001", "04001"]
        result = self._fn(W, county_order)
        same_row = result[0]
        assert same_row["pct_isolated"] == 100.0

    def test_no_links_nan_weights(self):
        """No links → mean/median weights should be NaN (no nonzero values)."""
        import math
        N = 4
        W = np.zeros((N, N))
        county_order = ["01001", "01002", "02001", "02002"]
        result = self._fn(W, county_order)
        for row in result:
            assert math.isnan(row["mean_nonzero_weight"])
            assert math.isnan(row["median_nonzero_weight"])

    def test_diagonal_ignored(self):
        """Self-links (diagonal) should not affect statistics."""
        # Same W but with nonzero diagonal
        W_no_diag = np.array([
            [0.0, 0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.5],
            [0.0, 0.0, 0.5, 0.0],
        ])
        W_with_diag = W_no_diag.copy()
        np.fill_diagonal(W_with_diag, 0.9)  # large diagonal
        county_order = ["01001", "01002", "02001", "02002"]
        result_no_diag = self._fn(W_no_diag, county_order)
        result_with_diag = self._fn(W_with_diag, county_order)
        # Diagonal values should be masked; results should be identical
        for i in range(2):
            assert result_no_diag[i]["nnz"] == result_with_diag[i]["nnz"]
            assert abs(
                result_no_diag[i]["mean_nonzero_weight"] -
                result_with_diag[i]["mean_nonzero_weight"]
            ) < 1e-9

    def test_density_pct_uses_eligible_pairs(self):
        """density_pct denominator = # of eligible pairs of the given type."""
        # 2 counties per state × 2 states = 4 counties
        # Same-state pairs: (0,1), (1,0), (2,3), (3,2) = 4 pairs
        # Cross-state pairs: (0,2),(0,3),(1,2),(1,3) + reverses = 8 pairs
        N = 4
        county_order = ["01001", "01002", "02001", "02002"]

        # One same-state link pair (0↔1, 2 entries)
        W_same = np.zeros((N, N))
        W_same[0, 1] = W_same[1, 0] = 1.0
        result = self._fn(W_same, county_order)
        # density_pct for same-state: 2 nnz / 4 eligible pairs = 50%
        assert abs(result[0]["density_pct"] - 50.0) < 1e-9
        # density_pct for cross-state: 0 / 8 = 0%
        assert result[1]["density_pct"] == 0.0
