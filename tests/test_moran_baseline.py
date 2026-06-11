"""
Tests for analysis/diagnostics/moran_baseline.py

Covers the pure helper functions introduced / refactored in this PR:
  - subset_weights   (extracted from the run() closure in the old version)
  - plot_moran       (new standalone function)
"""
import sys
import os
import importlib
from unittest.mock import MagicMock, patch, call
import numpy as np
import pytest

# conftest.py installs sys.modules mocks.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis", "diagnostics"))

_module = None


def _get_module():
    global _module
    if _module is None:
        _module = importlib.import_module("moran_baseline")
    return _module


# ---------------------------------------------------------------------------
# Minimal W stub used in subset_weights tests
# ---------------------------------------------------------------------------

class _FakeW:
    """Minimal PySAL W-like stub for testing subset_weights."""

    def __init__(self, neighbors_dict):
        """
        neighbors_dict : {int: [int, ...]}  – zero-indexed neighbor lists
        """
        self.neighbors = neighbors_dict
        self.n = len(neighbors_dict)
        self.transform = None

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)


# ---------------------------------------------------------------------------
# subset_weights
# ---------------------------------------------------------------------------

class TestSubsetWeights:
    """Tests for subset_weights(w, present_mask)."""

    def _fn(self, w, present_mask):
        return _get_module().subset_weights(w, present_mask)

    def test_all_present_preserves_structure(self):
        """When all units are present, neighbor lists are unchanged except re-indexed."""
        # 4-unit chain: 0-1-2-3
        nbrs = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
        w = _FakeW(nbrs)
        mask = np.array([True, True, True, True])
        w_sub = self._fn(w, mask)
        assert w_sub.n == 4
        assert w_sub.neighbors[0] == [1]
        assert set(w_sub.neighbors[1]) == {0, 2}

    def test_remove_first_unit(self):
        """Removing unit 0 renumbers remaining units 1→0, 2→1, 3→2."""
        nbrs = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
        w = _FakeW(nbrs)
        mask = np.array([False, True, True, True])
        w_sub = self._fn(w, mask)
        # Remaining: old [1,2,3] → new [0,1,2]
        assert w_sub.n == 3
        # Old 1 (new 0) neighbored old 2 (new 1); old 0 is gone
        assert 1 in w_sub.neighbors[0]
        assert 0 not in w_sub.neighbors[0]

    def test_remove_last_unit(self):
        """Removing the last unit from a chain."""
        nbrs = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
        w = _FakeW(nbrs)
        mask = np.array([True, True, True, False])
        w_sub = self._fn(w, mask)
        assert w_sub.n == 3
        # Old unit 2 (new 2) no longer neighbors old 3
        assert 3 not in w_sub.neighbors[2]

    def test_weight_entries_are_one(self):
        """All retained weights should be 1.0 (row-standardised later)."""
        nbrs = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
        w = _FakeW(nbrs)
        mask = np.array([True, True, True])
        w_sub = self._fn(w, mask)
        for i, wts in w_sub.weights.items():
            for wt in wts:
                assert wt == 1.0

    def test_isolated_unit_has_empty_neighbors(self):
        """A unit with all neighbors removed becomes isolated (empty neighbor list)."""
        nbrs = {0: [1], 1: [0], 2: []}  # unit 2 already isolated
        w = _FakeW(nbrs)
        mask = np.array([True, False, True])  # remove unit 1
        w_sub = self._fn(w, mask)
        # Old unit 0 (new 0) had only old unit 1 as neighbor; now isolated
        assert w_sub.neighbors[0] == []
        # Old unit 2 (new 1) was already isolated
        assert w_sub.neighbors[1] == []

    def test_single_unit_retained(self):
        """Subsetting to a single unit."""
        nbrs = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
        w = _FakeW(nbrs)
        mask = np.array([True, False, False])
        w_sub = self._fn(w, mask)
        assert w_sub.n == 1
        assert w_sub.neighbors[0] == []

    def test_transform_set_to_r(self):
        """After subset, w_sub.transform should be 'r' (row-standardised)."""
        nbrs = {0: [1], 1: [0]}
        w = _FakeW(nbrs)
        mask = np.array([True, True])
        w_sub = self._fn(w, mask)
        assert w_sub.transform == "r"

    def test_five_unit_graph_partial_removal(self):
        """Regression test: 5-unit star graph, remove centre."""
        # Centre (0) connected to all leaves
        nbrs = {0: [1, 2, 3, 4], 1: [0], 2: [0], 3: [0], 4: [0]}
        w = _FakeW(nbrs)
        mask = np.array([False, True, True, True, True])  # remove centre
        w_sub = self._fn(w, mask)
        assert w_sub.n == 4
        # All former leaves lose their only neighbor → all isolated
        for i in range(4):
            assert w_sub.neighbors[i] == []

    def test_two_disconnected_components(self):
        """Units forming two disconnected pairs; removing one from each pair."""
        nbrs = {0: [1], 1: [0], 2: [3], 3: [2]}
        w = _FakeW(nbrs)
        mask = np.array([True, False, True, False])  # keep 0 and 2
        w_sub = self._fn(w, mask)
        assert w_sub.n == 2
        # Both kept units lose their partners → both isolated
        assert w_sub.neighbors[0] == []
        assert w_sub.neighbors[1] == []


# ---------------------------------------------------------------------------
# plot_moran
# ---------------------------------------------------------------------------

class TestPlotMoran:
    """Tests for plot_moran(ax_bar, ax_z, df, title_prefix)."""

    def _fn(self, ax_bar, ax_z, df, title_prefix):
        return _get_module().plot_moran(ax_bar, ax_z, df, title_prefix)

    def _make_df(self, n=5, seed=0):
        """Create a minimal DataFrame matching what plot_moran expects.

        plot_moran accesses df["col"] and calls .mean() on numeric columns
        (specifically df["expected_I"].mean()), so numeric columns must be
        returned as numpy arrays.
        """
        rng = np.random.default_rng(seed)
        sig_flags = [bool(p < 0.05) for p in rng.uniform(0, 1, n).tolist()]

        class FakeDF:
            def __init__(self):
                self._data = {
                    "year":       np.arange(2000, 2000 + n),
                    "moran_I":    rng.standard_normal(n),
                    "expected_I": np.zeros(n),
                    "z_score":    rng.standard_normal(n),
                    "p_value":    rng.uniform(0, 1, n),
                    "significant": sig_flags,
                }

            def __getitem__(self, key):
                return self._data[key]

        return FakeDF()

    def test_calls_bar_and_plot(self):
        """plot_moran should call ax_bar.bar(...) and ax_z.plot(...)."""
        ax_bar = MagicMock()
        ax_z = MagicMock()
        df = self._make_df()
        self._fn(ax_bar, ax_z, df, "Test")
        assert ax_bar.bar.called, "ax_bar.bar should be called"
        assert ax_z.plot.called, "ax_z.plot should be called"

    def test_titles_set(self):
        """Both axes should have their titles set."""
        ax_bar = MagicMock()
        ax_z = MagicMock()
        df = self._make_df()
        self._fn(ax_bar, ax_z, df, "MyPrefix")
        # set_title should be called on both axes
        assert ax_bar.set_title.called
        assert ax_z.set_title.called
        # Title should contain the prefix
        title_call = ax_bar.set_title.call_args[0][0]
        assert "MyPrefix" in title_call

    def test_axhline_zero_drawn(self):
        """ax_bar.axhline(0, ...) should be called to draw zero line."""
        ax_bar = MagicMock()
        ax_z = MagicMock()
        df = self._make_df()
        self._fn(ax_bar, ax_z, df, "X")
        # Check that axhline was called with 0 as first arg somewhere
        zero_calls = [
            c for c in ax_bar.axhline.call_args_list
            if c[0] and c[0][0] == 0
        ]
        assert len(zero_calls) >= 1, "ax_bar.axhline(0) should be called"

    def test_no_exception_with_all_significant(self):
        """Should not raise when all observations are significant."""
        ax_bar = MagicMock()
        ax_z = MagicMock()

        class AllSigDF:
            year = np.array([2000, 2001, 2002])
            moran_I = np.array([0.3, 0.2, 0.4])
            expected_I = np.zeros(3)
            z_score = np.array([3.0, 2.5, 3.5])
            p_value = np.array([0.01, 0.02, 0.005])
            significant = [True, True, True]

            def __getitem__(self, key):
                return getattr(self, key)

        self._fn(ax_bar, ax_z, AllSigDF(), "Sig")

    def test_no_exception_with_none_significant(self):
        """Should not raise when no observations are significant."""
        ax_bar = MagicMock()
        ax_z = MagicMock()

        class NoneSigDF:
            year = np.array([2000, 2001])
            moran_I = np.array([0.05, -0.02])
            expected_I = np.zeros(2)
            z_score = np.array([0.5, -0.3])
            p_value = np.array([0.6, 0.8])
            significant = [False, False]

            def __getitem__(self, key):
                return getattr(self, key)

        self._fn(ax_bar, ax_z, NoneSigDF(), "NonSig")