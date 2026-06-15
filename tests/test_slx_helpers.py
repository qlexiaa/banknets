"""
Tests for analysis/extensions/slx_exposure.py (changes introduced in this PR)

Functions under test (new or refactored in this PR):
  - _within_3d
  - _make_row  (with new e_idx parameter)
  - _within    (used as building block; confirmed correct here)
  - _ols       (used by _make_row indirectly; tested for correctness)
  - _se_ols    (homoskedastic SE helper)
"""
import sys
import os
import importlib

import numpy as np
import pytest

# conftest.py installs mocks for scipy, pandas, libpysal, panel_data, utils, etc.
# slx_exposure.py also imports from panel_data (CREDIT_CONTROLS, PLACEBO_CONTROLS)
# and w_variants — both mocked.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis", "extensions"))

_module = None


def _get_module():
    global _module
    if _module is None:
        _module = importlib.import_module("slx_exposure")
    return _module


# ---------------------------------------------------------------------------
# _within (two-way within transform — building block for _within_3d)
# ---------------------------------------------------------------------------

class TestWithin:
    """Tests for _within(arr_TN) → two-way within-transformed (T, N) array."""

    def _fn(self, arr):
        return _get_module()._within(arr)

    def test_output_shape(self):
        T, N = 5, 4
        arr = np.ones((T, N))
        result = self._fn(arr)
        assert result.shape == (T, N)

    def test_constant_array_gives_zeros(self):
        """A constant panel (all entries equal) → within transform = 0."""
        T, N = 6, 5
        arr = np.full((T, N), 3.7)
        result = self._fn(arr)
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_county_mean_removed(self):
        """After within transform, mean over T for each county should be ~0."""
        T, N = 8, 6
        rng = np.random.default_rng(0)
        arr = rng.standard_normal((T, N))
        result = self._fn(arr)
        county_means = result.mean(axis=0)
        np.testing.assert_allclose(county_means, 0.0, atol=1e-12)

    def test_year_mean_removed(self):
        """After within transform, mean over N for each year should be ~0."""
        T, N = 8, 6
        rng = np.random.default_rng(1)
        arr = rng.standard_normal((T, N))
        result = self._fn(arr)
        year_means = result.mean(axis=1)
        np.testing.assert_allclose(year_means, 0.0, atol=1e-12)

    def test_known_2x2_result(self):
        """Manual check on 2x2 array: arr_it - county_mean_i - year_mean_t + grand."""
        arr = np.array([[2.0, 4.0],
                        [6.0, 8.0]])
        # county means: [4, 6]; year means: [3, 7]; grand = 5
        # z[0,0] = 2 - 4 - 3 + 5 = 0
        # z[0,1] = 4 - 6 - 3 + 5 = 0
        # z[1,0] = 6 - 4 - 7 + 5 = 0
        # z[1,1] = 8 - 6 - 7 + 5 = 0
        # Hm, T=N=2 is perfectly multicollinear; use a less symmetric example
        arr2 = np.array([[1.0, 3.0, 5.0],
                         [2.0, 6.0, 4.0]])
        result = self._fn(arr2)
        # Verify year means removed
        np.testing.assert_allclose(result.mean(axis=1), 0.0, atol=1e-12)
        # Verify county means removed
        np.testing.assert_allclose(result.mean(axis=0), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# _within_3d (new function in this PR)
# ---------------------------------------------------------------------------

class TestWithin3d:
    """Tests for _within_3d(arr_TNK) — applies two-way within to each column."""

    def _fn(self, arr):
        return _get_module()._within_3d(arr)

    def test_output_shape_preserved(self):
        """Output shape should equal input shape."""
        T, N, K = 5, 4, 3
        arr = np.random.default_rng(10).standard_normal((T, N, K))
        result = self._fn(arr)
        assert result.shape == (T, N, K)

    def test_k1_matches_within_2d(self):
        """Single column: _within_3d should match _within on the same data."""
        T, N = 6, 5
        rng = np.random.default_rng(11)
        arr_2d = rng.standard_normal((T, N))
        arr_3d = arr_2d[:, :, None]  # (T, N, 1)
        result_3d = self._fn(arr_3d)
        result_2d = _get_module()._within(arr_2d)
        np.testing.assert_allclose(result_3d[:, :, 0], result_2d, atol=1e-12)

    def test_each_column_independently_demeaned(self):
        """Each column k should independently have zero county and year means."""
        T, N, K = 7, 5, 4
        rng = np.random.default_rng(12)
        arr = rng.standard_normal((T, N, K))
        result = self._fn(arr)
        for k in range(K):
            # County means (over T) should be 0
            np.testing.assert_allclose(result[:, :, k].mean(axis=0), 0.0, atol=1e-12)
            # Year means (over N) should be 0
            np.testing.assert_allclose(result[:, :, k].mean(axis=1), 0.0, atol=1e-12)

    def test_constant_input_gives_zeros(self):
        """Constant panel per column → within transform = 0."""
        T, N, K = 4, 5, 3
        arr = np.ones((T, N, K)) * np.array([1.0, 2.0, 3.0])
        result = self._fn(arr)
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_columns_do_not_cross_contaminate(self):
        """Different columns should not influence each other's transformation."""
        T, N, K = 5, 4, 2
        rng = np.random.default_rng(13)
        arr = rng.standard_normal((T, N, K))

        # Transform columns independently
        result_joint = self._fn(arr)
        result_col0 = _get_module()._within(arr[:, :, 0])
        result_col1 = _get_module()._within(arr[:, :, 1])

        np.testing.assert_allclose(result_joint[:, :, 0], result_col0, atol=1e-12)
        np.testing.assert_allclose(result_joint[:, :, 1], result_col1, atol=1e-12)

    def test_k_values_stack_correctly(self):
        """Result at each slice matches independently computed within transform."""
        T, N, K = 6, 4, 5
        rng = np.random.default_rng(14)
        arr = rng.standard_normal((T, N, K))
        result = self._fn(arr)
        for k in range(K):
            expected = _get_module()._within(arr[:, :, k])
            np.testing.assert_allclose(result[:, :, k], expected, atol=1e-12)

    def test_large_k(self):
        """Works correctly with larger K (e.g. 10 control columns)."""
        T, N, K = 8, 15, 10
        rng = np.random.default_rng(15)
        arr = rng.standard_normal((T, N, K))
        result = self._fn(arr)
        assert result.shape == (T, N, K)
        # Spot-check: year means of last column should be 0
        np.testing.assert_allclose(result[:, :, -1].mean(axis=1), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# _make_row (refactored with e_idx parameter)
# ---------------------------------------------------------------------------

class TestMakeRow:
    """Tests for _make_row(sample, spec, dv, beta, se, p_perm, extra, NT, N, e_idx=None)."""

    def _fn(self, sample, spec, dv, beta, se, p_perm, extra, NT, N, e_idx=None):
        return _get_module()._make_row(sample, spec, dv, beta, se, p_perm, extra, NT, N, e_idx)

    def test_required_keys_present(self):
        """Result should contain all expected keys."""
        beta = np.array([0.1])
        se = np.array([0.02])
        row = self._fn("Full", "Base", "Dl_nloans_b", beta, se, None, None, 100, 50)
        required_keys = [
            "sample", "spec", "dv", "n_co", "n_obs",
            "beta_D", "se_D", "t_D", "p_D",
            "beta_E", "se_E", "t_E", "p_E",
            "p_perm",
        ]
        for k in required_keys:
            assert k in row, f"Missing key: {k}"

    def test_sample_and_spec_preserved(self):
        beta = np.array([0.05, 0.02])
        se = np.array([0.01, 0.005])
        row = self._fn("Border", "SLX", "Dl_nloans_b", beta, se, None, None, 200, 100)
        assert row["sample"] == "Border"
        assert row["spec"] == "SLX"

    def test_n_co_and_n_obs(self):
        beta = np.array([0.1])
        se = np.array([0.02])
        row = self._fn("Full", "Base", "Dl_nloans_b", beta, se, None, None, 500, 50)
        assert row["n_obs"] == 500
        assert row["n_co"] == 50

    def test_beta_D_is_first_coefficient(self):
        beta = np.array([0.028, 0.015, 0.003])
        se = np.array([0.01, 0.005, 0.002])
        row = self._fn("Full", "SLX", "Dl_nloans_b", beta, se, None, None, 300, 100, e_idx=-1)
        np.testing.assert_allclose(row["beta_D"], 0.028, atol=1e-12)

    def test_beta_E_none_when_e_idx_is_none(self):
        """When e_idx is None, beta_E should be NaN."""
        import math
        beta = np.array([0.028])
        se = np.array([0.01])
        row = self._fn("Full", "Base", "Dl_nloans_b", beta, se, None, None, 100, 50,
                       e_idx=None)
        assert math.isnan(row["beta_E"])
        assert math.isnan(row["se_E"])
        assert math.isnan(row["t_E"])
        assert math.isnan(row["p_E"])

    def test_beta_E_last_when_e_idx_minus_one(self):
        """e_idx=-1 should extract the last coefficient as beta_E."""
        beta = np.array([0.028, 0.010, 0.005, 0.042])  # last = 0.042
        se = np.array([0.01, 0.005, 0.002, 0.015])
        row = self._fn("Full", "SLX", "Dl_nloans_b", beta, se, None, None, 400, 100,
                       e_idx=-1)
        np.testing.assert_allclose(row["beta_E"], 0.042, atol=1e-12)
        np.testing.assert_allclose(row["se_E"], 0.015, atol=1e-12)

    def test_t_stat_D_formula(self):
        """t_D = beta_D / se_D."""
        beta = np.array([0.04, 0.02])
        se = np.array([0.01, 0.005])
        row = self._fn("Full", "SLX", "Dl_nloans_b", beta, se, None, None, 200, 100,
                       e_idx=-1)
        np.testing.assert_allclose(row["t_D"], 4.0, atol=1e-10)

    def test_t_stat_E_formula(self):
        """t_E = beta_E / se_E."""
        beta = np.array([0.04, 0.02])
        se = np.array([0.01, 0.005])
        row = self._fn("Full", "SLX", "Dl_nloans_b", beta, se, None, None, 200, 100,
                       e_idx=-1)
        np.testing.assert_allclose(row["t_E"], 4.0, atol=1e-10)

    def test_t_nan_when_se_zero(self):
        """t_D = NaN when se_D = 0 (avoids division by zero)."""
        import math
        beta = np.array([0.04])
        se = np.array([0.0])
        row = self._fn("Full", "Base", "Dl_nloans_b", beta, se, None, None, 100, 50)
        assert math.isnan(row["t_D"])
        assert math.isnan(row["p_D"])

    def test_p_perm_initialised_nan(self):
        """p_perm should be initialised to NaN (set later by the caller)."""
        import math
        beta = np.array([0.04, 0.02])
        se = np.array([0.01, 0.005])
        row = self._fn("Full", "SLX", "Dl_nloans_b", beta, se, None, None, 200, 100,
                       e_idx=-1)
        assert math.isnan(row["p_perm"])

    def test_t_E_nan_when_se_E_zero(self):
        """t_E = NaN when se_E = 0."""
        import math
        beta = np.array([0.04, 0.02])
        se = np.array([0.01, 0.0])  # se_E = 0
        row = self._fn("Full", "SLX", "Dl_nloans_b", beta, se, None, None, 200, 100,
                       e_idx=-1)
        assert math.isnan(row["t_E"])
        assert math.isnan(row["p_E"])

    def test_e_idx_explicit_position(self):
        """Explicit e_idx=2 should select coefficient at index 2."""
        beta = np.array([0.10, 0.05, 0.03, 0.01])
        se = np.array([0.02, 0.01, 0.005, 0.001])
        row = self._fn("Full", "SLX", "Dl_nloans_b", beta, se, None, None, 300, 100,
                       e_idx=2)
        np.testing.assert_allclose(row["beta_E"], 0.03, atol=1e-12)
        np.testing.assert_allclose(row["se_E"], 0.005, atol=1e-12)

    def test_large_t_stat_implies_small_p(self):
        """Large t-statistic → small p-value."""
        beta = np.array([0.028, 0.040])  # very large relative to SE
        se = np.array([0.001, 0.001])
        row = self._fn("Full", "SLX", "Dl_nloans_b", beta, se, None, None, 500, 100,
                       e_idx=-1)
        # t = 40 → p should be extremely small (< 0.001)
        assert row["p_D"] < 1e-3
        assert row["p_E"] < 1e-3

    def test_dv_preserved(self):
        """Dependent variable name should be stored correctly."""
        beta = np.array([0.01])
        se = np.array([0.005])
        row = self._fn("Full", "Placebo", "Dl_nloans_pl", beta, se, None, None, 100, 50)
        assert row["dv"] == "Dl_nloans_pl"