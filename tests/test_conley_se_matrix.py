"""
Tests for analysis/inference/conley_se.py (changes introduced in this PR)

Functions under test (all new or refactored in the PR):
  - ols_matrix
  - meat_spatial
  - meat_cluster_state
  - meat_twoway_overlap
  - meat_twoway
  - sandwich_se
  - df_cluster
  - df_conley
"""
import sys
import os
import importlib

import numpy as np
import pytest

# conftest.py pre-installs mocks for scipy, pandas, libpysal etc.
# conley_se.py additionally imports utils, panel_data, and w_variants
# (all mocked by conftest), as well as scipy.stats.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis", "inference"))

_module = None


def _get_module():
    global _module
    if _module is None:
        _module = importlib.import_module("conley_se")
    return _module


# ---------------------------------------------------------------------------
# Minimal sparse stub for W matrices
# ---------------------------------------------------------------------------

from conftest import FakeSparse


def _eye_sparse(n):
    return FakeSparse(np.eye(n))


def _diag_sparse(vals):
    return FakeSparse(np.diag(vals))


# ---------------------------------------------------------------------------
# ols_matrix
# ---------------------------------------------------------------------------

class TestOlsMatrix:
    """Tests for ols_matrix(y_flat, X_flat)."""

    def _fn(self, y, X):
        return _get_module().ols_matrix(y, X)

    def test_scalar_regressor_matches_formula(self):
        """With a single regressor, beta = x'y / x'x."""
        rng = np.random.default_rng(0)
        n = 50
        x = rng.standard_normal(n)
        beta_true = 2.5
        y = beta_true * x + 0.1 * rng.standard_normal(n)
        X = x[:, None]
        beta, u, XtX_inv = self._fn(y, X)
        np.testing.assert_allclose(beta[0], beta_true, atol=0.15)

    def test_residuals_orthogonal_to_X(self):
        """Residuals should be orthogonal to X (OLS first-order condition)."""
        rng = np.random.default_rng(1)
        n, k = 80, 3
        X = rng.standard_normal((n, k))
        beta_true = np.array([1.0, -0.5, 2.0])
        y = X @ beta_true + 0.2 * rng.standard_normal(n)
        _, u, _ = self._fn(y, X)
        residual_dot = X.T @ u
        np.testing.assert_allclose(residual_dot, 0.0, atol=1e-10)

    def test_beta_length(self):
        """beta vector should have same length as columns in X."""
        rng = np.random.default_rng(2)
        n, k = 60, 4
        X = rng.standard_normal((n, k))
        y = rng.standard_normal(n)
        beta, _, _ = self._fn(y, X)
        assert len(beta) == k

    def test_XtX_inv_is_inverse(self):
        """XtX_inv should satisfy X.T @ X @ XtX_inv ≈ I."""
        rng = np.random.default_rng(3)
        n, k = 40, 2
        X = rng.standard_normal((n, k))
        y = rng.standard_normal(n)
        _, _, XtX_inv = self._fn(y, X)
        XtX = X.T @ X
        product = XtX @ XtX_inv
        np.testing.assert_allclose(product, np.eye(k), atol=1e-10)

    def test_returns_triple(self):
        """Function should return (beta, u, XtX_inv)."""
        rng = np.random.default_rng(4)
        n, k = 30, 2
        X = rng.standard_normal((n, k))
        y = rng.standard_normal(n)
        result = self._fn(y, X)
        assert len(result) == 3

    def test_perfect_fit_zero_residuals(self):
        """When y = X @ beta exactly, residuals should be near zero."""
        rng = np.random.default_rng(5)
        n, k = 20, 2
        X = rng.standard_normal((n, k))
        beta_true = np.array([3.0, -1.0])
        y = X @ beta_true
        beta_hat, u, _ = self._fn(y, X)
        np.testing.assert_allclose(u, 0.0, atol=1e-10)
        np.testing.assert_allclose(beta_hat, beta_true, atol=1e-10)

    def test_multicolumn_regression(self):
        """Multi-column X with known coefficients should recover them."""
        rng = np.random.default_rng(6)
        n, k = 200, 5
        X = rng.standard_normal((n, k))
        beta_true = np.arange(1.0, k + 1)
        y = X @ beta_true + 0.05 * rng.standard_normal(n)
        beta_hat, _, _ = self._fn(y, X)
        np.testing.assert_allclose(beta_hat, beta_true, atol=0.1)


# ---------------------------------------------------------------------------
# meat_spatial
# ---------------------------------------------------------------------------

class TestMeatSpatial:
    """Tests for meat_spatial(u_TN, X_TNK, W_sp)."""

    def _fn(self, u_TN, X_TNK, W_sp):
        return _get_module().meat_spatial(u_TN, X_TNK, W_sp)

    def test_output_shape_k1(self):
        """With K=1, output should be (1, 1)."""
        T, N, K = 3, 4, 1
        u = np.ones((T, N))
        X = np.ones((T, N, K))
        W = FakeSparse(np.eye(N))
        B = self._fn(u, X, W)
        assert B.shape == (1, 1)

    def test_output_shape_k3(self):
        """With K=3, output should be (3, 3)."""
        T, N, K = 3, 4, 3
        rng = np.random.default_rng(10)
        u = rng.standard_normal((T, N))
        X = rng.standard_normal((T, N, K))
        W = FakeSparse(np.eye(N))
        B = self._fn(u, X, W)
        assert B.shape == (K, K)

    def test_symmetric(self):
        """Meat matrix should be symmetric (PSD)."""
        T, N, K = 5, 6, 2
        rng = np.random.default_rng(11)
        u = rng.standard_normal((T, N))
        X = rng.standard_normal((T, N, K))
        arr = rng.uniform(0, 1, (N, N))
        arr = arr + arr.T  # symmetric
        np.fill_diagonal(arr, 0)
        W = FakeSparse(arr)
        B = self._fn(u, X, W)
        np.testing.assert_allclose(B, B.T, atol=1e-10)

    def test_zero_residuals_gives_zero_meat(self):
        """Zero residuals → zero meat."""
        T, N, K = 4, 5, 2
        u = np.zeros((T, N))
        X = np.ones((T, N, K))
        W = FakeSparse(np.eye(N))
        B = self._fn(u, X, W)
        np.testing.assert_allclose(B, 0.0, atol=1e-14)

    def test_identity_W_equals_diagonal_formula(self):
        """With identity W, B = sum_t (S_t.T @ S_t) = sum_t diag of outer products."""
        T, N, K = 2, 3, 1
        rng = np.random.default_rng(12)
        u = rng.standard_normal((T, N))
        x = rng.standard_normal((T, N))
        X = x[:, :, None]
        W = FakeSparse(np.eye(N))
        B = self._fn(u, X, W)
        # Manual: B[0,0] = sum_t sum_n u[t,n]^2 * x[t,n]^2
        expected = float(((u * x) ** 2).sum())
        np.testing.assert_allclose(B[0, 0], expected, atol=1e-10)

    def test_zero_W_gives_zero_meat(self):
        """Zero weight matrix → zero meat regardless of residuals."""
        T, N, K = 3, 4, 2
        rng = np.random.default_rng(13)
        u = rng.standard_normal((T, N))
        X = rng.standard_normal((T, N, K))
        W = FakeSparse(np.zeros((N, N)))
        B = self._fn(u, X, W)
        np.testing.assert_allclose(B, 0.0, atol=1e-14)

    def test_k1_matches_scalar_formula(self):
        """K=1 result should match the old scalar formula B = sum_t v_t' W v_t."""
        T, N = 3, 4
        rng = np.random.default_rng(14)
        u = rng.standard_normal((T, N))
        x = rng.standard_normal((T, N))
        arr = rng.uniform(0, 1, (N, N))
        np.fill_diagonal(arr, 0)
        W = FakeSparse(arr)
        X = x[:, :, None]  # (T, N, 1)
        B = self._fn(u, X, W)
        # Scalar formula
        B_scalar = 0.0
        for t in range(T):
            v_t = u[t] * x[t]
            B_scalar += float(v_t @ arr @ v_t)
        np.testing.assert_allclose(B[0, 0], B_scalar, atol=1e-10)


# ---------------------------------------------------------------------------
# meat_cluster_state
# ---------------------------------------------------------------------------

class TestMeatClusterState:
    """Tests for meat_cluster_state(u_TN, X_TNK, county_states)."""

    def _fn(self, u_TN, X_TNK, county_states):
        return _get_module().meat_cluster_state(u_TN, X_TNK, county_states)

    def test_output_shape_k1(self):
        T, N = 4, 6
        u = np.ones((T, N))
        X = np.ones((T, N, 1))
        states = np.array([0, 0, 1, 1, 2, 2])
        B, G = self._fn(u, X, states)
        assert B.shape == (1, 1)

    def test_output_shape_k3(self):
        T, N, K = 4, 6, 3
        rng = np.random.default_rng(20)
        u = rng.standard_normal((T, N))
        X = rng.standard_normal((T, N, K))
        states = np.array([0, 0, 1, 1, 2, 2])
        B, G = self._fn(u, X, states)
        assert B.shape == (K, K)

    def test_G_equals_number_of_states(self):
        T, N = 3, 6
        u = np.ones((T, N))
        X = np.ones((T, N, 1))
        states = np.array([1, 1, 2, 2, 3, 3])
        _, G = self._fn(u, X, states)
        assert G == 3

    def test_single_state_equals_total_score_squared(self):
        """One state: meat = (sum of all scores)^2."""
        T, N, K = 3, 4, 1
        rng = np.random.default_rng(21)
        u = rng.standard_normal((T, N))
        x = rng.standard_normal((T, N))
        X = x[:, :, None]
        states = np.zeros(N, dtype=int)
        B, G = self._fn(u, X, states)
        total_score = float((u * x).sum())
        np.testing.assert_allclose(B[0, 0], total_score ** 2, atol=1e-10)
        assert G == 1

    def test_symmetric_output(self):
        T, N, K = 5, 8, 2
        rng = np.random.default_rng(22)
        u = rng.standard_normal((T, N))
        X = rng.standard_normal((T, N, K))
        states = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        B, _ = self._fn(u, X, states)
        np.testing.assert_allclose(B, B.T, atol=1e-10)

    def test_zero_residuals_gives_zero_meat(self):
        T, N, K = 4, 6, 2
        u = np.zeros((T, N))
        X = np.ones((T, N, K))
        states = np.array([0, 0, 1, 1, 2, 2])
        B, _ = self._fn(u, X, states)
        np.testing.assert_allclose(B, 0.0, atol=1e-14)

    def test_all_singleton_states(self):
        """With N singleton clusters, meat = sum of per-county score^2 outerproducts."""
        T, N, K = 3, 4, 1
        rng = np.random.default_rng(23)
        u = rng.standard_normal((T, N))
        x = rng.standard_normal((T, N))
        X = x[:, :, None]
        states = np.arange(N)
        B, G = self._fn(u, X, states)
        assert G == N
        county_scores = (u * x).sum(axis=0)
        expected = float((county_scores ** 2).sum())
        np.testing.assert_allclose(B[0, 0], expected, atol=1e-10)


# ---------------------------------------------------------------------------
# meat_twoway_overlap
# ---------------------------------------------------------------------------

class TestMeatTowayOverlap:
    """Tests for meat_twoway_overlap(u_TN, X_TNK, W_sp, county_states)."""

    def _fn(self, u_TN, X_TNK, W_sp, county_states):
        return _get_module().meat_twoway_overlap(u_TN, X_TNK, W_sp, county_states)

    def test_output_shape(self):
        T, N, K = 3, 4, 2
        rng = np.random.default_rng(30)
        u = rng.standard_normal((T, N))
        X = rng.standard_normal((T, N, K))
        W = FakeSparse(np.eye(N))
        states = np.array([0, 0, 1, 1])
        B = self._fn(u, X, W, states)
        assert B.shape == (K, K)

    def test_no_within_state_links_zero_overlap(self):
        """If W has no within-state links, overlap should be zero."""
        T, N, K = 3, 4, 1
        rng = np.random.default_rng(31)
        u = rng.standard_normal((T, N))
        X = rng.standard_normal((T, N, K))
        # Build W with only cross-state links: 0,1 in state A; 2,3 in state B
        # Links: 0-2, 0-3, 1-2, 1-3 (all cross-state)
        arr = np.zeros((N, N))
        arr[0, 2] = arr[2, 0] = 0.5
        arr[0, 3] = arr[3, 0] = 0.5
        arr[1, 2] = arr[2, 1] = 0.5
        arr[1, 3] = arr[3, 1] = 0.5
        W = FakeSparse(arr)
        states = np.array([0, 0, 1, 1])
        B = self._fn(u, X, W, states)
        np.testing.assert_allclose(B, 0.0, atol=1e-12)

    def test_only_within_state_links_equals_spatial_meat(self):
        """If W has ONLY within-state links, overlap equals full spatial meat."""
        T, N, K = 3, 4, 1
        rng = np.random.default_rng(32)
        u = rng.standard_normal((T, N))
        X = rng.standard_normal((T, N, K))
        # Build W with only within-state links: 0-1 (state A), 2-3 (state B)
        arr = np.zeros((N, N))
        arr[0, 1] = arr[1, 0] = 0.5
        arr[2, 3] = arr[3, 2] = 0.5
        W = FakeSparse(arr)
        states = np.array([0, 0, 1, 1])
        B_overlap = self._fn(u, X, W, states)
        B_spatial = _get_module().meat_spatial(u, X, W)
        np.testing.assert_allclose(B_overlap, B_spatial, atol=1e-10)

    def test_symmetric(self):
        T, N, K = 4, 6, 2
        rng = np.random.default_rng(33)
        u = rng.standard_normal((T, N))
        X = rng.standard_normal((T, N, K))
        arr = rng.uniform(0, 0.5, (N, N))
        arr = arr + arr.T
        np.fill_diagonal(arr, 0)
        W = FakeSparse(arr)
        states = np.array([0, 0, 0, 1, 1, 1])
        B = self._fn(u, X, W, states)
        np.testing.assert_allclose(B, B.T, atol=1e-10)

    def test_zero_residuals_gives_zero(self):
        T, N, K = 3, 4, 2
        u = np.zeros((T, N))
        X = np.ones((T, N, K))
        arr = np.ones((N, N)) - np.eye(N)
        W = FakeSparse(arr)
        states = np.array([0, 0, 1, 1])
        B = self._fn(u, X, W, states)
        np.testing.assert_allclose(B, 0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# meat_twoway
# ---------------------------------------------------------------------------

class TestMeatTwoway:
    """Tests for meat_twoway(B_state, B_spatial_bank, B_overlap)."""

    def _fn(self, B_state, B_spatial, B_overlap):
        return _get_module().meat_twoway(B_state, B_spatial, B_overlap)

    def test_additive_combination(self):
        """Result = B_state + B_spatial - B_overlap."""
        K = 2
        B_s = np.array([[4.0, 1.0], [1.0, 3.0]])
        B_p = np.array([[2.0, 0.5], [0.5, 1.5]])
        B_o = np.array([[1.0, 0.2], [0.2, 0.8]])
        result = self._fn(B_s, B_p, B_o)
        expected = B_s + B_p - B_o
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_zero_overlap_sums_state_and_spatial(self):
        K = 2
        B_s = np.eye(K) * 3.0
        B_p = np.eye(K) * 1.5
        B_o = np.zeros((K, K))
        result = self._fn(B_s, B_p, B_o)
        np.testing.assert_allclose(result, B_s + B_p, atol=1e-14)

    def test_full_overlap_equals_state_meat(self):
        """When overlap = spatial (all links within-state), result = B_state."""
        K = 2
        B_p = np.array([[2.0, 0.5], [0.5, 1.5]])
        B_s = np.array([[4.0, 1.0], [1.0, 3.0]])
        result = self._fn(B_s, B_p, B_p)   # B_overlap == B_spatial
        np.testing.assert_allclose(result, B_s, atol=1e-14)

    def test_scalar_case(self):
        """1×1 matrices (k=1 effectively)."""
        B_s = np.array([[5.0]])
        B_p = np.array([[2.0]])
        B_o = np.array([[1.0]])
        result = self._fn(B_s, B_p, B_o)
        np.testing.assert_allclose(result, np.array([[6.0]]), atol=1e-14)


# ---------------------------------------------------------------------------
# sandwich_se
# ---------------------------------------------------------------------------

class TestSandwichSe:
    """Tests for sandwich_se(XtX_inv, meat, df_corr, param_idx=0)."""

    def _fn(self, XtX_inv, meat, df_corr, param_idx=0):
        return _get_module().sandwich_se(XtX_inv, meat, df_corr, param_idx)

    def test_scalar_known_result(self):
        """With k=1: SE = sqrt(XtX_inv * meat * XtX_inv * df_corr)."""
        XtX_inv = np.array([[0.5]])
        meat = np.array([[8.0]])
        df_corr = 1.0
        expected = np.sqrt(0.5 * 8.0 * 0.5)
        result = self._fn(XtX_inv, meat, df_corr)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_df_corr_scales_se(self):
        """SE should scale with sqrt(df_corr)."""
        XtX_inv = np.eye(2)
        meat = np.array([[4.0, 0.0], [0.0, 9.0]])
        se1 = self._fn(XtX_inv, meat, 1.0, param_idx=0)
        se2 = self._fn(XtX_inv, meat, 4.0, param_idx=0)
        np.testing.assert_allclose(se2, 2.0 * se1, atol=1e-10)

    def test_param_idx_selects_correct_coefficient(self):
        """param_idx=1 should return SE for the second coefficient."""
        XtX_inv = np.eye(2)
        meat = np.array([[4.0, 0.0], [0.0, 9.0]])
        se0 = self._fn(XtX_inv, meat, 1.0, param_idx=0)
        se1 = self._fn(XtX_inv, meat, 1.0, param_idx=1)
        np.testing.assert_allclose(se0, 2.0, atol=1e-10)
        np.testing.assert_allclose(se1, 3.0, atol=1e-10)

    def test_returns_float(self):
        XtX_inv = np.eye(2)
        meat = np.eye(2)
        result = self._fn(XtX_inv, meat, 1.0)
        assert isinstance(result, float)

    def test_clamps_negative_variance_to_zero(self):
        """Negative (near-zero) diagonal variance should return 0.0 SE, not NaN."""
        XtX_inv = np.array([[1e-8]])
        meat = np.array([[-1e-20]])   # tiny negative variance due to numeric noise
        result = self._fn(XtX_inv, meat, 1.0)
        assert result >= 0.0
        assert np.isfinite(result)

    def test_zero_meat_gives_zero_se(self):
        XtX_inv = np.eye(2) * 10
        meat = np.zeros((2, 2))
        se = self._fn(XtX_inv, meat, 2.0)
        np.testing.assert_allclose(se, 0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# df_cluster and df_conley
# ---------------------------------------------------------------------------

class TestDfHelpers:
    """Tests for df_cluster and df_conley."""

    def test_df_cluster_formula(self):
        fn = _get_module().df_cluster
        assert fn(10) == pytest.approx(10 / 9)
        assert fn(50) == pytest.approx(50 / 49)
        assert fn(2) == pytest.approx(2.0)

    def test_df_conley_formula(self):
        fn = _get_module().df_conley
        NT, k = 1000, 1
        expected = NT / (NT - k - 1)
        result = fn(NT, k)
        assert result == pytest.approx(expected)

    def test_df_conley_default_k(self):
        fn = _get_module().df_conley
        NT = 500
        # Default k=1
        assert fn(NT) == pytest.approx(NT / (NT - 2))

    def test_df_cluster_larger_G_smaller_correction(self):
        fn = _get_module().df_cluster
        # As G grows, G/(G-1) → 1
        assert fn(100) < fn(10)

    def test_df_conley_larger_NT_smaller_correction(self):
        fn = _get_module().df_conley
        assert fn(10000, 1) < fn(100, 1)


# ---------------------------------------------------------------------------
# Regression: meat_twoway with real within-transformed data
# ---------------------------------------------------------------------------

class TestMeatTowayRegression:
    """Integration-style regression test using realistic within-transformed data."""

    def test_twoway_ge_state_when_cross_state_dominant(self):
        """
        When W_bank has mostly cross-state links, B_twoway > B_state
        because there is no double-counting to subtract.
        """
        mod = _get_module()
        rng = np.random.default_rng(99)
        T, N, K = 5, 8, 1

        u = rng.standard_normal((T, N))
        X = rng.standard_normal((T, N, K))
        # All links are cross-state (first 4 counties in state 0, last 4 in state 1)
        arr = np.zeros((N, N))
        for i in range(4):
            for j in range(4, 8):
                arr[i, j] = arr[j, i] = 0.25
        W = FakeSparse(arr)
        states = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        B_state, G = mod.meat_cluster_state(u, X, states)
        B_spatial = mod.meat_spatial(u, X, W)
        B_overlap = mod.meat_twoway_overlap(u, X, W, states)
        B_two = mod.meat_twoway(B_state, B_spatial, B_overlap)

        # With purely cross-state links, overlap should be ~0
        np.testing.assert_allclose(B_overlap, 0.0, atol=1e-12)
        # Two-way should equal state + spatial
        np.testing.assert_allclose(B_two, B_state + B_spatial, atol=1e-12)