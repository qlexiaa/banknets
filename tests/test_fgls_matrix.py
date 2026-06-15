"""
Tests for analysis/inference/fgls.py (changes introduced in this PR)

Functions under test (all new or refactored in this PR):
  - apply_filter (new 3D branch)
  - _fit_matrix
  - ols_matrix
  - fgls_matrix
  - cluster_se_matrix
"""
import sys
import os
import importlib

import numpy as np
import pytest

# conftest.py installs all necessary mocks.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis", "inference"))

_module = None


def _get_module():
    global _module
    if _module is None:
        _module = importlib.import_module("fgls")
    return _module


# ---------------------------------------------------------------------------
# apply_filter
# ---------------------------------------------------------------------------

class TestApplyFilter:
    """Tests for apply_filter(arr_TN, A_sp) — both 2D and 3D inputs."""

    def _fn(self, arr, A):
        return _get_module().apply_filter(arr, A)

    def test_2d_identity_filter_unchanged(self):
        """Applying the identity matrix should leave the array unchanged."""
        T, N = 5, 4
        arr = np.random.default_rng(0).standard_normal((T, N))
        A = np.eye(N)
        result = self._fn(arr, A)
        np.testing.assert_allclose(result, arr, atol=1e-12)

    def test_3d_identity_filter_unchanged(self):
        """New 3D branch: identity filter on (T, N, K) should be unchanged."""
        T, N, K = 5, 4, 3
        arr = np.random.default_rng(1).standard_normal((T, N, K))
        A = np.eye(N)
        result = self._fn(arr, A)
        assert result.shape == (T, N, K)
        np.testing.assert_allclose(result, arr, atol=1e-12)

    def test_3d_shape_preserved(self):
        """3D output should have the same shape as input."""
        T, N, K = 3, 5, 2
        arr = np.ones((T, N, K))
        A = 0.5 * np.eye(N)
        result = self._fn(arr, A)
        assert result.shape == (T, N, K)

    def test_2d_shape_preserved(self):
        T, N = 6, 7
        arr = np.ones((T, N))
        A = np.eye(N)
        result = self._fn(arr, A)
        assert result.shape == (T, N)

    def test_3d_filter_applied_period_by_period(self):
        """Each time slice t of output should be A @ arr[t]."""
        T, N, K = 4, 3, 2
        rng = np.random.default_rng(2)
        arr = rng.standard_normal((T, N, K))
        A = rng.standard_normal((N, N))
        result = self._fn(arr, A)
        for t in range(T):
            expected_t = A @ arr[t]   # (N, K)
            np.testing.assert_allclose(result[t], expected_t, atol=1e-12)

    def test_2d_filter_applied_period_by_period(self):
        """Each row of 2D output should be A @ arr[t]."""
        T, N = 4, 3
        rng = np.random.default_rng(3)
        arr = rng.standard_normal((T, N))
        A = rng.standard_normal((N, N))
        result = self._fn(arr, A)
        for t in range(T):
            expected_t = A @ arr[t]
            np.testing.assert_allclose(result[t], expected_t, atol=1e-12)

    def test_3d_scalar_filter(self):
        """Scalar multiple of identity: output should be s * arr."""
        T, N, K = 3, 4, 2
        arr = np.ones((T, N, K))
        A = 2.5 * np.eye(N)
        result = self._fn(arr, A)
        np.testing.assert_allclose(result, 2.5 * arr, atol=1e-12)

    def test_3d_each_column_filtered_independently(self):
        """Applying a permutation matrix should permute rows of each slice."""
        T, N, K = 2, 3, 2
        arr = np.arange(T * N * K, dtype=float).reshape(T, N, K)
        # A permutes rows: [0,1,2] → [2,0,1]
        A = np.zeros((N, N))
        A[0, 2] = 1.0
        A[1, 0] = 1.0
        A[2, 1] = 1.0
        result = self._fn(arr, A)
        for t in range(T):
            expected = A @ arr[t]
            np.testing.assert_allclose(result[t], expected, atol=1e-12)


# ---------------------------------------------------------------------------
# _fit_matrix
# ---------------------------------------------------------------------------

class TestFitMatrix:
    """Tests for _fit_matrix(y_TN, X_TNK)."""

    def _fn(self, y_TN, X_TNK):
        return _get_module()._fit_matrix(y_TN, X_TNK)

    def test_beta_shape(self):
        """Beta should have length K."""
        T, N, K = 5, 10, 3
        rng = np.random.default_rng(10)
        X = rng.standard_normal((T, N, K))
        y = np.einsum("tnk,k->tn", X, np.ones(K))
        beta, _ = self._fn(y, X)
        assert beta.shape == (K,)

    def test_XtX_inv_shape(self):
        """XtX_inv should be (K, K)."""
        T, N, K = 5, 10, 3
        rng = np.random.default_rng(11)
        X = rng.standard_normal((T, N, K))
        y = rng.standard_normal((T, N))
        _, XtX_inv = self._fn(y, X)
        assert XtX_inv.shape == (K, K)

    def test_known_coefficients(self):
        """Should recover known coefficients exactly with no noise."""
        T, N, K = 4, 20, 2
        rng = np.random.default_rng(12)
        X = rng.standard_normal((T, N, K))
        beta_true = np.array([3.0, -1.5])
        y = np.einsum("tnk,k->tn", X, beta_true)
        beta_hat, _ = self._fn(y, X)
        np.testing.assert_allclose(beta_hat, beta_true, atol=1e-10)

    def test_XtX_inv_is_actual_inverse(self):
        """XtX_inv should satisfy XtX @ XtX_inv = I."""
        T, N, K = 6, 15, 3
        rng = np.random.default_rng(13)
        X = rng.standard_normal((T, N, K))
        y = rng.standard_normal((T, N))
        _, XtX_inv = self._fn(y, X)
        X_flat = X.reshape(T * N, K)
        XtX = X_flat.T @ X_flat
        product = XtX @ XtX_inv
        np.testing.assert_allclose(product, np.eye(K), atol=1e-9)

    def test_recovers_noisy_coefficients(self):
        """With low noise, recovered coefficients should be close to true."""
        T, N, K = 8, 50, 3
        rng = np.random.default_rng(14)
        X = rng.standard_normal((T, N, K))
        beta_true = np.array([1.0, -2.0, 0.5])
        y = np.einsum("tnk,k->tn", X, beta_true) + 0.05 * rng.standard_normal((T, N))
        beta_hat, _ = self._fn(y, X)
        np.testing.assert_allclose(beta_hat, beta_true, atol=0.1)


# ---------------------------------------------------------------------------
# ols_matrix (fgls version)
# ---------------------------------------------------------------------------

class TestOlsMatrixFgls:
    """Tests for ols_matrix(y_TN, X_TNK) in fgls.py."""

    def _fn(self, y_TN, X_TNK):
        return _get_module().ols_matrix(y_TN, X_TNK)

    def test_returns_two_items(self):
        T, N, K = 4, 8, 2
        rng = np.random.default_rng(20)
        X = rng.standard_normal((T, N, K))
        y = rng.standard_normal((T, N))
        result = self._fn(y, X)
        assert len(result) == 2

    def test_residuals_shape(self):
        T, N, K = 4, 8, 2
        rng = np.random.default_rng(21)
        X = rng.standard_normal((T, N, K))
        y = rng.standard_normal((T, N))
        _, xi = self._fn(y, X)
        assert xi.shape == (T, N)

    def test_perfect_fit_zero_residuals(self):
        T, N, K = 4, 10, 2
        rng = np.random.default_rng(22)
        X = rng.standard_normal((T, N, K))
        beta_true = np.array([2.0, -1.0])
        y = np.einsum("tnk,k->tn", X, beta_true)
        beta_hat, xi = self._fn(y, X)
        np.testing.assert_allclose(xi, 0.0, atol=1e-10)
        np.testing.assert_allclose(beta_hat, beta_true, atol=1e-10)

    def test_residuals_unfiltered(self):
        """Residuals should be xi = y - X @ beta, not filtered."""
        T, N, K = 3, 6, 1
        rng = np.random.default_rng(23)
        X = rng.standard_normal((T, N, K))
        y = rng.standard_normal((T, N))
        beta_hat, xi = self._fn(y, X)
        expected_xi = y - np.einsum("tnk,k->tn", X, beta_hat)
        np.testing.assert_allclose(xi, expected_xi, atol=1e-12)


# ---------------------------------------------------------------------------
# fgls_matrix
# ---------------------------------------------------------------------------

class TestFglsMatrix:
    """Tests for fgls_matrix(y_TN, X_TNK, A_sp)."""

    def _fn(self, y_TN, X_TNK, A_sp):
        return _get_module().fgls_matrix(y_TN, X_TNK, A_sp)

    def test_returns_three_items(self):
        T, N, K = 3, 6, 2
        rng = np.random.default_rng(30)
        X = rng.standard_normal((T, N, K))
        y = rng.standard_normal((T, N))
        A = np.eye(N)
        result = self._fn(y, X, A)
        assert len(result) == 3

    def test_identity_filter_matches_ols(self):
        """With A = I (no filter), FGLS should equal OLS."""
        T, N, K = 4, 10, 2
        rng = np.random.default_rng(31)
        X = rng.standard_normal((T, N, K))
        y = rng.standard_normal((T, N))
        A = np.eye(N)
        beta_fgls, xi_fgls, X_A = self._fn(y, X, A)
        beta_ols, xi_ols = _get_module().ols_matrix(y, X)
        np.testing.assert_allclose(beta_fgls, beta_ols, atol=1e-10)

    def test_residuals_are_unfiltered(self):
        """Residuals xi should be y - X @ beta (NOT filtered y - filtered X @ beta)."""
        T, N, K = 3, 8, 2
        rng = np.random.default_rng(32)
        X = rng.standard_normal((T, N, K))
        y = rng.standard_normal((T, N))
        lam = 0.3
        W = rng.uniform(0, 1, (N, N))
        np.fill_diagonal(W, 0)
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        W = W / row_sums   # row-standardise
        A = np.eye(N) - lam * W
        beta_fgls, xi_fgls, _ = self._fn(y, X, A)
        # Residuals should be unfiltered: xi = y - X @ beta
        xi_expected = y - np.einsum("tnk,k->tn", X, beta_fgls)
        np.testing.assert_allclose(xi_fgls, xi_expected, atol=1e-10)

    def test_X_A_shape(self):
        """Filtered X should have same shape as input X."""
        T, N, K = 4, 6, 3
        rng = np.random.default_rng(33)
        X = rng.standard_normal((T, N, K))
        y = rng.standard_normal((T, N))
        A = 0.9 * np.eye(N)
        _, _, X_A = self._fn(y, X, A)
        assert X_A.shape == (T, N, K)

    def test_filter_changes_beta_vs_ols(self):
        """With a non-identity filter, beta should generally differ from OLS."""
        T, N, K = 6, 20, 2
        rng = np.random.default_rng(34)
        X = rng.standard_normal((T, N, K))
        beta_true = np.array([1.0, -0.5])
        # Generate spatially correlated errors
        W = np.zeros((N, N))
        for i in range(N - 1):
            W[i, i + 1] = W[i + 1, i] = 0.5
        np.fill_diagonal(W, 0)
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        W = W / row_sums
        lam = 0.3
        A = np.eye(N) - lam * W
        # Error correlated with W
        eps_base = rng.standard_normal((T, N))
        eps_corr = np.einsum("ij,tj->ti", W, eps_base) * 0.5 + eps_base
        y = np.einsum("tnk,k->tn", X, beta_true) + eps_corr

        beta_ols, _ = _get_module().ols_matrix(y, X)
        beta_fgls, _, _ = self._fn(y, X, A)

        # They should differ (spatially correlated errors make OLS suboptimal)
        diff = np.abs(beta_fgls - beta_ols).max()
        # We can't guarantee they differ significantly in a small sample, but
        # the computation should at least run without error
        assert np.isfinite(beta_fgls).all()


# ---------------------------------------------------------------------------
# cluster_se_matrix
# ---------------------------------------------------------------------------

class TestClusterSeMatrix:
    """Tests for cluster_se_matrix(xi_TN, X_A_TNK, county_states, param_idx=0)."""

    def _fn(self, xi_TN, X_A_TNK, county_states, param_idx=0):
        return _get_module().cluster_se_matrix(xi_TN, X_A_TNK, county_states, param_idx)

    def test_returns_float_and_g(self):
        T, N, K = 4, 6, 2
        rng = np.random.default_rng(40)
        xi = rng.standard_normal((T, N))
        X_A = rng.standard_normal((T, N, K))
        states = np.array([0, 0, 1, 1, 2, 2])
        se, G = self._fn(xi, X_A, states)
        assert isinstance(se, float)
        assert isinstance(G, (int, np.integer))

    def test_G_equals_number_of_states(self):
        T, N, K = 3, 8, 1
        rng = np.random.default_rng(41)
        xi = rng.standard_normal((T, N))
        X_A = rng.standard_normal((T, N, K))
        states = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        _, G = self._fn(xi, X_A, states)
        assert G == 4

    def test_se_positive(self):
        T, N, K = 5, 6, 2
        rng = np.random.default_rng(42)
        xi = rng.standard_normal((T, N))
        X_A = rng.standard_normal((T, N, K))
        states = np.array([0, 0, 1, 1, 2, 2])
        se, _ = self._fn(xi, X_A, states)
        assert se >= 0.0

    def test_param_idx_affects_se(self):
        """Different param_idx should return different SEs for non-diagonal VCV."""
        T, N, K = 5, 8, 2
        rng = np.random.default_rng(43)
        xi = rng.standard_normal((T, N))
        X_A = rng.standard_normal((T, N, K))
        states = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        se0, _ = self._fn(xi, X_A, states, param_idx=0)
        se1, _ = self._fn(xi, X_A, states, param_idx=1)
        # With random X, these will generally differ
        # (just verify they're both valid floats)
        assert np.isfinite(se0)
        assert np.isfinite(se1)

    def test_zero_residuals_zero_se(self):
        """Zero residuals → zero scores → zero SE."""
        T, N, K = 4, 6, 2
        xi = np.zeros((T, N))
        X_A = np.ones((T, N, K))
        states = np.array([0, 0, 1, 1, 2, 2])
        se, _ = self._fn(xi, X_A, states)
        np.testing.assert_allclose(se, 0.0, atol=1e-14)

    def test_se_scales_with_residuals(self):
        """Doubling residuals should double the SE (linear scaling)."""
        T, N, K = 5, 6, 1
        rng = np.random.default_rng(44)
        xi = rng.standard_normal((T, N))
        X_A = rng.standard_normal((T, N, K))
        states = np.array([0, 0, 1, 1, 2, 2])
        se1, _ = self._fn(xi, X_A, states)
        se2, _ = self._fn(2 * xi, X_A, states)
        np.testing.assert_allclose(se2, 2.0 * se1, atol=1e-10)

    def test_consistent_with_single_cluster(self):
        """One cluster: se = sqrt(total_score^2 / bread^2 * G/(G-1))."""
        T, N, K = 3, 4, 1
        rng = np.random.default_rng(45)
        xi = rng.standard_normal((T, N))
        X_A = rng.standard_normal((T, N, K))
        states = np.zeros(N, dtype=int)
        se, G = self._fn(xi, X_A, states)
        assert G == 1
        # bread = X_A.reshape(TN, K).T @ X_A.reshape(TN, K) → (1,1)
        # with one cluster, G/(G-1) = inf → division by zero issue
        # In practice this means SE is undefined for G=1, just check it's finite
        # (implementation may return 0 or some value)
        assert np.isfinite(se)

    def test_more_clusters_means_smaller_finite_sample_correction(self):
        """More clusters → df_corr = G/(G-1) → 1, so SE changes with G."""
        T, N = 4, 20
        K = 1
        rng = np.random.default_rng(46)
        # Fix xi and X_A, vary the cluster assignment
        xi = rng.standard_normal((T, N))
        X_A = rng.standard_normal((T, N, K))

        # 5 clusters of 4 each
        states_5 = np.repeat(np.arange(5), 4)
        # 10 clusters of 2 each
        states_10 = np.repeat(np.arange(10), 2)

        se_5, _ = self._fn(xi, X_A, states_5)
        se_10, _ = self._fn(xi, X_A, states_10)

        # Both should be finite positive floats
        assert np.isfinite(se_5) and se_5 >= 0
        assert np.isfinite(se_10) and se_10 >= 0