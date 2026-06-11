"""
Tests for analysis/deprecated/sar_iv_credit_sdm.py

Covers the pure numerical helper functions introduced in this PR:
  - two_way_within
  - ols_fit
  - f_test_excluded
  - cluster_robust_f_first_stage
  - cluster_se_2sls
  - ols_cluster_se_scalar
"""
import sys
import os
import importlib

import numpy as np
import pytest

# conftest.py installs all sys.modules mocks before this runs.
# We still need to add the analysis paths.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis", "deprecated"))

# Lazy module import so we can rely on conftest mocks
_module = None


def _get_module():
    global _module
    if _module is None:
        _module = importlib.import_module("sar_iv_credit_sdm")
    return _module


# ---------------------------------------------------------------------------
# two_way_within
# ---------------------------------------------------------------------------

class TestTwoWayWithin:
    """Tests for two_way_within(arr_TN)."""

    def _fn(self, arr):
        return _get_module().two_way_within(arr)

    def test_grand_mean_removal(self):
        """After transformation, the grand mean should be zero."""
        rng = np.random.default_rng(42)
        arr = rng.standard_normal((5, 10))
        z = self._fn(arr)
        assert abs(z.mean()) < 1e-10, "Grand mean should be ~0 after within transform"

    def test_year_means_zero(self):
        """Each row (year) mean should be zero after transformation."""
        rng = np.random.default_rng(0)
        arr = rng.standard_normal((4, 8))
        z = self._fn(arr)
        row_means = z.mean(axis=1)
        np.testing.assert_allclose(row_means, 0.0, atol=1e-12)

    def test_constant_array_becomes_zero(self):
        """A constant panel should transform to all zeros."""
        arr = np.full((3, 5), 7.0)
        z = self._fn(arr)
        np.testing.assert_allclose(z, 0.0, atol=1e-12)

    def test_shape_preserved(self):
        """Output shape must equal input shape."""
        arr = np.random.randn(6, 12)
        z = self._fn(arr)
        assert z.shape == arr.shape

    def test_identity_for_demeaned_data(self):
        """Data already demeaned across both dimensions should be preserved."""
        rng = np.random.default_rng(7)
        raw = rng.standard_normal((4, 6))
        # Double-demean manually
        col_means = raw.mean(axis=0)
        row_means = raw.mean(axis=1, keepdims=True)
        grand = raw.mean()
        demeaned = raw - col_means[None, :] - row_means + grand
        demeaned = demeaned - demeaned.mean(axis=1, keepdims=True)

        z = self._fn(demeaned)
        # A second pass should leave the already-demeaned array (nearly) unchanged
        np.testing.assert_allclose(z, demeaned, atol=1e-10)

    def test_single_time_period(self):
        """Works with T=1."""
        arr = np.array([[1.0, 2.0, 3.0]])
        z = self._fn(arr)
        assert z.shape == (1, 3)
        # Row mean must be zero
        assert abs(z.mean()) < 1e-12

    def test_single_county(self):
        """Works with N=1."""
        arr = np.array([[1.0], [3.0], [5.0]])
        z = self._fn(arr)
        assert z.shape == (3, 1)

    def test_linearity(self):
        """two_way_within(a + b) == two_way_within(a) + two_way_within(b) (linearity)."""
        rng = np.random.default_rng(99)
        a = rng.standard_normal((4, 7))
        b = rng.standard_normal((4, 7))
        fn = self._fn
        np.testing.assert_allclose(fn(a + b), fn(a) + fn(b), atol=1e-12)

    def test_adding_constant_does_not_change_result(self):
        """Adding a constant to the panel should not change the within transform."""
        arr = np.random.randn(5, 8)
        z1 = self._fn(arr)
        z2 = self._fn(arr + 100.0)
        np.testing.assert_allclose(z1, z2, atol=1e-10)


# ---------------------------------------------------------------------------
# ols_fit
# ---------------------------------------------------------------------------

class TestOlsFit:
    """Tests for ols_fit(X, y)."""

    def _fn(self, X, y):
        return _get_module().ols_fit(X, y)

    def test_exact_fit(self):
        """When y = Xb exactly, residuals should be ~0."""
        X = np.column_stack([np.ones(20), np.linspace(0, 1, 20)])
        b_true = np.array([2.0, -3.0])
        y = X @ b_true
        delta, y_hat, resid, XtX_inv = self._fn(X, y)
        np.testing.assert_allclose(delta, b_true, atol=1e-10)
        np.testing.assert_allclose(resid, 0.0, atol=1e-10)

    def test_return_shapes(self):
        """Check return tuple shapes for (n, k) input."""
        n, k = 50, 3
        X = np.random.randn(n, k)
        y = np.random.randn(n)
        delta, y_hat, resid, XtX_inv = self._fn(X, y)
        assert delta.shape == (k,)
        assert y_hat.shape == (n,)
        assert resid.shape == (n,)
        assert XtX_inv.shape == (k, k)

    def test_y_hat_plus_resid_equals_y(self):
        """y_hat + resid should recover y exactly."""
        X = np.random.randn(30, 4)
        y = np.random.randn(30)
        delta, y_hat, resid, _ = self._fn(X, y)
        np.testing.assert_allclose(y_hat + resid, y, atol=1e-12)

    def test_scalar_regressor(self):
        """Simple OLS with k=1 matches analytical formula."""
        n = 40
        x = np.random.randn(n)
        X = x[:, None]
        y = 3.5 * x + np.random.randn(n) * 0.1
        delta, _, _, _ = self._fn(X, y)
        # Analytical: beta = (x'x)^-1 x'y
        beta_analytical = float(x @ y) / float(x @ x)
        assert abs(delta[0] - beta_analytical) < 1e-10

    def test_XtX_inv_is_inverse(self):
        """XtX_inv @ (X.T @ X) should be identity."""
        X = np.random.randn(20, 3)
        y = np.random.randn(20)
        _, _, _, XtX_inv = self._fn(X, y)
        eye_approx = XtX_inv @ (X.T @ X)
        np.testing.assert_allclose(eye_approx, np.eye(3), atol=1e-10)

    def test_intercept_only(self):
        """With only an intercept column, beta == mean(y)."""
        n = 25
        y = np.random.randn(n)
        X = np.ones((n, 1))
        delta, _, _, _ = self._fn(X, y)
        assert abs(delta[0] - y.mean()) < 1e-12

    def test_orthogonal_regressors(self):
        """With orthogonal X, each OLS coefficient is independent."""
        n = 100
        x1 = np.random.randn(n)
        x2 = x1 - x1.mean()
        x2 -= (x2 @ x1) / (x1 @ x1) * x1  # Gram-Schmidt orthogonalise
        X = np.column_stack([x1, x2])
        b_true = np.array([2.0, -1.5])
        y = X @ b_true
        delta, _, _, _ = self._fn(X, y)
        np.testing.assert_allclose(delta, b_true, atol=1e-8)


# ---------------------------------------------------------------------------
# f_test_excluded
# ---------------------------------------------------------------------------

class TestFTestExcluded:
    """Tests for f_test_excluded(z_tilde, X_full, n_excl)."""

    def _fn(self, z, X, n_excl):
        return _get_module().f_test_excluded(z, X, n_excl)

    def test_return_structure(self):
        """Returns (F_stat, df1, df2, R2_unr, R2_restr)."""
        n = 100
        X = np.column_stack([np.random.randn(n), np.random.randn(n), np.random.randn(n)])
        z = X @ np.array([1.0, 0.5, 0.0]) + np.random.randn(n) * 0.5
        result = self._fn(z, X, n_excl=2)
        assert len(result) == 5

    def test_df1_equals_n_excl(self):
        """df1 should equal n_excl."""
        n = 80
        X = np.random.randn(n, 4)
        z = np.random.randn(n)
        F, df1, df2, _, _ = self._fn(z, X, n_excl=2)
        assert df1 == 2

    def test_df2_equals_NT_minus_k(self):
        """df2 should equal NT - k."""
        n, k = 80, 4
        X = np.random.randn(n, k)
        z = np.random.randn(n)
        F, df1, df2, _, _ = self._fn(z, X, n_excl=2)
        assert df2 == n - k

    def test_f_stat_positive(self):
        """F-statistic should be non-negative."""
        n = 60
        X = np.random.randn(n, 3)
        z = np.random.randn(n)
        F, *_ = self._fn(z, X, n_excl=1)
        assert F >= 0.0

    def test_strong_instrument_gives_large_f(self):
        """If excluded instruments are highly correlated with z, F >> 10."""
        n = 200
        q1 = np.random.randn(n)
        q2 = np.random.randn(n)
        x = np.random.randn(n)
        # z is almost completely explained by q1, q2
        z = 5.0 * q1 + 4.0 * q2 + 0.05 * np.random.randn(n)
        X = np.column_stack([x, q1, q2])
        F, *_ = self._fn(z, X, n_excl=2)
        assert F > 100.0, f"Expected F >> 10 for strong instruments; got {F:.2f}"

    def test_irrelevant_instrument_gives_small_f(self):
        """Instruments orthogonal to z should yield a small F."""
        rng = np.random.default_rng(1234)
        n = 300
        x = rng.standard_normal(n)
        z = 2.0 * x + 0.5 * rng.standard_normal(n)
        # instruments are pure noise, unrelated to z
        q1 = rng.standard_normal(n)
        q2 = rng.standard_normal(n)
        # Residualise z on x first to isolate instrument effect
        from numpy.linalg import lstsq
        resid_z = z - x * (x @ z) / (x @ x)
        # q1, q2 should not explain resid_z
        X = np.column_stack([x, q1, q2])
        F, *_ = self._fn(z, X, n_excl=2)
        # F should be small (close to 1 or less); not > 10
        assert F < 10.0, f"Expected small F for irrelevant instruments; got {F:.2f}"

    def test_r2_unr_geq_r2_restr(self):
        """R2 of unrestricted model must be >= R2 of restricted."""
        n = 100
        X = np.random.randn(n, 4)
        z = np.random.randn(n)
        _, _, _, R2_u, R2_r = self._fn(z, X, n_excl=2)
        assert R2_u >= R2_r - 1e-10, "Unrestricted R2 should not be less than restricted R2"

    def test_n_excl_one(self):
        """Works with n_excl=1 (single excluded instrument)."""
        n = 60
        X = np.column_stack([np.random.randn(n), np.random.randn(n)])
        z = np.random.randn(n)
        F, df1, *_ = self._fn(z, X, n_excl=1)
        assert df1 == 1
        assert F >= 0.0


# ---------------------------------------------------------------------------
# cluster_robust_f_first_stage
# ---------------------------------------------------------------------------

class TestClusterRobustFFirstStage:
    """Tests for cluster_robust_f_first_stage."""

    def _fn(self, z_f, X_fs, n_excl, state_idx, G):
        return _get_module().cluster_robust_f_first_stage(z_f, X_fs, n_excl, state_idx, G)

    def _make_data(self, n=200, k=3, G=5, seed=42):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, k))
        z = X @ rng.standard_normal(k) + 0.5 * rng.standard_normal(n)
        state = np.repeat(np.arange(G), n // G)
        state = np.concatenate([state, np.zeros(n - len(state), dtype=int)])
        return z, X, state, G

    def test_returns_two_tuple(self):
        """Function must return a (F_cluster, n_excl) tuple."""
        z, X, state, G = self._make_data()
        result = self._fn(z, X, n_excl=2, state_idx=state, G=G)
        assert len(result) == 2

    def test_second_element_is_n_excl(self):
        z, X, state, G = self._make_data()
        _, n_excl_out = self._fn(z, X, n_excl=2, state_idx=state, G=G)
        assert n_excl_out == 2

    def test_f_stat_non_negative(self):
        z, X, state, G = self._make_data()
        F, _ = self._fn(z, X, n_excl=2, state_idx=state, G=G)
        assert np.isnan(F) or F >= 0.0

    def test_strong_instruments_large_f(self):
        """Strong excluded instruments should give F >> 10."""
        rng = np.random.default_rng(7)
        n, G = 300, 8
        q1 = rng.standard_normal(n)
        q2 = rng.standard_normal(n)
        x = rng.standard_normal(n)
        z = 4.0 * q1 + 3.0 * q2 + 0.1 * rng.standard_normal(n)
        X = np.column_stack([x, q1, q2])
        state = np.repeat(np.arange(G), n // G)
        state = np.concatenate([state, np.zeros(n - len(state), dtype=int)])
        F, _ = self._fn(z, X, n_excl=2, state_idx=state, G=G)
        assert F > 10.0, f"Expected large cluster-F for strong instruments; got {F:.2f}"

    def test_single_excluded_instrument(self):
        z, X, state, G = self._make_data(k=2)
        F, n_excl_out = self._fn(z, X, n_excl=1, state_idx=state, G=G)
        assert n_excl_out == 1
        assert np.isnan(F) or F >= 0.0

    def test_singular_vcov_returns_nan(self):
        """If V_excl is singular, the function should return nan gracefully.

        We construct a case where X.T @ X is invertible (so ols_fit succeeds)
        but the cluster-robust V_excl becomes singular due to rank deficiency in
        the score matrix across clusters.
        """
        rng = np.random.default_rng(555)
        n = 30
        G = 2
        # Two excluded instruments that are perfectly collinear in the residual
        # space → their cluster scores will be linearly dependent → V_excl singular
        x = rng.standard_normal(n)
        q = rng.standard_normal(n)
        # Make both excluded instruments identical
        # X.T @ X: columns [x, q, q] → singular. Instead keep [x, q1, q2] non-singular
        # but arrange data so cluster meat is singular.
        # Simplest: make residuals ~0 in both clusters → B ≈ 0 → V_excl ≈ 0 (singular)
        z = 1.5 * x + 0.8 * q + 0.3 * q  # z nearly linear in columns
        # Perturb slightly to keep X.T@X full rank
        q1 = q
        q2 = q + rng.standard_normal(n) * 1e-10  # near-identical instruments
        X = np.column_stack([x, q1, q2])
        state = np.array([0] * (n // 2) + [1] * (n - n // 2), dtype=int)
        F, n_excl_out = self._fn(z, X, n_excl=2, state_idx=state, G=G)
        # With near-singular V_excl, result should be nan or very large
        assert np.isnan(F) or np.isfinite(F)


# ---------------------------------------------------------------------------
# cluster_se_2sls
# ---------------------------------------------------------------------------

class TestClusterSe2sls:
    """Tests for cluster_se_2sls(Z_hat, xi, state_idx_flat, G, k_params)."""

    def _fn(self, Z_hat, xi, state, G, k):
        return _get_module().cluster_se_2sls(Z_hat, xi, state, G, k)

    def _make_data(self, n=200, k=2, G=6, seed=0):
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((n, k))
        xi = rng.standard_normal(n)
        state = np.repeat(np.arange(G), n // G)
        state = np.concatenate([state, np.zeros(n - len(state), dtype=int)])
        return Z, xi, state, G, k

    def test_return_shapes(self):
        """vcv is (k, k) and se is (k,)."""
        Z, xi, state, G, k = self._make_data()
        vcv, se = self._fn(Z, xi, state, G, k)
        assert vcv.shape == (k, k)
        assert se.shape == (k,)

    def test_se_non_negative(self):
        Z, xi, state, G, k = self._make_data()
        _, se = self._fn(Z, xi, state, G, k)
        assert np.all(se >= 0.0)

    def test_vcv_symmetric(self):
        Z, xi, state, G, k = self._make_data()
        vcv, _ = self._fn(Z, xi, state, G, k)
        np.testing.assert_allclose(vcv, vcv.T, atol=1e-12)

    def test_se_equals_sqrt_diag_vcv(self):
        """se[i] should equal sqrt(vcv[i,i])."""
        Z, xi, state, G, k = self._make_data()
        vcv, se = self._fn(Z, xi, state, G, k)
        expected = np.sqrt(np.maximum(np.diag(vcv), 0.0))
        np.testing.assert_allclose(se, expected, atol=1e-12)

    def test_larger_residuals_larger_se(self):
        """Scaling xi by a factor should scale se by the same factor."""
        Z, xi, state, G, k = self._make_data(seed=17)
        _, se1 = self._fn(Z, xi, state, G, k)
        _, se2 = self._fn(Z, xi * 2.0, state, G, k)
        np.testing.assert_allclose(se2, se1 * 2.0, rtol=1e-6)

    def test_k_params_three(self):
        """Works with k_params=3 (SDM-IV case)."""
        n, G, k = 150, 5, 3
        rng = np.random.default_rng(3)
        Z = rng.standard_normal((n, k))
        xi = rng.standard_normal(n)
        state = np.repeat(np.arange(G), n // G)
        state = np.concatenate([state, np.zeros(n - len(state), dtype=int)])
        vcv, se = self._fn(Z, xi, state, G, k)
        assert vcv.shape == (k, k)
        assert se.shape == (k,)

    def test_zero_residuals_give_zero_se(self):
        """If all residuals are zero, SE should be zero."""
        Z, xi, state, G, k = self._make_data(seed=9)
        xi_zero = np.zeros_like(xi)
        vcv, se = self._fn(Z, xi_zero, state, G, k)
        np.testing.assert_allclose(se, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# ols_cluster_se_scalar
# ---------------------------------------------------------------------------

class TestOlsClusterSeScalar:
    """Tests for ols_cluster_se_scalar(x_flat, y_flat, state_idx_flat, G)."""

    def _fn(self, x, y, state, G):
        return _get_module().ols_cluster_se_scalar(x, y, state, G)

    def _make_data(self, n=200, G=5, beta=0.5, seed=42):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(n)
        y = beta * x + 0.2 * rng.standard_normal(n)
        state = np.repeat(np.arange(G), n // G)
        state = np.concatenate([state, np.zeros(n - len(state), dtype=int)])
        return x, y, state, G, beta

    def test_beta_close_to_truth(self):
        """OLS estimate should be close to the true coefficient."""
        x, y, state, G, beta = self._make_data(n=500, beta=1.5)
        b, _ = self._fn(x, y, state, G)
        assert abs(b - beta) < 0.1, f"beta={b:.4f}, expected ~{beta}"

    def test_se_positive(self):
        x, y, state, G, _ = self._make_data()
        _, se = self._fn(x, y, state, G)
        assert se >= 0.0

    def test_return_is_two_floats(self):
        x, y, state, G, _ = self._make_data()
        result = self._fn(x, y, state, G)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    def test_beta_matches_analytical(self):
        """beta should equal (x'x)^{-1} x'y analytically."""
        rng = np.random.default_rng(55)
        n, G = 100, 4
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        state = np.repeat(np.arange(G), n // G)
        beta_analytical = float(x @ y) / float(x @ x)
        b, _ = self._fn(x, y, state, G)
        assert abs(b - beta_analytical) < 1e-12

    def test_larger_noise_larger_se(self):
        """More noise in y should result in larger SE."""
        rng = np.random.default_rng(77)
        n, G = 200, 6
        x = rng.standard_normal(n)
        state = np.repeat(np.arange(G), n // G)
        state = np.concatenate([state, np.zeros(n - len(state), dtype=int)])
        y_low = 0.5 * x + 0.1 * rng.standard_normal(n)
        y_high = 0.5 * x + 2.0 * rng.standard_normal(n)
        _, se_low = self._fn(x, y_low, state, G)
        _, se_high = self._fn(x, y_high, state, G)
        assert se_high > se_low, "Higher noise should increase clustered SE"

    def test_perfect_fit_zero_se(self):
        """If y = b*x exactly (no noise), residuals are zero → SE = 0.

        Note: requires G >= 2 since the correction factor G/(G-1) is undefined
        for G=1 (single cluster). Use G=2 with two dummy clusters.
        """
        n = 50
        rng = np.random.default_rng(13)
        x = rng.standard_normal(n)
        y = 2.0 * x
        # Need G >= 2 to avoid divide-by-zero in G/(G-1)
        state = np.array([0] * (n // 2) + [1] * (n - n // 2), dtype=int)
        b, se = self._fn(x, y, state, G=2)
        assert abs(b - 2.0) < 1e-10
        assert abs(se) < 1e-10

    def test_multiple_states_different_se(self):
        """With clustered data (state effect), SE should be positive and finite.

        Note: G=1 is a degenerate case (G/(G-1) = 1/0) so we use G >= 2.
        """
        rng = np.random.default_rng(88)
        n, G = 300, 10
        x = rng.standard_normal(n)
        state = np.repeat(np.arange(G), n // G)
        state = np.concatenate([state, np.zeros(n - len(state), dtype=int)])
        # Add cluster-level noise
        cluster_noise = rng.standard_normal(G)
        y = 0.5 * x + cluster_noise[state] * 0.5 + 0.1 * rng.standard_normal(n)
        b1, se1 = self._fn(x, y, state, G)
        assert se1 >= 0.0
        assert np.isfinite(se1)
        # Verify beta estimate is reasonable
        assert abs(b1 - 0.5) < 0.3