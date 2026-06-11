"""
conftest.py
===========
Shared test fixtures and module-level mocking for the analysis package tests.

Many analysis modules depend on packages that are not available in the test
environment (scipy, matplotlib, pandas, libpysal, esda, pyreadstat,
statsmodels, geopandas, spreg).  This conftest pre-populates sys.modules with
MagicMock stubs so that the analysis modules can be imported and their pure
numerical functions can be exercised with numpy arrays.

A minimal FakeSparse class is also provided so that sparse-matrix logic in
_row_stats and _build_knn can be exercised without a real scipy installation.
"""
import sys
import numpy as np
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Minimal sparse-matrix stub
# ---------------------------------------------------------------------------

class FakeSparse:
    """
    Lightweight numpy-backed stub that mimics the subset of the
    scipy.sparse CSR API used by the analysis modules under test.
    """

    def __init__(self, data):
        arr = np.asarray(data, dtype=np.float64)
        self._data = arr
        self.shape = arr.shape

    # ---- construction helpers ----
    @classmethod
    def from_dense(cls, arr):
        return cls(arr)

    def tocsr(self):
        return self

    def toarray(self):
        return self._data.copy()

    def copy(self):
        return FakeSparse(self._data.copy())

    # ---- mutation ----
    def setdiag(self, val):
        np.fill_diagonal(self._data, val)

    def eliminate_zeros(self):
        pass  # no-op for dense stub

    # ---- properties ----
    @property
    def nnz(self):
        return int((self._data != 0).sum())

    def diagonal(self):
        return np.diag(self._data)

    # ---- arithmetic ----
    def sum(self, axis=None):
        result = self._data.sum(axis=axis)
        if axis is not None:
            # scipy.sparse.sum returns a matrix; mimic that shape
            return np.asmatrix(result.reshape(-1, 1) if axis == 1 else result.reshape(1, -1))
        return result

    def __gt__(self, val):
        result = self._data > val
        return _BoolSparse(result)

    def __getitem__(self, key):
        return FakeSparse(self._data[key])

    def __matmul__(self, other):
        if isinstance(other, FakeSparse):
            return FakeSparse(self._data @ other._data)
        return FakeSparse(self._data @ np.asarray(other))

    def __rmatmul__(self, other):
        return FakeSparse(np.asarray(other) @ self._data)


class _BoolSparse:
    """Boolean result of (FakeSparse > scalar)."""

    def __init__(self, mask):
        self._mask = mask

    def sum(self, axis=None):
        result = self._mask.sum(axis=axis)
        if axis is not None:
            return np.asmatrix(result.reshape(-1, 1) if axis == 1 else result.reshape(1, -1))
        return result


# ---------------------------------------------------------------------------
# Mock factory helpers
# ---------------------------------------------------------------------------

def _make_fake_scipy():
    """Return a MagicMock for scipy with a usable sparse sub-module."""
    scipy_mock = MagicMock()

    # sparse sub-module needs csr_matrix and diags
    def _fake_csr_matrix(arg, shape=None, dtype=None):
        if isinstance(arg, FakeSparse):
            return arg
        arr = np.asarray(arg, dtype=np.float64)
        if shape is not None:
            arr = arr.reshape(shape)
        return FakeSparse(arr)

    def _fake_diags(d, *args, **kwargs):
        d = np.asarray(d, dtype=np.float64).flatten()
        return FakeSparse(np.diag(d))

    sparse_mock = MagicMock()
    sparse_mock.csr_matrix.side_effect = _fake_csr_matrix
    sparse_mock.diags.side_effect = _fake_diags
    sparse_mock.load_npz = MagicMock(return_value=FakeSparse(np.eye(3)))

    scipy_mock.sparse = sparse_mock
    scipy_mock.stats = MagicMock()
    scipy_mock.linalg = MagicMock()

    # norm.ppf(0.975) ≈ 1.96
    scipy_mock.stats.norm.ppf = lambda p: 1.959963985  # close enough

    return scipy_mock


def _make_fake_libpysal():
    """Return a MagicMock for libpysal with a minimal W constructor."""
    libpysal_mock = MagicMock()

    class _FakeW:
        def __init__(self, neighbors, weights, silence_warnings=True):
            self.neighbors = neighbors
            self.weights = weights
            self.n = len(neighbors)
            self.transform = None
            self.mean_neighbors = (
                sum(len(v) for v in neighbors.values()) / len(neighbors)
                if neighbors else 0.0
            )

        def __setattr__(self, name, value):
            object.__setattr__(self, name, value)

    libpysal_mock.weights.W.side_effect = _FakeW
    return libpysal_mock


# ---------------------------------------------------------------------------
# Install mocks into sys.modules BEFORE any analysis import
# ---------------------------------------------------------------------------

def _install_mocks():
    scipy_mock = _make_fake_scipy()
    libpysal_mock = _make_fake_libpysal()

    mocks = {
        "scipy":                        scipy_mock,
        "scipy.sparse":                 scipy_mock.sparse,
        "scipy.stats":                  scipy_mock.stats,
        "scipy.linalg":                 scipy_mock.linalg,
        "matplotlib":                   MagicMock(),
        "matplotlib.pyplot":            MagicMock(),
        "matplotlib.lines":             MagicMock(),
        "matplotlib.patches":           MagicMock(),
        "pandas":                       MagicMock(),
        "libpysal":                     libpysal_mock,
        "libpysal.weights":             libpysal_mock.weights,
        "esda":                         MagicMock(),
        "esda.moran":                   MagicMock(),
        "pyreadstat":                   MagicMock(),
        "statsmodels":                  MagicMock(),
        "statsmodels.formula":          MagicMock(),
        "statsmodels.formula.api":      MagicMock(),
        "geopandas":                    MagicMock(),
        "spreg":                        MagicMock(),
        "spreg.summary_output":         MagicMock(),
        "spreg.output":                 MagicMock(),
        "spreg.ml_error":               MagicMock(),
        "spreg.ml_lag":                 MagicMock(),
        "panel_data":                   MagicMock(),
        "w_variants":                   MagicMock(),
        "pyproj":                       MagicMock(),
        "shapely":                      MagicMock(),
        "linearmodels":                 MagicMock(),
        "formulaic":                    MagicMock(),
        "patsy":                        MagicMock(),
        "networkx":                     MagicMock(),
        "seaborn":                      MagicMock(),
        "joblib":                       MagicMock(),
    }
    for name, mock in mocks.items():
        if name not in sys.modules:
            sys.modules[name] = mock


_install_mocks()