"""
Tests for analysis/panel_data.py (changes introduced in this PR)

Covers:
  - CREDIT_CONTROLS constant  (new in this PR)
  - PLACEBO_CONTROLS constant (new in this PR)
  - _merge_missing()           (new refactored helper)
  - load_panel_with_credit()  (refactored to use _merge_missing)
  - load_panel_with_placebo() (refactored to use _merge_missing)
"""
import sys
import os
import importlib
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Bootstrap: ensure we can import the REAL panel_data module.
#
# conftest.py installs sys.modules["panel_data"] = MagicMock() so that other
# analysis modules can import it without needing real data files.  For these
# tests we want the *actual* implementation, so we temporarily bypass the
# conftest mock by removing the stub and importing fresh.
# ---------------------------------------------------------------------------

ANALYSIS_DIR = os.path.join(os.path.dirname(__file__), "..", "analysis")
sys.path.insert(0, ANALYSIS_DIR)

_real_panel_data = None


def _get_panel_data():
    """Return the real panel_data module, loading it exactly once."""
    global _real_panel_data
    if _real_panel_data is not None:
        return _real_panel_data

    # Temporarily replace the conftest MagicMock with the real module.
    # We keep a backup of any mock already in sys.modules and restore it
    # after import so that other tests are unaffected.
    backup = sys.modules.get("panel_data")
    sys.modules.pop("panel_data", None)

    # panel_data.py also imports utils (mocked via conftest), pyreadstat
    # (mocked), scipy.sparse (mocked), and pandas (real, installed).
    # Those mocks are already in place; we only need to temporarily unblock
    # "panel_data" itself.
    try:
        mod = importlib.import_module("panel_data")
        _real_panel_data = mod
    finally:
        # If the import failed, restore the original mock so other tests work.
        if _real_panel_data is None and backup is not None:
            sys.modules["panel_data"] = backup

    return _real_panel_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeDF:
    """Minimal DataFrame-like object for _merge_missing testing."""

    def __init__(self, columns):
        self.columns = list(columns)
        self._was_merged = False

    def merge(self, other, on, how):
        """Simulate a successful left-merge by adding other's columns."""
        combined_cols = list(self.columns) + [
            c for c in other.columns if c not in self.columns
        ]
        result = _FakeDF(combined_cols)
        result._was_merged = True
        return result


# ---------------------------------------------------------------------------
# CREDIT_CONTROLS
# ---------------------------------------------------------------------------

class TestCreditControls:
    """Tests for the CREDIT_CONTROLS list constant."""

    def test_type_is_list(self):
        pd_mod = _get_panel_data()
        assert isinstance(pd_mod.CREDIT_CONTROLS, list)

    def test_length(self):
        pd_mod = _get_panel_data()
        assert len(pd_mod.CREDIT_CONTROLS) == 9

    def test_first_element_lagged_dep_var(self):
        pd_mod = _get_panel_data()
        assert pd_mod.CREDIT_CONTROLS[0] == "LDl_nloans_b", (
            "First credit control should be the lagged dependent variable"
        )

    def test_contains_income_controls(self):
        pd_mod = _get_panel_data()
        cc = pd_mod.CREDIT_CONTROLS
        assert "Dl_inc" in cc
        assert "LDl_inc" in cc

    def test_contains_population_controls(self):
        pd_mod = _get_panel_data()
        cc = pd_mod.CREDIT_CONTROLS
        assert "Dl_pop" in cc
        assert "LDl_pop" in cc

    def test_contains_hpi_controls(self):
        pd_mod = _get_panel_data()
        cc = pd_mod.CREDIT_CONTROLS
        assert "Dl_hpi" in cc
        assert "LDl_hpi" in cc

    def test_contains_herfindahl_controls(self):
        pd_mod = _get_panel_data()
        cc = pd_mod.CREDIT_CONTROLS
        assert "Dl_her_v" in cc
        assert "LDl_her_v" in cc

    def test_no_duplicates(self):
        pd_mod = _get_panel_data()
        cc = pd_mod.CREDIT_CONTROLS
        assert len(cc) == len(set(cc)), "CREDIT_CONTROLS should have no duplicates"

    def test_all_strings(self):
        pd_mod = _get_panel_data()
        for item in pd_mod.CREDIT_CONTROLS:
            assert isinstance(item, str)

    def test_does_not_contain_placebo_var(self):
        pd_mod = _get_panel_data()
        assert "LDl_nloans_pl" not in pd_mod.CREDIT_CONTROLS, (
            "CREDIT_CONTROLS should not include the placebo lagged DV"
        )


# ---------------------------------------------------------------------------
# PLACEBO_CONTROLS
# ---------------------------------------------------------------------------

class TestPlaceboControls:
    """Tests for the PLACEBO_CONTROLS list constant."""

    def test_type_is_list(self):
        pd_mod = _get_panel_data()
        assert isinstance(pd_mod.PLACEBO_CONTROLS, list)

    def test_length(self):
        pd_mod = _get_panel_data()
        assert len(pd_mod.PLACEBO_CONTROLS) == 9

    def test_first_element_placebo_lagged_dep_var(self):
        pd_mod = _get_panel_data()
        assert pd_mod.PLACEBO_CONTROLS[0] == "LDl_nloans_pl", (
            "First placebo control should be the placebo lagged DV"
        )

    def test_differs_from_credit_controls_in_first(self):
        pd_mod = _get_panel_data()
        assert pd_mod.PLACEBO_CONTROLS[0] != pd_mod.CREDIT_CONTROLS[0]

    def test_shares_macro_controls_with_credit(self):
        """PLACEBO_CONTROLS should share all controls except the lagged DV."""
        pd_mod = _get_panel_data()
        cc = set(pd_mod.CREDIT_CONTROLS)
        pc = set(pd_mod.PLACEBO_CONTROLS)
        # Both share the macro controls (income, pop, hpi, herfindahl)
        shared = cc & pc
        assert "Dl_inc" in shared
        assert "LDl_inc" in shared
        assert "Dl_pop" in shared
        assert "LDl_pop" in shared
        assert "Dl_hpi" in shared
        assert "LDl_hpi" in shared
        assert "Dl_her_v" in shared
        assert "LDl_her_v" in shared

    def test_no_duplicates(self):
        pd_mod = _get_panel_data()
        pc = pd_mod.PLACEBO_CONTROLS
        assert len(pc) == len(set(pc))

    def test_all_strings(self):
        pd_mod = _get_panel_data()
        for item in pd_mod.PLACEBO_CONTROLS:
            assert isinstance(item, str)

    def test_does_not_contain_credit_lagged_dv(self):
        pd_mod = _get_panel_data()
        assert "LDl_nloans_b" not in pd_mod.PLACEBO_CONTROLS


# ---------------------------------------------------------------------------
# _merge_missing
# ---------------------------------------------------------------------------

class TestMergeMissing:
    """Tests for _merge_missing(panel, required_cols)."""

    def _fn(self, panel, required_cols):
        return _get_panel_data()._merge_missing(panel, required_cols)

    def test_no_missing_returns_same_object(self):
        """If all required columns already exist, panel is returned unchanged."""
        panel = _FakeDF(["fips5", "year", "Dl_nloans_b", "LDl_nloans_b"])
        result = self._fn(panel, ["Dl_nloans_b", "LDl_nloans_b"])
        # Should be the same object (no merge happened)
        assert result is panel

    def test_empty_required_list_returns_same_object(self):
        panel = _FakeDF(["fips5", "year", "Dl_nloans_b"])
        result = self._fn(panel, [])
        assert result is panel

    def test_missing_column_triggers_hmda_merge(self):
        """A missing HMDA column triggers a merge with the HMDA source."""
        pd_mod = _get_panel_data()

        # Panel is missing Dl_nloans_b (an HMDA column)
        panel = _FakeDF(["fips5", "year"])

        fake_hmda_df = _FakeDF(["fips5", "year", "Dl_nloans_b", "LDl_nloans_b",
                                 "Dl_nloans_pl", "LDl_nloans_pl", "Dl_her_v", "LDl_her_v"])

        with patch.object(pd_mod, "_source_with_fips", return_value=fake_hmda_df):
            result = self._fn(panel, ["Dl_nloans_b"])

        # Result should have Dl_nloans_b now
        assert "Dl_nloans_b" in result.columns

    def test_missing_non_hmda_column_triggers_controls_merge(self):
        """Columns not in _HMDA_SOURCE_COLS are sourced from hp_dereg_controls."""
        pd_mod = _get_panel_data()

        # Panel is missing Dl_inc (not in HMDA source)
        panel = _FakeDF(["fips5", "year"])

        fake_hmda_df = _FakeDF(["fips5", "year"])  # HMDA doesn't have Dl_inc
        fake_controls_df = _FakeDF(["fips5", "year", "Dl_inc", "LDl_inc",
                                     "Dl_pop", "LDl_pop", "Dl_hpi", "LDl_hpi"])

        call_count = [0]

        def fake_source(path):
            call_count[0] += 1
            if "hmda" in str(path):
                return fake_hmda_df
            return fake_controls_df

        with patch.object(pd_mod, "_source_with_fips", side_effect=fake_source):
            result = self._fn(panel, ["Dl_inc"])

        assert "Dl_inc" in result.columns

    def test_truly_missing_column_raises_key_error(self):
        """If column is absent from all sources, KeyError should be raised."""
        pd_mod = _get_panel_data()

        panel = _FakeDF(["fips5", "year"])
        # Both sources are missing "phantom_col"
        empty_source = _FakeDF(["fips5", "year"])

        with patch.object(pd_mod, "_source_with_fips", return_value=empty_source):
            with pytest.raises(KeyError, match="phantom_col"):
                self._fn(panel, ["phantom_col"])

    def test_all_credit_controls_present_returns_unchanged(self):
        """Panel containing all CREDIT_CONTROLS returns without merging."""
        pd_mod = _get_panel_data()
        all_cols = ["fips5", "year", "Dl_nloans_b"] + pd_mod.CREDIT_CONTROLS
        panel = _FakeDF(all_cols)
        result = self._fn(panel, ["Dl_nloans_b"] + pd_mod.CREDIT_CONTROLS)
        assert result is panel


# ---------------------------------------------------------------------------
# load_panel_with_credit and load_panel_with_placebo (smoke tests)
# ---------------------------------------------------------------------------

class TestLoadFunctionSignatures:
    """Smoke tests for the public load functions."""

    def test_load_credit_calls_merge_missing(self):
        """load_panel_with_credit calls _merge_missing with credit controls."""
        pd_mod = _get_panel_data()

        fake_panel = _FakeDF(["fips5", "year", "Dl_nloans_b"] + pd_mod.CREDIT_CONTROLS)

        with patch.object(pd_mod, "_read_panel", return_value=fake_panel), \
             patch.object(pd_mod, "_merge_missing", return_value=fake_panel) as mock_mm:
            pd_mod.load_panel_with_credit()

        mock_mm.assert_called_once()
        call_args = mock_mm.call_args[0]
        required = call_args[1]
        # "Dl_nloans_b" plus all 9 credit controls
        assert "Dl_nloans_b" in required
        for ctrl in pd_mod.CREDIT_CONTROLS:
            assert ctrl in required, f"Missing control: {ctrl}"

    def test_load_placebo_calls_merge_missing(self):
        """load_panel_with_placebo calls _merge_missing with bank + placebo cols."""
        pd_mod = _get_panel_data()

        all_cols = (
            ["fips5", "year", "Dl_nloans_b", "Dl_nloans_pl"]
            + pd_mod.CREDIT_CONTROLS
            + pd_mod.PLACEBO_CONTROLS
        )
        fake_panel = _FakeDF(all_cols)

        with patch.object(pd_mod, "_read_panel", return_value=fake_panel), \
             patch.object(pd_mod, "_merge_missing", return_value=fake_panel) as mock_mm:
            pd_mod.load_panel_with_placebo()

        mock_mm.assert_called_once()
        call_args = mock_mm.call_args[0]
        required = call_args[1]
        assert "Dl_nloans_b" in required
        assert "Dl_nloans_pl" in required

    def test_load_credit_requires_all_credit_controls(self):
        """The required list passed to _merge_missing must include all 9 controls."""
        pd_mod = _get_panel_data()
        all_cols = ["fips5", "year", "Dl_nloans_b"] + pd_mod.CREDIT_CONTROLS
        fake_panel = _FakeDF(all_cols)

        captured = {}

        def capturing_merge(panel, required_cols):
            captured["required"] = required_cols
            return panel

        with patch.object(pd_mod, "_read_panel", return_value=fake_panel), \
             patch.object(pd_mod, "_merge_missing", side_effect=capturing_merge):
            pd_mod.load_panel_with_credit()

        for ctrl in pd_mod.CREDIT_CONTROLS:
            assert ctrl in captured["required"], f"{ctrl} not in required list"
        assert len([c for c in captured["required"] if c in pd_mod.CREDIT_CONTROLS]) == 9