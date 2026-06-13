"""Shared panel construction helpers for credit-growth models."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
import scipy.sparse

from utils import row_standardize, sparse_to_pysal_w


ROOT = Path(__file__).parents[1]
PANEL_PATH = ROOT / "data" / "estimation_panel.csv"
HMDA_PATH = ROOT / "Replication" / "20121416_1data" / "data" / "hmda.dta"
CONTROLS_PATH = ROOT / "Replication" / "20121416_1data" / "data" / "hp_dereg_controls.dta"

CREDIT_CONTROLS = [
    "LDl_nloans_b",
    "Dl_inc", "LDl_inc",
    "Dl_pop", "LDl_pop",
    "Dl_hpi", "LDl_hpi",
    "Dl_her_v", "LDl_her_v",
]

PLACEBO_CONTROLS = [
    "LDl_nloans_pl",
    "Dl_inc", "LDl_inc",
    "Dl_pop", "LDl_pop",
    "Dl_hpi", "LDl_hpi",
    "Dl_her_v", "LDl_her_v",
]

_PLACEBO_COLS = ["Dl_nloans_pl", "LDl_nloans_pl"]
_HMDA_SOURCE_COLS = [
    "Dl_nloans_b", "LDl_nloans_b",
    "Dl_nloans_pl", "LDl_nloans_pl",
    "Dl_her_v", "LDl_her_v",
]


def _read_panel():
    panel = pd.read_csv(PANEL_PATH)
    panel["fips5"] = panel["fips5"].astype(str).str.zfill(5)
    panel["year"] = panel["year"].astype(int)
    return panel


def _source_with_fips(path):
    df, _ = pyreadstat.read_dta(path)
    df["fips5"] = pd.to_numeric(df["county"], errors="coerce").astype("Int64")
    df["fips5"] = df["fips5"].astype(str).str.zfill(5)
    df["year"] = df["year"].astype(int)
    return df


def _merge_missing(panel, required_cols):
    """Left-merge missing replication columns while preserving panel rows."""
    missing = [c for c in required_cols if c not in panel.columns]
    if not missing:
        return panel

    hmda_needed = [c for c in missing if c in _HMDA_SOURCE_COLS]
    if hmda_needed:
        hmda_df = _source_with_fips(HMDA_PATH)
        avail = [c for c in hmda_needed if c in hmda_df.columns]
        if avail:
            panel = panel.merge(
                hmda_df[["fips5", "year"] + avail].copy(),
                on=["fips5", "year"],
                how="left",
            )

    missing = [c for c in required_cols if c not in panel.columns]
    if missing:
        controls_df = _source_with_fips(CONTROLS_PATH)
        avail = [c for c in missing if c in controls_df.columns]
        if avail:
            panel = panel.merge(
                controls_df[["fips5", "year"] + avail].copy(),
                on=["fips5", "year"],
                how="left",
            )

    missing = [c for c in required_cols if c not in panel.columns]
    if missing:
        raise KeyError(
            "Required credit-control columns missing from panel and replication "
            f"sources: {missing}"
        )
    return panel


def load_panel_with_credit():
    """Load estimation_panel.csv with the full Favara-Imbs credit controls.

    Returns Dl_nloans_b plus the nine time-varying county controls used in
    Favara & Imbs (2015), equation (1): lagged bank-loan growth, income,
    population, HPI, and Herfindahl loan-concentration changes with lags.
    Existing estimation_panel.csv columns are used directly. Missing loan and
    Herfindahl columns are read from hmda.dta in this replication bundle; the
    income, population, and HPI controls are read from hp_dereg_controls.dta if
    absent from estimation_panel.csv.
    """
    panel = _read_panel()
    return _merge_missing(panel, ["Dl_nloans_b"] + CREDIT_CONTROLS)


def load_panel_with_placebo():
    """Load estimation panel with bank/placebo lending and control columns.

    Placebo variable: Dl_nloans_pl is the log-change in mortgage loans
    originated by independent mortgage companies, thrifts, and credit unions
    (Favara & Imbs 2015). Under the identification assumption, interstate
    branching deregulation did not directly affect these non-bank lenders, so
    a near-zero coefficient on the deregulation instrument in regressions with
    Dl_nloans_pl as the DV supports the exclusion restriction.

    PLACEBO_CONTROLS mirrors CREDIT_CONTROLS but uses LDl_nloans_pl, the
    placebo equation's lagged dependent variable, as the first control.
    """
    panel = _read_panel()
    required = (
        ["Dl_nloans_b", "Dl_nloans_pl"]
        + CREDIT_CONTROLS
        + [c for c in PLACEBO_CONTROLS if c not in CREDIT_CONTROLS]
    )
    return _merge_missing(panel, required)


def get_samples(panel):
    """Return [(label, dataframe), ...] for all three estimation samples.

    'Full'      : all urban counties with available data (Favara & Imbs 2015,
                  Table 2 design).
    'Contig'    : counties in MSAs traversed by a state border (border == 1).
                  Favara & Imbs (2015) Appendix Tables A1/A2 design.
                  Identification assumption: local economic conditions vary
                  continuously across the border (Favara & Imbs 2015, p. 971).
                  The panel contains 273 contiguous-MSA counties across 43 MSAs.
    'NonContig' : counties NOT in border-straddling MSAs (border == 0).
                  Complementary comparison group to Contig.
    """
    return [
        ("Full",      panel),
        ("Contig",    panel[panel["border"] == 1].copy()),
        ("NonContig", panel[panel["border"] == 0].copy()),
    ]


def get_common_sample(panel, county_order, W_geo_all, W_bank_all, dv="Dl_nloans_b"):
    """Return the set of counties usable under BOTH W_geo and W_bank.

    A county is excluded if it:
      (a) has any missing year in `dv` (balanced-panel requirement), OR
      (b) is a structural island (zero row-sum) under W_geo, OR
      (c) is a structural island (zero row-sum) under W_bank.

    Variant-specific islands (e.g. W_bank_nonGeo may isolate extra counties)
    can still drop within that variant; this function only establishes the
    baseline common estimation set for comparability of log-likelihoods and
    gap z-tests across W specifications.

    Parameters
    ----------
    panel        : DataFrame with columns fips5, dv, border
    county_order : list of fips5 strings (master ordering from W_geo GAL file)
    W_geo_all    : (N_all, N_all) scipy sparse W_geo matrix aligned to county_order
    W_bank_all   : (N_all, N_all) scipy sparse W_bank matrix aligned to county_order

    Returns
    -------
    usable : list of fips5 strings in county_order order, safe under both W
    """
    expected_shape = (len(county_order), len(county_order))
    for name, W in (("W_geo_all", W_geo_all), ("W_bank_all", W_bank_all)):
        if not isinstance(W, scipy.sparse.spmatrix):
            raise ValueError(
                f"{name} must be a scipy.sparse matrix with expected shape "
                f"{expected_shape}; got {type(W).__name__}."
            )
        if W.shape[0] != W.shape[1]:
            raise ValueError(
                f"{name} must be square with expected shape {expected_shape}; "
                f"got shape {W.shape}."
            )

    if W_geo_all.shape != W_bank_all.shape:
        raise ValueError(
            "W_geo_all and W_bank_all must have identical shapes with "
            f"expected shape {expected_shape}; got W_geo_all shape "
            f"{W_geo_all.shape} and W_bank_all shape {W_bank_all.shape}."
        )
    if W_geo_all.shape[0] != len(county_order):
        raise ValueError(
            "W_geo_all and W_bank_all must align to county_order with "
            f"expected shape {expected_shape}; got W_geo_all shape "
            f"{W_geo_all.shape}, W_bank_all shape {W_bank_all.shape}, and "
            f"len(county_order)={len(county_order)}."
        )

    # (a) counties with any missing dv
    any_nan = panel.groupby("fips5")[dv].apply(lambda s: s.isna().any())
    sub_co  = set(panel["fips5"].unique())

    # (b) island detection in full matrices
    rs_geo  = np.array(W_geo_all.sum(axis=1)).flatten()
    rs_bank = np.array(W_bank_all.sum(axis=1)).flatten()
    island_geo  = {county_order[i] for i, r in enumerate(rs_geo)  if r == 0}
    island_bank = {county_order[i] for i, r in enumerate(rs_bank) if r == 0}
    islands = island_geo | island_bank

    usable = [
        c for c in county_order
        if c in sub_co
        and not any_nan.get(c, True)
        and c not in islands
    ]
    return usable


def build_arrays(panel, county_order, W_geo_all, W_bank_all,
                 YEARS, year_pos, N_ALL, panel_sub, sample_label):
    """
    Build long-format arrays for Panel_FE_Error or Panel_FE_Lag.

    Counties with any missing Dl_nloans_b, Linter_bra, or credit control are
    excluded so the panel remains balanced for PySAL's fixed-effects estimators.
    """
    T = len(YEARS)
    xvars = ["Linter_bra"] + CREDIT_CONTROLS

    any_nan = panel_sub.groupby("fips5")[[ "Dl_nloans_b"] + xvars].apply(
        lambda g: g.isna().any().any()
    )
    sub_co = set(panel_sub["fips5"].unique())
    usable = [
        c for c in county_order
        if c in sub_co and not any_nan.get(c, True)
    ]
    N = len(usable)
    usable_pos = {c: i for i, c in enumerate(usable)}

    df = (
        panel_sub[panel_sub["fips5"].isin(set(usable))]
        .assign(
            t_idx=lambda d: d["year"].map(year_pos),
            c_idx=lambda d: d["fips5"].map(usable_pos),
        )
        .sort_values(["t_idx", "c_idx"])
    )

    y_long = df["Dl_nloans_b"].values.reshape(-1, 1)
    dummy_years = YEARS[1:]
    t_idx_vec = df["t_idx"].values
    year_dummies = np.column_stack([
        (t_idx_vec == year_pos[yr]).astype(np.float64)
        for yr in dummy_years
    ])
    x_long = np.hstack([df[xvars].values.astype(np.float64), year_dummies])

    assert not np.isnan(y_long).any(), f"NaN in y ({sample_label})"
    assert not np.isnan(x_long).any(), f"NaN in X ({sample_label})"
    assert y_long.shape == (N * T, 1), f"y shape {y_long.shape}"
    assert x_long.shape == (N * T, len(xvars) + len(YEARS) - 1), f"x shape {x_long.shape}"

    idx = np.array([county_order.index(c) for c in usable])
    W_geo_sub = row_standardize(W_geo_all[idx, :][:, idx])
    W_bank_sub = row_standardize(W_bank_all[idx, :][:, idx])

    return (
        y_long,
        x_long,
        sparse_to_pysal_w(W_geo_sub),
        sparse_to_pysal_w(W_bank_sub),
        N,
    )
