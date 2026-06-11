"""Shared panel construction helpers for credit-growth models."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

from utils import row_standardize, sparse_to_pysal_w


ROOT = Path(__file__).parents[1]
PANEL_PATH = ROOT / "data" / "estimation_panel.csv"
HMDA_PATH = ROOT / "Replication" / "20121416_1data" / "data" / "hmda.dta"

_PLACEBO_COLS = ["Dl_nloans_pl", "LDl_nloans_pl"]


def load_panel_with_credit():
    """Load estimation_panel.csv merged with Dl_nloans_b from hmda.dta."""
    panel = pd.read_csv(PANEL_PATH)
    panel["fips5"] = panel["fips5"].astype(str).str.zfill(5)

    if "Dl_nloans_b" in panel.columns:
        return panel

    hmda_df, _ = pyreadstat.read_dta(HMDA_PATH)
    hmda_df["fips5"] = (
        hmda_df["county"].dropna().astype(int).astype(str).str.zfill(5)
    )
    hmda_df["year"] = hmda_df["year"].astype(int)
    hmda = hmda_df[["fips5", "year", "Dl_nloans_b"]].copy()

    return panel.merge(hmda, on=["fips5", "year"], how="left")


def load_panel_with_placebo():
    """Load estimation panel merged with bank (Dl_nloans_b) and placebo
    (Dl_nloans_pl) lending variables from hmda.dta.

    Placebo variable: Dl_nloans_pl is the log-change in mortgage loans
    originated by independent mortgage companies, thrifts, and credit unions
    (Favara & Imbs 2015). Under the identification assumption, interstate
    branching deregulation did not directly affect these non-bank lenders, so
    a near-zero coefficient on the deregulation instrument in regressions with
    Dl_nloans_pl as the DV supports the exclusion restriction.
    """
    panel = pd.read_csv(PANEL_PATH)
    panel["fips5"] = panel["fips5"].astype(str).str.zfill(5)

    merge_cols = ["Dl_nloans_b"] + _PLACEBO_COLS
    need = [c for c in merge_cols if c not in panel.columns]
    if not need:
        return panel

    hmda_df, _ = pyreadstat.read_dta(HMDA_PATH)
    hmda_df["fips5"] = (
        hmda_df["county"].dropna().astype(int).astype(str).str.zfill(5)
    )
    hmda_df["year"] = hmda_df["year"].astype(int)
    avail = [c for c in merge_cols if c in hmda_df.columns]
    hmda = hmda_df[["fips5", "year"] + avail].copy()

    return panel.merge(hmda, on=["fips5", "year"], how="left")


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

    Counties with any missing Dl_nloans_b are excluded so the panel remains
    balanced for PySAL's fixed-effects estimators.
    """
    T = len(YEARS)

    any_nan = panel_sub.groupby("fips5")["Dl_nloans_b"].apply(
        lambda s: s.isna().any()
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
    x_long = np.hstack([df["Linter_bra"].values.reshape(-1, 1), year_dummies])

    assert not np.isnan(y_long).any(), f"NaN in y ({sample_label})"
    assert not np.isnan(x_long[:, 0]).any(), f"NaN in Linter_bra ({sample_label})"
    assert y_long.shape == (N * T, 1), f"y shape {y_long.shape}"
    assert x_long.shape == (N * T, 11), f"x shape {x_long.shape}"

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
