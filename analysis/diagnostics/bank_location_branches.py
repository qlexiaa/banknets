"""
bank_location_branches.py
=========================
Favara & Imbs (2015) Table 3 / Table A2 decomposition by bank type and
branch location.  Reproduces the paper's finding that interstate-bank credit
growth requires a *local branch presence* (Panel A > Panel B ~ Panel C ~ 0).

Three panels:
  A: Out-of-state banks with local branches  -- the transmission channel
  B: Out-of-state banks without local branches -- no local presence
  C: In-state banks                           -- baseline comparison

Two samples (matching F&I):
  Full    : all urban counties with complete data (Table 3 equivalent)
  Contig  : counties in border-straddling MSAs (Table A2 equivalent)

Outputs (relative to output_dir / ROOT/tables):
  output/bank_location_branches.csv          -- tidy panel-level results
  output/bank_network_descriptives.csv       -- SOD BHC & county-pair stats
  tables/bank_location_branches.tex          -- LaTeX fragment for thesis

Specification replicates F&I equation (1):
  delta_ln(loans)_ct = beta * Linter_bra_ct
                     + gamma' X_ct          (8 economic controls)
                     + L_delta_ln(own_DV)_ct  (lagged own DV)
                     + alpha_c              (county FE, absorbed by PanelOLS)
                     + tau_t               (year FE, via year dummies 1995-2005)
                     + u_ct

SEs clustered by state (Cameron & Miller 2015; Favara & Imbs 2015).

References:
  Favara & Imbs (2015, AER 104(3): 272-323)
  Cameron & Miller (2015, J. Human Resources)
  Wooldridge (2010) Econometric Analysis of Cross Section and Panel Data
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyreadstat
import scipy.sparse
import scipy.stats as _st
from linearmodels.panel import PanelOLS

ROOT = Path(__file__).parents[2]
DATA = ROOT / "data"
REPL = ROOT / "Replication" / "20121416_1data" / "data"
HMDA_PATH     = REPL / "hmda.dta"
CONTROLS_PATH = REPL / "hp_dereg_controls.dta"
SOD_PATH      = DATA / "fdic_deposits_1994_2005.csv"
W_BINARY_PATH = DATA / "W_bank_binary.npz"
CO_PATH       = DATA / "county_order_Wgeo.csv"
TABLES_DIR    = ROOT / "tables"

# ── Economic controls (F&I D_CTL) ─────────────────────────────────────────────
D_CTL = [
    "Dl_inc",  "LDl_inc",
    "Dl_pop",  "LDl_pop",
    "Dl_hpi",  "LDl_hpi",
    "Dl_her_v","LDl_her_v",
]

# ── Panel definitions ─────────────────────────────────────────────────────────
# Each panel maps canonical "outcome key" -> (DV column, lagged-DV column)
# The outcome key allows cross-panel Wald tests on matched outcomes.
PANELS: dict[str, dict] = {
    "A": {
        "label": "OOS with local branches",
        "outcomes": [
            ("nloans",  "Dl_nloans_b_oos_lb",  "LDl_nloans_b_oos_lb"),
            ("vloans",  "Dl_vloans_b_oos_lb",  "LDl_vloans_b_oos_lb"),
            ("nden",    "Dl_nden_b_oos_lb",    "LDl_nden_b_oos_lb"),
            ("lir",     "Dl_lir_b_oos_lb",     "LDl_lir_b_oos_lb"),
            ("nsold",   "Dl_nsold_b_oos_lb",   "LDl_nsold_b_oos_lb"),
            ("nbra",    "Dl_nbra_oos_lb",      "LDl_nbra_oos_lb"),
        ],
    },
    "B": {
        "label": "OOS without local branches",
        "outcomes": [
            ("nloans",  "Dl_nloans_b_oos",     "LDl_nloans_b_oos"),
            ("vloans",  "Dl_vloans_b_oos",     "LDl_vloans_b_oos"),
            ("nden",    "Dl_nden_b_oos",       "LDl_nden_b_oos"),
            ("lir",     "Dl_lir_b_oos",        "LDl_lir_b_oos"),
            ("nsold",   "Dl_nsold_b_oos",      "LDl_nsold_b_oos"),
        ],
    },
    "C": {
        "label": "In-state banks",
        "outcomes": [
            ("nloans",  "Dl_nloans_b_is",      "LDl_nloans_b_is"),
            ("vloans",  "Dl_vloans_b_is",      "LDl_vloans_b_is"),
            ("nden",    "Dl_nden_b_is",        "LDl_nden_b_is"),
            ("lir",     "Dl_lir_b_is",         "LDl_lir_b_is"),
            ("nsold",   "Dl_nsold_b_is",       "LDl_nsold_b_is"),
            ("nbra",    "Dl_nbra_is",          "LDl_nbra_is"),
        ],
    },
}


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_raw_data() -> pd.DataFrame:
    """Merge hmda.dta + hp_dereg_controls.dta into a county×year panel.

    Returns a DataFrame with columns:
      fips5 (str, zero-padded), year (int), state_n (int), border (0/1),
      Linter_bra, D_CTL columns, all Panel A/B/C outcome columns.
    """
    hmda, _ = pyreadstat.read_dta(str(HMDA_PATH))
    hp,   _ = pyreadstat.read_dta(str(CONTROLS_PATH))

    # Drop year dummies already in hp (recreate from year column)
    yr_cols = [c for c in hp.columns if c.startswith("yryear")]
    hp = hp.drop(columns=yr_cols + ["state_n"], errors="ignore")

    # Ensure consistent key types
    hmda["county"] = pd.to_numeric(hmda["county"], errors="coerce").astype("Int64")
    hmda["year"]   = hmda["year"].astype(int)
    hp["county"]   = pd.to_numeric(hp["county"],   errors="coerce").astype("Int64")
    hp["year"]     = hp["year"].astype(int)

    df = hmda.merge(hp, on=["county", "year"], how="left")

    # FIPS and state key
    df["fips5"]   = df["county"].astype(str).str.zfill(5)
    df["state_n"] = df["state_n"].astype("Int64")

    return df


def _build_year_dummies(df: pd.DataFrame) -> list[str]:
    """Add year-dummy columns for 1995-2005; return their names."""
    years = range(1995, 2006)
    cols  = []
    for yr in years:
        col = f"yryear_{yr}"
        df[col] = (df["year"] == yr).astype(float)
        cols.append(col)
    return cols


# ── Regression helper ─────────────────────────────────────────────────────────

def _fe_reg(
    df: pd.DataFrame,
    dep: str,
    lag: Optional[str],
    yr_dummies: list[str],
) -> Optional[dict]:
    """Two-way FE regression (entity FE + year dummies), state-clustered SEs.

    Returns dict with beta, se, pval, N, n_counties, n_states, R2_within
    or None if the regression cannot be run (collinear, too few obs).
    """
    xvars = ["Linter_bra"] + D_CTL
    if lag is not None and lag in df.columns:
        xvars = xvars + [lag]
    xvars = xvars + [c for c in yr_dummies if c in df.columns]

    # Subset to complete cases on dep and all regressors
    cols_needed = ["fips5", "year", "state_n", dep] + xvars
    sub = df[[c for c in cols_needed if c in df.columns]].dropna().copy()

    if sub[dep].std() < 1e-12:
        return None
    if len(sub) < 50:
        return None

    # Drop constant columns (e.g. a year with no variation)
    valid_x = [c for c in xvars if c in sub.columns and sub[c].std() > 1e-12]
    if "Linter_bra" not in valid_x:
        return None

    sub = sub.set_index(["fips5", "year"])

    # PanelOLS requires integer cluster index to be in index or passed separately
    clusters = sub["state_n"].astype(int)
    sub = sub.drop(columns=["state_n"])

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = PanelOLS(sub[dep], sub[valid_x], entity_effects=True)
            res = mod.fit(cov_type="clustered", clusters=clusters)
    except Exception:
        return None

    beta  = float(res.params["Linter_bra"])
    se    = float(res.std_errors["Linter_bra"])
    tval  = float(res.tstats["Linter_bra"])
    pval  = float(res.pvalues["Linter_bra"])
    r2w   = float(res.rsquared_within)

    # Recover entity identifiers from index
    idx   = res.model.dependent.dataframe.index
    fips_vals = idx.get_level_values(0)
    n_co  = int(fips_vals.nunique())
    n_obs = int(res.nobs)

    # State count
    n_st  = int(clusters.loc[idx].nunique()) if hasattr(idx, "names") else np.nan

    return {
        "beta": beta, "se": se, "tval": tval, "pval": pval,
        "N": n_obs, "n_counties": n_co, "n_states": n_st,
        "R2_within": r2w,
    }


# ── Joint cross-panel equality test (SUR-score approach) ─────────────────────

def _fe_reg_full(
    df: pd.DataFrame,
    dep: str,
    lag: Optional[str],
    yr_dummies: list[str],
) -> Optional[dict]:
    """Like _fe_reg but also returns the PanelOLS result object.

    Returns dict with the usual scalar keys PLUS:
      'res'        : linearmodels PanelOLS result object
      'sample_idx' : (entity, year) MultiIndex of the estimation sample
    """
    xvars = ["Linter_bra"] + D_CTL
    if lag is not None and lag in df.columns:
        xvars = xvars + [lag]
    xvars = xvars + [c for c in yr_dummies if c in df.columns]

    cols_needed = ["fips5", "year", "state_n", dep] + xvars
    sub = df[[c for c in cols_needed if c in df.columns]].dropna().copy()

    if sub[dep].std() < 1e-12 or len(sub) < 50:
        return None

    valid_x = [c for c in xvars if c in sub.columns and sub[c].std() > 1e-12]
    if "Linter_bra" not in valid_x:
        return None

    sub = sub.set_index(["fips5", "year"])
    clusters = sub["state_n"].astype(int)
    sub = sub.drop(columns=["state_n"])

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = PanelOLS(sub[dep], sub[valid_x], entity_effects=True)
            res = mod.fit(cov_type="clustered", clusters=clusters)
    except Exception:
        return None

    beta  = float(res.params["Linter_bra"])
    se    = float(res.std_errors["Linter_bra"])
    tval  = float(res.tstats["Linter_bra"])
    pval  = float(res.pvalues["Linter_bra"])
    r2w   = float(res.rsquared_within)
    idx   = res.model.dependent.dataframe.index
    fips_vals = idx.get_level_values(0)
    n_co  = int(fips_vals.nunique())
    n_obs = int(res.nobs)
    n_st  = int(clusters.loc[idx].nunique()) if hasattr(idx, "names") else np.nan

    return {
        "beta": beta, "se": se, "tval": tval, "pval": pval,
        "N": n_obs, "n_counties": n_co, "n_states": n_st,
        "R2_within": r2w,
        "res": res,
        "sample_idx": idx,
        "clusters": clusters,
        "valid_x": valid_x,
        "sub": sub,            # entity×year indexed DataFrame with dep + valid_x
        "dep": dep,
    }


def _fwl_residual_and_scores(full: dict) -> Optional[tuple]:
    """Frisch-Waugh-Lovell residual for Linter_bra and state-level scores.

    Computes ẍ (Linter_bra after partialing out entity FE and all other
    regressors within the estimation sample) and the state-level scores
    S_s = Σ_{ct∈s} ẍ_ct ε̃_ct.

    Returns (x_ddot, scores_by_state, bread, state_array) or None on failure.
    """
    res      = full["res"]
    valid_x  = full["valid_x"]
    idx      = full["sample_idx"]
    clusters = full["clusters"]

    if "Linter_bra" not in valid_x:
        return None

    try:
        # Entity-demeaned X: PanelData.demean() → .dataframe gives (N*T × K)
        X_dmd_df = res.model.exog.demean("entity").dataframe   # DataFrame, (N*T, K)
        X_dmd    = np.asarray(X_dmd_df, dtype=float)           # (N*T, K)
    except Exception:
        return None

    linter_col = list(X_dmd_df.columns).index("Linter_bra")
    x_tilde    = X_dmd[:, linter_col]                          # (N*T,)
    Z_tilde    = np.delete(X_dmd, linter_col, axis=1)          # (N*T, K-1)

    # FWL: partial out all other entity-demeaned regressors from Linter_bra
    if Z_tilde.shape[1] > 0:
        try:
            coef_fw, _, _, _ = np.linalg.lstsq(Z_tilde, x_tilde, rcond=None)
            x_ddot = x_tilde - Z_tilde @ coef_fw
        except np.linalg.LinAlgError:
            x_ddot = x_tilde
    else:
        x_ddot = x_tilde

    eps_tilde = np.asarray(res.resids, dtype=float).ravel()    # (N*T,)
    if len(eps_tilde) != len(x_ddot):
        return None

    # State labels aligned to estimation sample index
    state_s = clusters.reindex(idx).astype(int).values         # (N*T,)

    scores = pd.Series(x_ddot * eps_tilde, index=state_s)
    scores_by_state = scores.groupby(level=0).sum()
    bread = float((x_ddot ** 2).sum())
    return x_ddot, scores_by_state, bread, state_s


def _stacked_equality_test(
    df: pd.DataFrame,
    outcome_key: str,
    yr_dummies: list[str],
    full_fits: dict[str, Optional[dict]],
) -> dict[str, dict]:
    """Joint cross-panel test H0: beta_B = beta_A  (and H0: beta_C = beta_A).

    Uses the SUR-score approach (Cameron & Miller 2015, Section 2.5):
    runs the individual regressions, extracts state-level score vectors via
    the Frisch-Waugh-Lovell theorem, and estimates the cross-equation covariance
    of beta_A with beta_B (and beta_C) from the cross products of scores.
    This correctly accounts for the positive covariance between within-panel
    estimators that share counties and years, making the test more powerful
    than the independence-assumption Wald test.

    Formally:
      Var(δ_B) = Var(β_A) + Var(β_B) − 2·Cov(β_A, β_B)
      Cov(β_A, β_B) = G/(G−1) · Σ_s S_A_s · S_B_s / (bread_A · bread_B)

    where S_k_s = Σ_{ct∈s} ẍ_k,ct · ε̃_k,ct, ẍ is the Frisch-Waugh
    treatment residual, and bread_k = Σ ẍ_k,ct².

    Parameters
    ----------
    df          : full panel DataFrame
    outcome_key : e.g. 'nloans'
    yr_dummies  : column names from _build_year_dummies()
    full_fits   : dict pid → _fe_reg_full() output (may be None)

    Returns
    -------
    dict keyed by non-A panel_id → {'delta': ..., 'se': ..., 'pval': ...}
    """
    fit_A = full_fits.get("A")
    if fit_A is None:
        return {}

    # Panel defs for this outcome_key
    def _get(pid, key):
        return next((x for x in PANELS[pid]["outcomes"] if x[0] == key), None)

    non_A_defs = [(pid, d) for pid in ("B", "C")
                  if (d := _get(pid, outcome_key)) is not None]
    if not non_A_defs:
        return {}

    # FWL residuals and scores for Panel A
    fwl_A = _fwl_residual_and_scores(fit_A)
    if fwl_A is None:
        return {}
    _, scores_A, bread_A, states_A = fwl_A

    # SE_A² from PanelOLS (G/G-1 corrected)
    var_A = float(fit_A["se"]) ** 2
    G_A   = int(fit_A["n_states"])

    out: dict[str, dict] = {}
    for pid, (_, dep_pid, lag_pid) in non_A_defs:
        fit_B = full_fits.get(pid)
        if fit_B is None:
            continue
        fwl_B = _fwl_residual_and_scores(fit_B)
        if fwl_B is None:
            continue
        _, scores_B, bread_B, states_B = fwl_B

        beta_A = float(fit_A["beta"])
        beta_B = float(fit_B["beta"])
        var_B  = float(fit_B["se"]) ** 2
        G_B    = int(fit_B["n_states"])

        # Use the harmonic mean of G for cross-panel scaling
        G      = max(G_A, G_B)   # both should be ~50

        # Cross-panel score product (align on common states)
        common_states = scores_A.index.intersection(scores_B.index)
        if len(common_states) < 2:
            continue
        cross_meat = float(
            (scores_A.reindex(common_states) * scores_B.reindex(common_states)).sum()
        )

        # Cov(beta_A, beta_B) = G/(G-1) · cross_meat / (bread_A · bread_B)
        cov_AB = (G / (G - 1)) * cross_meat / (bread_A * bread_B)

        # Var(delta) = Var(beta_A) + Var(beta_B) - 2*Cov(beta_A, beta_B)
        var_delta = var_A + var_B - 2.0 * cov_AB
        if var_delta <= 0:
            # Fallback to independence assumption (conservative)
            var_delta = var_A + var_B

        delta = beta_B - beta_A
        se    = float(np.sqrt(var_delta))
        tval  = delta / se if se > 0 else float("nan")
        pval  = float(2 * _st.norm.sf(abs(tval)))

        out[pid] = {"delta": delta, "se": se, "pval": pval}

    return out


# ── Main computation ──────────────────────────────────────────────────────────

def _run_sample(df: pd.DataFrame, sample_label: str) -> list[dict]:
    """Run all panel/outcome regressions for one sample; return list of rows."""
    yr_dummies = _build_year_dummies(df)
    rows: list[dict] = []

    # ── Step 1: individual per-panel regressions (point estimates + SEs) ──
    # Use _fe_reg_full to also retain residuals / FWL ingredients for tests.
    panel_full: dict[str, dict[str, Optional[dict]]] = {}
    for panel_id, pdef in PANELS.items():
        panel_full[panel_id] = {}
        print(f"  Sample={sample_label}  Panel {panel_id}", flush=True)
        for key, dep, lag in pdef["outcomes"]:
            r = _fe_reg_full(df, dep, lag, yr_dummies)
            panel_full[panel_id][key] = r
            if r is not None:
                print(f"    {key}: beta={r['beta']:.4f}  SE={r['se']:.4f}  "
                      f"N={r['N']:,}", flush=True)

    # ── Step 2: joint cross-panel tests per outcome key (SUR-score method) ─
    all_keys = {key for pdef in PANELS.values() for key, _, _ in pdef["outcomes"]}
    print(f"  Sample={sample_label}  Joint cross-panel tests (SUR scores) ...",
          flush=True)
    stacked_tests: dict[str, dict[str, dict]] = {}
    for key in sorted(all_keys):
        # Collect the per-panel full fits for this outcome key
        full_fits_key: dict[str, Optional[dict]] = {}
        for pid in ("A", "B", "C"):
            full_fits_key[pid] = panel_full.get(pid, {}).get(key)
        stacked_tests[key] = _stacked_equality_test(
            df, key, yr_dummies, full_fits_key
        )
        info = stacked_tests[key]
        if info:
            parts = [f"{pid}: delta={d['delta']:.4f} p={d['pval']:.3f}"
                     for pid, d in info.items()]
            print(f"    {key}: {',  '.join(parts)}", flush=True)

    # ── Step 3: assemble output rows (strip internal fit objects) ─────────
    def _scalars(r):
        """Return only the scalar fields from _fe_reg_full output."""
        if r is None:
            return None
        return {k: v for k, v in r.items()
                if k not in ("res", "sample_idx", "clusters", "valid_x", "sub", "dep")}

    for panel_id, pdef in PANELS.items():
        for key, dep, lag in pdef["outcomes"]:
            r = _scalars(panel_full[panel_id].get(key))
            test_pval = float("nan")
            if panel_id != "A" and key in stacked_tests:
                test_pval = stacked_tests[key].get(panel_id, {}).get("pval",
                                                                       float("nan"))
            if r is not None:
                rows.append({
                    "sample":       sample_label,
                    "panel":        panel_id,
                    "panel_label":  pdef["label"],
                    "outcome_key":  key,
                    "outcome":      dep,
                    **r,
                    "test_vs_A_pval": test_pval,
                })
            else:
                rows.append({
                    "sample": sample_label, "panel": panel_id,
                    "panel_label": pdef["label"],
                    "outcome_key": key, "outcome": dep,
                    "beta": float("nan"), "se": float("nan"),
                    "tval": float("nan"), "pval": float("nan"),
                    "N": 0, "n_counties": 0, "n_states": 0,
                    "R2_within": float("nan"),
                    "test_vs_A_pval": float("nan"),
                })

    return rows


# ── SOD descriptives (Phase 3) ────────────────────────────────────────────────

def _build_sod_descriptives() -> pd.DataFrame:
    """Compute BHC multi-market statistics and county-pair classifications.

    Uses fdic_deposits_1994_2005.csv (1994-2005 FDIC Summary of Deposits)
    and W_bank_binary.npz (county-pair bank network).

    Returns a one-row DataFrame with scalar statistics.
    """
    # ── SOD BHC statistics ─────────────────────────────────────────────────
    sod = pd.read_csv(SOD_PATH)
    sod["fips5"] = sod["STCNTYBR"].astype(str).str.zfill(5)
    sod["state_fips"] = sod["fips5"].str[:2]
    # Keep only records with a valid BHC (RSSDHCR > 0)
    sod = sod[sod["RSSDHCR"] > 0].copy()

    # Per BHC-year: number of distinct counties and states
    bhc_yr = (sod.groupby(["RSSDHCR", "year"])
              .agg(n_counties=("fips5",      "nunique"),
                   n_states  =("state_fips", "nunique"))
              .reset_index())

    # Per BHC: average over years
    bhc = (bhc_yr.groupby("RSSDHCR")
           .agg(mean_n_counties=("n_counties", "mean"),
                mean_n_states  =("n_states",   "mean"),
                max_n_counties =("n_counties", "max"),
                max_n_states   =("n_states",   "max"))
           .reset_index())

    n_bhc_total        = len(bhc)
    n_bhc_multi_county = int((bhc["mean_n_counties"] > 1).sum())
    n_bhc_multi_state  = int((bhc["mean_n_states"]   > 1).sum())
    share_multi_county = n_bhc_multi_county / n_bhc_total
    share_multi_state  = n_bhc_multi_state  / n_bhc_total

    mc = bhc[bhc["mean_n_counties"] > 1]
    mean_co_per_mkt   = float(mc["mean_n_counties"].mean())
    median_co_per_mkt = float(mc["mean_n_counties"].median())

    # ── County-pair classification via W_bank_binary ───────────────────────
    # Load county ordering
    co_df = pd.read_csv(CO_PATH)
    # fips5 in county_order_Wgeo.csv is an integer (no leading zero)
    county_order = co_df["fips5"].astype(str).str.zfill(5).tolist()
    N = len(county_order)

    # Build county -> set of BHC IDs mapping (use max-year for representativeness)
    last_yr = sod["year"].max()
    sod_last = sod[sod["year"] == last_yr]
    county_bhc_map: dict[str, set] = (
        sod_last.groupby("fips5")["RSSDHCR"]
        .apply(set)
        .to_dict()
    )

    # BHC -> multi-state flag (use max_n_states across all years)
    bhc_multi = set(bhc.loc[bhc["max_n_states"] > 1, "RSSDHCR"])

    # Analyse W_bank_binary (row-standardised; non-zero = connected)
    W = scipy.sparse.load_npz(str(W_BINARY_PATH)).tocsr()

    # State assignment from county_order
    state_of: list[str] = [c[:2] for c in county_order]

    coo = W.tocoo()
    n_pairs_total        = 0
    n_pairs_cross_state  = 0
    n_pairs_only_multi   = 0   # connected exclusively through multi-state BHCs

    for i, j in zip(coo.row, coo.col):
        if i >= j:     # count each pair once (upper triangle)
            continue
        if W[i, j] == 0 and W[j, i] == 0:
            continue
        n_pairs_total += 1
        ci = county_order[i]
        cj = county_order[j]
        if state_of[i] != state_of[j]:
            n_pairs_cross_state += 1
        # Shared BHC set
        bhcs_i = county_bhc_map.get(ci, set())
        bhcs_j = county_bhc_map.get(cj, set())
        shared  = bhcs_i & bhcs_j
        if shared and all(b in bhc_multi for b in shared):
            n_pairs_only_multi += 1

    share_cross_state = (n_pairs_cross_state / n_pairs_total
                         if n_pairs_total > 0 else float("nan"))
    share_only_multi  = (n_pairs_only_multi / n_pairs_total
                         if n_pairs_total > 0 else float("nan"))

    return pd.DataFrame([{
        "stat": "n_bhc_total",            "value": n_bhc_total},
        {"stat": "n_bhc_multi_county",    "value": n_bhc_multi_county},
        {"stat": "n_bhc_multi_state",     "value": n_bhc_multi_state},
        {"stat": "share_bhc_multi_county","value": round(share_multi_county, 4)},
        {"stat": "share_bhc_multi_state", "value": round(share_multi_state, 4)},
        {"stat": "mean_counties_per_multi_bhc",
                                          "value": round(mean_co_per_mkt, 2)},
        {"stat": "median_counties_per_multi_bhc",
                                          "value": round(median_co_per_mkt, 2)},
        {"stat": "n_county_pairs_connected",
                                          "value": n_pairs_total},
        {"stat": "n_pairs_cross_state",   "value": n_pairs_cross_state},
        {"stat": "share_pairs_cross_state",
                                          "value": round(share_cross_state, 4)},
        {"stat": "n_pairs_only_via_multi_bhc",
                                          "value": n_pairs_only_multi},
        {"stat": "share_pairs_only_via_multi_bhc",
                                          "value": round(share_only_multi, 4)},
    ])


# ── LaTeX table builder ───────────────────────────────────────────────────────

def _stars(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.01:
        return "^{***}"
    if p < 0.05:
        return "^{**}"
    if p < 0.10:
        return "^{*}"
    return ""


def _fmt(x, d=4):
    if pd.isna(x):
        return "--"
    return f"{float(x):.{d}f}"


def _build_latex(results: pd.DataFrame, sample: str) -> str:
    """Build a LaTeX tabular for one sample (Full or Contig)."""
    sub = results[results["sample"] == sample].copy()

    # Canonical outcome ordering
    outcome_order = ["nloans", "vloans", "nden", "lir", "nsold", "nbra"]
    outcome_labels = {
        "nloans": r"\# mortgages originated",
        "vloans": r"Volume mortgages",
        "nden":   r"Denial rate",
        "lir":    r"Loan-to-income ratio",
        "nsold":  r"\# loans sold",
        "nbra":   r"\# branches",
    }
    panel_order  = ["A", "B", "C"]
    panel_labels = {
        "A": r"Panel A: OOS, local branches",
        "B": r"Panel B: OOS, no branches",
        "C": r"Panel C: In-state",
    }

    def _cell(panel_id: str, key: str) -> str:
        r = sub[(sub["panel"] == panel_id) & (sub["outcome_key"] == key)]
        if r.empty or np.isnan(r.iloc[0]["beta"]):
            return r"-- \\ & "
        row   = r.iloc[0]
        b_str = _fmt(row["beta"])
        s_str = _fmt(row["se"])
        st    = _stars(row["pval"])
        p_str = (f"[{_fmt(row['test_vs_A_pval'], 3)}]"
                 if panel_id != "A" and not np.isnan(row["test_vs_A_pval"])
                 else "")
        return f"{b_str}{st} \\\\\n& ({s_str}) {p_str}"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        (r"\caption{Credit Growth and Bank Deregulation by Bank Type and Branch Location"
         + (r" (Contiguous Counties)" if sample == "Contig" else r" (Full Sample)")
         + r"}"),
        (r"\label{tab:bank_location_branches"
         + ("_contig" if sample == "Contig" else "") + r"}"),
        r"\small",
        r"\begin{tabular}{l" + "cc" * len(panel_order) + r"}",
        r"\toprule",
    ]

    # Header
    hdr_cols = " & ".join(
        rf"\multicolumn{{2}}{{c}}{{{panel_labels[p]}}}"
        for p in panel_order
    )
    lines.append(r" & " + hdr_cols + r" \\")
    lines.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}")
    subcols = " & ".join([r"$\hat\beta$ & SE"] * len(panel_order))
    lines.append(r" & " + subcols + r" \\")
    lines.append(r"\midrule")

    for key in outcome_order:
        # Check if any panel has this outcome
        has_outcome = sub[sub["outcome_key"] == key]
        if has_outcome.empty:
            continue
        label = outcome_labels.get(key, key)

        # Get N from Panel A (or whichever panel has it)
        n_vals = {}
        for pid in panel_order:
            r = sub[(sub["panel"] == pid) & (sub["outcome_key"] == key)]
            n_vals[pid] = int(r.iloc[0]["N"]) if not r.empty and r.iloc[0]["N"] > 0 else 0

        # Build row (coeff line)
        coeff_cells = []
        for pid in panel_order:
            r = sub[(sub["panel"] == pid) & (sub["outcome_key"] == key)]
            if r.empty or np.isnan(r.iloc[0]["beta"]):
                coeff_cells.append("--")
                coeff_cells.append("")
            else:
                row  = r.iloc[0]
                bstr = _fmt(row["beta"])
                st_  = _stars(row["pval"])
                coeff_cells.append(f"{bstr}{st_}")
                coeff_cells.append("")

        # Coeff line
        lines.append(rf"\multirow{{2}}{{*}}{{{label}}}"
                     + " & " + " & ".join(coeff_cells) + r" \\")

        # SE / Wald-test line
        se_cells = []
        for pid in panel_order:
            r = sub[(sub["panel"] == pid) & (sub["outcome_key"] == key)]
            if r.empty or np.isnan(r.iloc[0]["beta"]):
                se_cells.append("--")
                se_cells.append("")
            else:
                row    = r.iloc[0]
                sstr   = f"({_fmt(row['se'])})"
                if pid != "A" and not np.isnan(row.get("test_vs_A_pval", float("nan"))):
                    sstr += rf" $[{_fmt(row['test_vs_A_pval'], 3)}]$"
                se_cells.append(sstr)
                se_cells.append("")
        lines.append(r" & " + " & ".join(se_cells) + r" \\[2pt]")

    lines.append(r"\midrule")

    # Observation counts
    for pid in panel_order:
        sub_p = sub[sub["panel"] == pid]
        if sub_p.empty:
            continue
        n_max = int(sub_p["N"].max()) if sub_p["N"].notna().any() else 0
        n_co  = int(sub_p["n_counties"].max()) if sub_p["n_counties"].notna().any() else 0
        lines.append(rf"$N$ obs ({pid}) & \multicolumn{{6}}{{r}}{{{n_max:,}}} \\")
        lines.append(rf"$N$ counties ({pid}) & \multicolumn{{6}}{{r}}{{{n_co:,}}} \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item \textit{Notes:} Panel fixed-effects regression (county FE + year FE).",
        r"Dependent variable: log-change in outcome. Treatment variable: \textit{Linter\_bra}",
        r"(Favara \& Imbs 2015 interstate-branching deregulation index).",
        r"Controls: $\Delta\ln(\text{income})$, $\Delta\ln(\text{pop})$,",
        r"$\Delta\ln(\text{HPI})$, $\Delta\ln(\text{Herfindahl})$ and their lags;",
        r"lagged own DV (when available); year dummies 1995--2005.",
        r"Standard errors (in parentheses) clustered by state.",
        r"Brackets show $p$-value of the test $H_0: \hat\beta_{\text{panel}} = \hat\beta_A$.",
        r"The cross-panel variance $\mathrm{Var}(\hat\beta_{\text{panel}} - \hat\beta_A)$",
        r"is estimated via the SUR-score method (Cameron \& Miller 2015, \S2.5):",
        r"state-level score cross-products from Frisch-Waugh-Lovell residuals",
        r"identify $\mathrm{Cov}(\hat\beta_A, \hat\beta_{\text{panel}})$ without changing",
        r"the individual equation estimates.",
        r"$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.",
        r"\end{tablenotes}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# ── Entry point ───────────────────────────────────────────────────────────────

def run(output_dir: Path) -> None:
    """Run the full bank-location / bank-branches analysis.

    Parameters
    ----------
    output_dir : Path
        Directory where output CSVs are written (e.g. ROOT/output).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading panel data ...")
    df = _load_raw_data()

    # Samples: Full and Contig (border == 1)
    samples = [
        ("Full",   df),
        ("Contig", df[df["border"] == 1].copy()),
    ]

    all_rows: list[dict] = []
    for sample_label, df_s in samples:
        print(f"\n-- {sample_label} (N={len(df_s):,} obs) --")
        rows = _run_sample(df_s, sample_label)
        all_rows.extend(rows)

    results = pd.DataFrame(all_rows)
    col_order = [
        "sample", "panel", "panel_label", "outcome_key", "outcome",
        "beta", "se", "tval", "pval",
        "test_vs_A_pval",
        "N", "n_counties", "n_states", "R2_within",
    ]
    results = results[[c for c in col_order if c in results.columns]]

    csv_path = output_dir / "bank_location_branches.csv"
    results.to_csv(csv_path, index=False)
    print(f"\nResults written to {csv_path}  ({len(results)} rows)")

    # ── LaTeX tables ──────────────────────────────────────────────────────
    for sample in ("Full", "Contig"):
        if sample not in results["sample"].values:
            continue
        tex = _build_latex(results, sample)
        fname = (f"bank_location_branches"
                 + ("_contig" if sample == "Contig" else "") + ".tex")
        tex_path = TABLES_DIR / fname
        tex_path.write_text(tex, encoding="utf-8")
        print(f"LaTeX table written to {tex_path}")

    # ── SOD descriptives ──────────────────────────────────────────────────
    sod_path = output_dir / "bank_network_descriptives.csv"
    if not SOD_PATH.exists():
        print(f"WARNING: SOD data not found at {SOD_PATH}; skipping descriptives.")
    elif not W_BINARY_PATH.exists():
        print(f"WARNING: W_bank_binary.npz not found; skipping pair analysis.")
    else:
        print("\nComputing SOD/network descriptives ...")
        sod_df = _build_sod_descriptives()
        sod_df.to_csv(sod_path, index=False)
        print(f"Descriptives written to {sod_path}")
        for _, row in sod_df.iterrows():
            print(f"  {row['stat']}: {row['value']}")

    # ── Quick verification printout ───────────────────────────────────────
    print("\n-- F&I target verification --")
    for sample in ("Full", "Contig"):
        for key, dep in [("nloans", "Dl_nloans_b_oos_lb"),
                         ("nbra",   "Dl_nbra_oos_lb")]:
            mask = ((results["sample"] == sample)
                    & (results["panel"] == "A")
                    & (results["outcome_key"] == key))
            r = results[mask]
            if r.empty or np.isnan(r.iloc[0]["beta"]):
                continue
            row = r.iloc[0]
            tgt = "~0.16" if key == "nloans" else "~0.10"
            print(f"  {sample} Panel A {key}: "
                  f"beta={row['beta']:.4f}  SE={row['se']:.4f}  "
                  f"N={row['N']:,}  (F&I target {tgt})")


if __name__ == "__main__":
    run(ROOT / "output")
