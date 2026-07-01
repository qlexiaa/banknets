"""
sar_robustness_credit.py
========================
SAR (Spatial Lag) robustness check for the credit-growth SEM result.

Estimates spreg.Panel_FE_Lag on Dl_nloans_b under W_geo and all bank-network
W variants for the Full, Contig, and NonContig samples using the same panel
construction, county filters, and W-loading as sem_credit.py.

Panel_FE_Lag structural equation:
  y = rho * W*y + X*beta + county FE + u
  (spatial autoregressive process in the outcome, vs SEM's error process)

Reports rho, SE(rho), beta_D (Linter_bra coefficient), SE(beta_D),
and the gap delta_rho = rho_bank - rho_geo with combined SE and one-sided
z-statistic (same formula as the lambda gap in sem_credit.py).

Side-by-side comparison with the SEM lambdas is printed at the end.

Output: output/sar_robustness_credit.csv
Columns: sample, w_matrix, n_counties, n_obs, rho, rho_se,
         beta_D, beta_D_se, delta_rho, delta_rho_se, z_stat
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import scipy.sparse
import scipy.stats as stats_mod
import spreg

sys.path.insert(0, str(Path(__file__).parents[1]))
import utils  # noqa: applies spreg Panel_FE_Lag compatibility patch
from utils import row_standardize, sparse_to_pysal_w
from panel_data import CREDIT_CONTROLS, load_panel_with_credit, get_samples
from w_variants import load_w_geo, load_bank_variants

ROOT           = Path(__file__).parents[2]
COUNTY_PATH    = ROOT / "data" / "county_order_Wgeo.csv"
CREDIT_SEM_CSV = ROOT / "output" / "panel_fe_credit_results.csv"
X_VARS         = ["Linter_bra"] + CREDIT_CONTROLS


def run_panel_fe_lag(y, x, w_pysal, w_label, ds_label, YEARS):
    nx = X_VARS + [f"yr{yr}" for yr in YEARS[1:]]
    return spreg.Panel_FE_Lag(
        y, x, w_pysal,
        name_y="Dl_nloans_b",
        name_x=nx,
        name_w=w_label,
        name_ds=ds_label,
    )


def extract_lag(res, N, T):
    """Extract rho and beta_D from a Panel_FE_Lag result.

    betas layout (Panel_FE_Lag): [beta_Linter_bra, controls..., yr..., rho]
    rho is last in betas, std_err, and z_stat — same convention as lambda
    in Panel_FE_Error.
    """
    beta_D = float(res.betas[0, 0])
    rho    = float(res.rho)                # spatial lag parameter
    se_b   = float(res.std_err[0])
    se_r   = float(res.std_err[-1])
    z_b, p_b = res.z_stat[0]
    z_r, p_r = res.z_stat[-1]
    return dict(
        beta_D=beta_D, se_beta_D=se_b, z_beta_D=float(z_b), p_beta_D=float(p_b),
        rho=rho,       rho_se=se_r,    z_rho=float(z_r),    p_rho=float(p_r),
        n_co=N,        n_obs=N * T,
    )


def run(output_dir=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load shared inputs ───────────────────────────────────────────────────
    co_df        = pd.read_csv(COUNTY_PATH, dtype={"fips5": str})
    county_order = co_df["fips5"].str.zfill(5).tolist()

    W_geo_all, gal_order = load_w_geo(county_order)
    assert gal_order == county_order, "GAL order mismatch"

    bank_variants = load_bank_variants(county_order, W_geo_all=W_geo_all)

    panel    = load_panel_with_credit()
    panel["fips5"] = panel["fips5"].astype(str).str.zfill(5)
    YEARS    = sorted(panel["year"].unique())
    T        = len(YEARS)
    year_pos = {yr: i for i, yr in enumerate(YEARS)}

    bank_matrix_order = [
        "W_bank",
        "W_bank_count",
        "W_bank_binary",
        "W_bank_knn3",
        "W_bank_knn4",
        "W_bank_count_knn3",
        "W_bank_count_knn4",
        "W_bank_binary_knn3",
        "W_bank_binary_knn4",
        "W_bank_nonGeo",
        "W_bank_interstate",
        "W_bank_intrastate",
    ]
    missing = [w for w in bank_matrix_order if w not in bank_variants]
    if missing:
        raise KeyError(f"load_bank_variants() did not return: {missing}")

    BANK_MATRICES = [(w, bank_variants[w]) for w in bank_matrix_order]
    W_MATRICES = [("W_geo", W_geo_all)] + BANK_MATRICES

    # ── Build arrays for one (panel_sub, W_all) combination ──────────────────
    def _usable_counties(panel_sub, W_all):
        DV      = "Dl_nloans_b"
        any_nan = panel_sub.groupby("fips5")[[DV] + X_VARS].apply(
            lambda g: g.isna().any().any()
        )
        sub_co  = set(panel_sub["fips5"].unique())
        full_rs = np.array(W_all.sum(axis=1)).flatten()
        islands = {county_order[i] for i, r in enumerate(full_rs) if r == 0}
        return [c for c in county_order
                if c in sub_co and not any_nan.get(c, True) and c not in islands]

    def _build(panel_sub, W_all, sample_label, usable=None):
        DV      = "Dl_nloans_b"
        usable  = list(usable) if usable is not None else _usable_counties(panel_sub, W_all)
        N       = len(usable)
        u_pos   = {c: i for i, c in enumerate(usable)}
        df = (panel_sub[panel_sub["fips5"].isin(set(usable))]
              .assign(t_idx=lambda d: d["year"].map(year_pos),
                      c_idx=lambda d: d["fips5"].map(u_pos))
              .sort_values(["t_idx", "c_idx"]))
        y_long       = df[DV].values.reshape(-1, 1)
        t_idx_vec    = df["t_idx"].values
        year_dummies = np.column_stack([
            (t_idx_vec == year_pos[yr]).astype(np.float64) for yr in YEARS[1:]])
        x_long  = np.hstack([df[X_VARS].values.astype(np.float64), year_dummies])
        assert not np.isnan(y_long).any()
        assert not np.isnan(x_long).any()
        idx     = np.array([county_order.index(c) for c in usable])
        W_sub   = row_standardize(W_all[idx, :][:, idx])
        return y_long, x_long, sparse_to_pysal_w(W_sub), N, tuple(usable)

    # ── Estimate ──────────────────────────────────────────────────────────────
    results = {}
    for sample_label, panel_sub in get_samples(panel):
        for w_name, W_all in BANK_MATRICES:
            print(f"Estimating SAR ({sample_label}, {w_name}) ...", flush=True)
            try:
                y, x, w, N, usable = _build(panel_sub, W_all, sample_label)
                res = run_panel_fe_lag(y, x, w, w_name, sample_label, YEARS)
                results[(sample_label, w_name)] = extract_lag(res, N, T)
                results[(sample_label, w_name)]["_usable_counties"] = usable
            except Exception as exc:
                print(f"  [SKIP] {sample_label} x {w_name}: {exc}")
                results[(sample_label, w_name)] = None

        base_bank = results.get((sample_label, "W_bank"))
        base_usable = None if base_bank is None else base_bank["_usable_counties"]
        print(f"Estimating SAR ({sample_label}, W_geo on W_bank counties) ...", flush=True)
        try:
            y, x, w, N, usable = _build(
                panel_sub, W_geo_all, sample_label, usable=base_usable)
            res = run_panel_fe_lag(y, x, w, "W_geo", sample_label, YEARS)
            results[(sample_label, "W_geo")] = extract_lag(res, N, T)
            results[(sample_label, "W_geo")]["_usable_counties"] = usable
        except Exception as exc:
            print(f"  [SKIP] {sample_label} x W_geo: {exc}")
            results[(sample_label, "W_geo")] = None

    paired_geo = {}
    paired_geo_cache = {
        (sample_label, r["_usable_counties"]): r
        for sample_label, _ in get_samples(panel)
        for r in [results.get((sample_label, "W_geo"))]
        if r is not None
    }
    for sample_label, panel_sub in get_samples(panel):
        for w_name, _ in BANK_MATRICES:
            r_bank = results.get((sample_label, w_name))
            if r_bank is None:
                paired_geo[(sample_label, w_name)] = None
                continue
            cache_key = (sample_label, r_bank["_usable_counties"])
            if cache_key not in paired_geo_cache:
                print(f"Estimating paired SAR W_geo ({sample_label}, {w_name} sample) ...", flush=True)
                y, x, w, N, usable = _build(
                    panel_sub, W_geo_all, sample_label, usable=r_bank["_usable_counties"])
                res = run_panel_fe_lag(y, x, w, "W_geo", sample_label, YEARS)
                r_geo = extract_lag(res, N, T)
                r_geo["_usable_counties"] = usable
                paired_geo_cache[cache_key] = r_geo
            paired_geo[(sample_label, w_name)] = paired_geo_cache[cache_key]

    # ── Gap statistics (same formula as lambda gap in sem_credit.py) ─────
    def gap_stat(r_geo, r_bank, one_sided=True):
        gap    = r_bank["rho"]   - r_geo["rho"]
        se_gap = np.sqrt(r_geo["rho_se"]**2 + r_bank["rho_se"]**2)
        z      = gap / se_gap
        p      = float(stats_mod.norm.sf(z)) if one_sided else float(2 * stats_mod.norm.sf(abs(z)))
        return gap, se_gap, z, p

    def stars(p):
        return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))

    # ── Print table ───────────────────────────────────────────────────────────
    W = 86
    print()
    print("=" * W)
    print("Panel_FE_Lag (SAR) -- DV: Dl_nloans_b | Robustness vs SEM result")
    print("Regressors: " + ", ".join(X_VARS) + " | Two-way FE")
    print("=" * W)
    print(f"{'Sample':<10} {'W':<22} {'N':>6}  "
          f"{'rho':>8} {'SE':>6}  {'beta_D':>8} {'SE':>6}")
    print("-" * W)

    w_keys = ["W_geo"] + bank_matrix_order
    for sample_label, _ in get_samples(panel):
        for w_name in w_keys:
            r = results.get((sample_label, w_name))
            if r is None:
                print(f"{sample_label:<10} {w_name:<22}  SKIP")
                continue
            sr = stars(r["p_rho"])
            sb = stars(r["p_beta_D"])
            print(f"{sample_label:<10} {w_name:<22} {r['n_co']:>6}  "
                  f"{r['rho']:>8.4f}{sr} {r['rho_se']:>6.4f}  "
                  f"{r['beta_D']:>8.4f}{sb} {r['se_beta_D']:>6.4f}")
        print()

    print("Significance: *** p<0.01  ** p<0.05  * p<0.10")

    # ── Build CSV rows ─────────────────────────────────────────────────────────
    csv_rows = []
    for sample_label, _ in get_samples(panel):
        for w_name in w_keys:
            r = results.get((sample_label, w_name))
            if r is None:
                continue
            r_geo = r if w_name == "W_geo" else paired_geo.get((sample_label, w_name))
            if r_geo is not None and w_name != "W_geo":
                gap_val, se_gap_val, z_gap_val, _ = gap_stat(r_geo, r)
            else:
                gap_val, se_gap_val, z_gap_val = np.nan, np.nan, np.nan
            csv_rows.append(dict(
                sample     = sample_label,
                w_matrix   = w_name,
                n_counties = r["n_co"],
                n_obs      = r["n_obs"],
                rho        = r["rho"],
                rho_se     = r["rho_se"],
                beta_D     = r["beta_D"],
                beta_D_se  = r["se_beta_D"],
                delta_rho    = gap_val,
                delta_rho_se = se_gap_val,
                z_stat       = z_gap_val,
            ))

    if output_dir is not None:
        cols = ["sample", "w_matrix", "n_counties", "n_obs",
                "rho", "rho_se", "beta_D", "beta_D_se",
                "delta_rho", "delta_rho_se", "z_stat"]
        pd.DataFrame(csv_rows)[cols].to_csv(
            output_dir / "sar_robustness_credit.csv", index=False)
        print(f"\nSaved sar_robustness_credit.csv to {output_dir}")

    return results, csv_rows


if __name__ == "__main__":
    run(Path(__file__).parents[2] / "output")
