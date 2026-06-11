"""
panel_fe_hpi.py -- Panel_FE_Error with house-price growth as the DV
===================================================================
Favara-Imbs HPI reduced form estimated under two spatial weights:
  W_geo  : queen contiguity
  W_bank : time-averaged bank holding-company cosine similarity

Structural equation:
  Dl_hpi_it = beta_D * Linter_bra_it
            + beta_Deta * Linter_ela_it
            + controls_it + county FE + year FE + spatial error

Notes
-----
Favara and Imbs weight house-price regressions by the inverse number of
counties per state. spreg.Panel_FE_Error does not natively support observation
weights, so this SEM is estimated unweighted, matching the credit SEM setup.

Outputs
-------
  output/panel_fe_hpi_results.csv
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.stats as stats_mod
import spreg

sys.path.insert(0, str(Path(__file__).parents[1]))
import utils  # noqa: applies spreg Panel_FE_Error compatibility patch
from utils import row_standardize, sparse_to_pysal_w
from panel_data import get_samples
from w_variants import load_w_geo, load_bank_variants


ROOT        = Path(__file__).parents[2]
PANEL_PATH  = ROOT / "data" / "estimation_panel.csv"
COUNTY_PATH = ROOT / "data" / "county_order_Wgeo.csv"

YVAR = "Dl_hpi"
D_VAR = "Linter_bra"
DETA_VAR = "Linter_ela"
CONTROL_CANDIDATES = [
    "LDl_hpi",
    "Dl_pop", "LDl_pop",
    "Dl_inc", "LDl_inc",
    "Dl_hsosf", "LDl_hsosf",
]


def load_panel():
    panel = pd.read_csv(PANEL_PATH)
    panel["fips5"] = panel["fips5"].astype(str).str.zfill(5)
    panel["year"] = panel["year"].astype(int)
    return panel


def build_arrays(panel_sub, county_order, W_all, years, year_pos, xvars,
                 sample_label, w_label):
    """
    Build balanced long-format arrays for Panel_FE_Error.

    The arrays are sorted time-major: all counties in W order for year t,
    then all counties in W order for year t+1. Counties with any missing value
    in y or x over the panel are dropped.
    """
    t = len(years)
    required = [YVAR] + xvars

    any_nan = (
        panel_sub.groupby("fips5")[required]
        .apply(lambda g: g.isna().any().any())
    )
    counts = panel_sub.groupby("fips5")["year"].nunique()
    sub_co = set(panel_sub["fips5"].unique())

    usable = [
        c for c in county_order
        if c in sub_co
        and counts.get(c, 0) == t
        and not any_nan.get(c, True)
    ]

    n = len(usable)
    usable_pos = {c: i for i, c in enumerate(usable)}
    df = (
        panel_sub[panel_sub["fips5"].isin(set(usable))]
        .assign(
            t_idx=lambda d: d["year"].map(year_pos),
            c_idx=lambda d: d["fips5"].map(usable_pos),
        )
        .sort_values(["t_idx", "c_idx"])
    )

    y_long = df[YVAR].values.reshape(-1, 1).astype(float)
    t_idx_vec = df["t_idx"].values
    year_dummies = np.column_stack([
        (t_idx_vec == year_pos[yr]).astype(np.float64)
        for yr in years[1:]
    ])
    x_long = np.hstack([
        df[xvars].values.astype(float),
        year_dummies,
    ])

    name_x = xvars + [f"yr{yr}" for yr in years[1:]]

    assert not np.isnan(y_long).any(), f"NaN in y ({sample_label}, {w_label})"
    assert not np.isnan(x_long).any(), f"NaN in x ({sample_label}, {w_label})"
    assert y_long.shape == (n * t, 1), f"y shape {y_long.shape}"
    assert x_long.shape == (n * t, len(name_x)), f"x shape {x_long.shape}"

    idx = np.array([county_order.index(c) for c in usable])
    W_sub = row_standardize(W_all[idx, :][:, idx])

    return y_long, x_long, sparse_to_pysal_w(W_sub), n, name_x


def run_panel_fe(y, x, w_pysal, name_x, w_label, sample_label):
    return spreg.Panel_FE_Error(
        y, x, w_pysal,
        name_y=YVAR,
        name_x=name_x,
        name_w=w_label,
        name_ds=sample_label,
    )


def extract(res, n_counties, n_obs, name_x):
    idx_d = name_x.index(D_VAR)
    idx_deta = name_x.index(DETA_VAR)
    return {
        "n_counties": n_counties,
        "n_obs": n_obs,
        "lambda": float(res.lam),
        "lambda_se": float(res.std_err[-1]),
        "beta_D": float(res.betas[idx_d, 0]),
        "beta_D_se": float(res.std_err[idx_d]),
        "beta_Deta": float(res.betas[idx_deta, 0]),
        "beta_Deta_se": float(res.std_err[idx_deta]),
    }


def gap_stats(r_geo, r_bank):
    delta = r_bank["lambda"] - r_geo["lambda"]
    se = np.sqrt(r_geo["lambda_se"]**2 + r_bank["lambda_se"]**2)
    z = delta / se
    return float(delta), float(se), float(z)


def run(output_dir=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    co_df        = pd.read_csv(COUNTY_PATH, dtype={"fips5": str})
    county_order = co_df["fips5"].str.zfill(5).tolist()

    W_geo_all, gal_order = load_w_geo(county_order)
    assert gal_order == county_order, "GAL order mismatch"

    bank_variants = load_bank_variants(county_order, W_geo_all=W_geo_all)

    panel    = load_panel()
    years    = sorted(panel["year"].unique())
    year_pos = {yr: i for i, yr in enumerate(years)}

    xvars = [D_VAR, DETA_VAR] + [
        c for c in CONTROL_CANDIDATES if c in panel.columns
    ]
    missing = [c for c in [YVAR, D_VAR, DETA_VAR] if c not in panel.columns]
    if missing:
        raise ValueError(f"Missing required panel columns: {missing}")

    SAMPLE_MAP = get_samples(panel)

    W_MATRICES = [("W_geo", W_geo_all)] + list(bank_variants.items())

    results = {}
    for sample_label, panel_sub in SAMPLE_MAP:
        for w_label, W_all in W_MATRICES:
            print(f"  Estimating: {sample_label} x {w_label} ...", flush=True)
            try:
                y, x, w, n, name_x = build_arrays(
                    panel_sub, county_order, W_all, years, year_pos, xvars,
                    sample_label, w_label,
                )
                res = run_panel_fe(y, x, w, name_x, w_label, sample_label)
                results[(sample_label, w_label)] = extract(
                    res, n, n * len(years), name_x,
                )
            except Exception as exc:
                print(f"  [SKIP] {sample_label} x {w_label}: {exc}")
                results[(sample_label, w_label)] = None

    # ── Build rows ────────────────────────────────────────────────────────────
    rows = []
    for sample_label, _ in SAMPLE_MAP:
        r_geo = results.get((sample_label, "W_geo"))
        for w_label, _ in W_MATRICES:
            r = results.get((sample_label, w_label))
            if r is None:
                continue
            if r_geo is not None and w_label != "W_geo":
                delta, se_delta, z = gap_stats(r_geo, r)
            else:
                delta, se_delta, z = np.nan, np.nan, np.nan
            rows.append({
                "sample": sample_label,
                "w_matrix": w_label,
                **r,
                "delta_lambda": delta,
                "delta_lambda_se": se_delta,
                "z_stat": z,
            })

    # ── Print table ───────────────────────────────────────────────────────────
    W_COL = 108
    print()
    print("=" * W_COL)
    print("Panel_FE_Error -- DV: Dl_hpi (house-price growth)")
    print("Reduced form: Linter_bra + Linter_ela + HPI controls | Two-way FE (ML)")
    print("=" * W_COL)
    print(f"{'Sample':<12} {'W matrix':<22} {'N':>6} "
          f"{'beta_D':>9} {'SE':>7} {'beta_Deta':>10} {'SE':>7} "
          f"{'lambda':>9} {'SE':>7}")
    print("-" * W_COL)

    def stars(p):
        if p is None or np.isnan(p):
            return ""
        return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))

    for row in rows:
        print(f"{row['sample']:<12} {row['w_matrix']:<22} {row['n_counties']:>6} "
              f"{row['beta_D']:>9.4f} {row['beta_D_se']:>7.4f} "
              f"{row['beta_Deta']:>10.4f} {row['beta_Deta_se']:>7.4f} "
              f"{row['lambda']:>9.4f} {row['lambda_se']:>7.4f}")

    print()
    print("Lambda gap (W_bank_* - W_geo), one-sided H1: lambda_bank > lambda_geo")
    print(f"{'Sample':<12} {'W variant':<22} {'delta':>9} {'SE':>8} {'z':>8}")
    print("-" * 62)
    for row in rows:
        if row["w_matrix"] == "W_geo" or np.isnan(row["delta_lambda"]):
            continue
        print(f"{row['sample']:<12} {row['w_matrix']:<22} "
              f"{row['delta_lambda']:>9.4f} {row['delta_lambda_se']:>8.4f} "
              f"{row['z_stat']:>8.3f}")
    print("=" * W_COL)

    if output_dir is not None:
        cols = [
            "sample", "w_matrix", "n_counties", "n_obs",
            "lambda", "lambda_se",
            "beta_D", "beta_D_se",
            "beta_Deta", "beta_Deta_se",
            "delta_lambda", "delta_lambda_se", "z_stat",
        ]
        pd.DataFrame(rows)[cols].to_csv(
            output_dir / "panel_fe_hpi_results.csv", index=False,
        )
        print(f"\nSaved panel_fe_hpi_results.csv to {output_dir}")

    return rows


if __name__ == "__main__":
    run(Path(__file__).parents[2] / "output")
