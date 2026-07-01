"""
jtest.py -- Davidson & MacKinnon (1981) J-test for competing spatial weights
============================================================================
Tests whether the spatial prediction component from one W (the alternative)
has significant explanatory power when added to a model estimated under another
W (the null). Both directions are run for each pair.

Mechanism (verified from spreg source):
  res.u          -- (N*T, 1) within-demeaned OLS residuals
  res.e_filtered -- (N*T, 1) spatially filtered residuals = (I - lam*W_block) @ u
  => m_hat = res.u - res.e_filtered  = lam * W_block @ u  (spatial prediction component)

J-test direction (H0=W_null, H1=W_alt):
  1. Estimate Panel_FE_Error with W_alt → m_alt = get_m_hat(res_alt)
  2. x_aug = hstack([x_long, m_alt])
  3. Estimate Panel_FE_Error with W_null on (y, x_aug)
  4. J-coeff = res_aug.betas[-2, 0],  J-SE = res_aug.std_err[-2],
     J-z, J-p = res_aug.z_stat[-2]   -- two-sided

Both directions are run; result classified as:
  'W_alt preferred'  -- only D1 rejects
  'W_null preferred' -- only D2 rejects
  'complementary'    -- both reject
  'indeterminate'    -- neither rejects

Pairs tested: W_geo vs all bank-network variants.
Samples:      Full, Contig, NonContig

Output: output/jtest_credit_results.csv
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse
import spreg

import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
import utils  # noqa: applies spreg compatibility patch
from panel_data import CREDIT_CONTROLS, load_panel_with_credit
from utils import row_standardize, sparse_to_pysal_w
from panel_data import get_samples
from w_variants import load_w_geo, load_bank_variants

ROOT        = Path(__file__).parents[2]
COUNTY_PATH = ROOT / "data" / "county_order_Wgeo.csv"
DV          = "Dl_nloans_b"
X_VARS      = ["Linter_bra"] + CREDIT_CONTROLS

SAMPLE_ORDER = ["Full", "Contig", "NonContig"]
ALT_MATRICES = [
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


# ── Core helpers ─────────────────────────────────────────────────────────────

def get_m_hat(res):
    """Extract the spatial prediction component from a Panel_FE_Error result.

    m_hat = lam * W_block @ u  = res.u - res.e_filtered
    Shape: (N*T, 1), time-major, same ordering as y_long.
    """
    return res.u - res.e_filtered


def run_jtest_direction(y_long, x_long, m_alt, w_null_pysal,
                        w_null_label, w_alt_label, sample_label, YEARS):
    """Run one J-test direction: H₀ = W_null, H₁ = W_alt.

    Augments x_long with m_alt (the alt model's spatial prediction component)
    and re-estimates Panel_FE_Error under W_null. The J-statistic is the
    coefficient on m_alt (second-to-last in betas; last is lambda).

    Parameters
    ----------
    y_long        : (N*T, 1) outcome array
    x_long        : (N*T, k) regressor array (Linter_bra + controls + year dummies)
    m_alt         : (N*T, 1) spatial prediction from the alternative model
    w_null_pysal  : PySAL W object for the null specification
    w_null_label  : string label for W_null
    w_alt_label   : string label for W_alt
    sample_label  : string label for the sample
    YEARS         : list of years, sorted

    Returns
    -------
    dict with j_coeff, j_se, j_z, j_p (two-sided), reject (p < 0.05)
    """
    x_aug = np.hstack([x_long, m_alt])
    nx    = X_VARS + [f"yr{yr}" for yr in YEARS[1:]] + ["m_alt"]

    res_aug = spreg.Panel_FE_Error(
        y_long, x_aug, w_null_pysal,
        name_y=DV,
        name_x=nx,
        name_w=w_null_label,
        name_ds=f"{sample_label}|J({w_null_label}<-{w_alt_label})",
    )

    # m_alt coefficient: second-to-last (last is lambda)
    j_coeff = float(res_aug.betas[-2, 0])
    j_se    = float(res_aug.std_err[-2])
    j_z     = float(res_aug.z_stat[-2][0])
    j_p     = float(res_aug.z_stat[-2][1])

    return dict(j_coeff=j_coeff, j_se=j_se, j_z=j_z, j_p=j_p, reject=(j_p < 0.05))


# ── Pair runner ───────────────────────────────────────────────────────────────

def run_pair(panel_sub, county_order, W_null_all, W_alt_all,
             w_null_label, w_alt_label, sample_label, YEARS, year_pos):
    """Run both J-test directions for one (W_null, W_alt) pair and sample.

    Uses the intersection of valid counties under both W matrices so that
    m_hat from the alt model has the same shape as y_long / x_long for
    the null-model augmented regression.

    County filtering:
      1. Counties present in panel_sub with no missing Dl_nloans_b or controls
      2. Full-matrix islands (zero rows) in EITHER W are excluded
    """
    T = len(YEARS)

    # Islands in the full matrices, using the same filtering as sem_w_variants.py.
    full_rs_null = np.array(W_null_all.sum(axis=1)).flatten()
    full_rs_alt  = np.array(W_alt_all.sum(axis=1)).flatten()
    islands = (
        {county_order[i] for i, r in enumerate(full_rs_null) if r == 0} |
        {county_order[i] for i, r in enumerate(full_rs_alt)  if r == 0}
    )

    any_nan = panel_sub.groupby("fips5")[[DV] + X_VARS].apply(
        lambda g: g.isna().any().any()
    )
    sub_co = set(panel_sub["fips5"].unique())
    usable = [
        c for c in county_order
        if c in sub_co and not any_nan.get(c, True) and c not in islands
    ]
    N          = len(usable)
    usable_pos = {c: i for i, c in enumerate(usable)}

    df = (
        panel_sub[panel_sub["fips5"].isin(set(usable))]
        .assign(
            t_idx=lambda d: d["year"].map(year_pos),
            c_idx=lambda d: d["fips5"].map(usable_pos),
        )
        .sort_values(["t_idx", "c_idx"])
    )

    y_long       = df[DV].values.reshape(-1, 1)
    t_idx_vec    = df["t_idx"].values
    year_dummies = np.column_stack([
        (t_idx_vec == year_pos[yr]).astype(np.float64)
        for yr in YEARS[1:]
    ])
    x_long = np.hstack([df[X_VARS].values.astype(np.float64), year_dummies])

    assert not np.isnan(y_long).any(), f"NaN in y ({sample_label})"
    assert not np.isnan(x_long).any(), f"NaN in X ({sample_label})"
    assert y_long.shape == (N * T, 1),  f"y shape {y_long.shape}"
    assert x_long.shape == (N * T, len(X_VARS) + len(YEARS) - 1), f"x shape {x_long.shape}"

    # Common W submatrices, row-standardised
    idx          = np.array([county_order.index(c) for c in usable])
    w_null_pysal = sparse_to_pysal_w(row_standardize(W_null_all[idx, :][:, idx]))
    w_alt_pysal  = sparse_to_pysal_w(row_standardize(W_alt_all[idx, :][:, idx]))

    # ── Base estimates (both W, same county set) ──────────────────────────────
    nx = X_VARS + [f"yr{yr}" for yr in YEARS[1:]]

    res_null = spreg.Panel_FE_Error(
        y_long, x_long, w_null_pysal,
        name_y=DV, name_x=nx,
        name_w=w_null_label, name_ds=sample_label,
    )
    res_alt = spreg.Panel_FE_Error(
        y_long, x_long, w_alt_pysal,
        name_y=DV, name_x=nx,
        name_w=w_alt_label, name_ds=sample_label,
    )

    m_null = get_m_hat(res_null)   # spatial prediction under null
    m_alt  = get_m_hat(res_alt)    # spatial prediction under alt

    # ── Direction 1: H₀ = W_null, H₁ = W_alt ────────────────────────────────
    print(f"      D1 H0={w_null_label} ...", end=" ", flush=True)
    d1 = run_jtest_direction(
        y_long, x_long, m_alt, w_null_pysal,
        w_null_label, w_alt_label, sample_label, YEARS,
    )
    print("done")

    # ── Direction 2: H₀ = W_alt, H₁ = W_null ────────────────────────────────
    print(f"      D2 H0={w_alt_label} ...", end=" ", flush=True)
    d2 = run_jtest_direction(
        y_long, x_long, m_null, w_alt_pysal,
        w_alt_label, w_null_label, sample_label, YEARS,
    )
    print("done")

    # ── Conclusion ────────────────────────────────────────────────────────────
    if d1["reject"] and not d2["reject"]:
        conclusion = "W_alt preferred"
    elif d2["reject"] and not d1["reject"]:
        conclusion = "W_null preferred"
    elif d1["reject"] and d2["reject"]:
        conclusion = "complementary"
    else:
        conclusion = "indeterminate"

    return dict(
        sample     = sample_label,
        pair       = f"{w_null_label} vs {w_alt_label}",
        w_null     = w_null_label,
        w_alt      = w_alt_label,
        n_counties = N,
        d1_j_coeff = d1["j_coeff"], d1_j_se = d1["j_se"],
        d1_j_z     = d1["j_z"],     d1_j_p  = d1["j_p"],  d1_reject = d1["reject"],
        d2_j_coeff = d2["j_coeff"], d2_j_se = d2["j_se"],
        d2_j_z     = d2["j_z"],     d2_j_p  = d2["j_p"],  d2_reject = d2["reject"],
        conclusion = conclusion,
    )


# ── Master run ────────────────────────────────────────────────────────────────

def run(output_dir=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load shared inputs ────────────────────────────────────────────────────
    co_df        = pd.read_csv(COUNTY_PATH, dtype={"fips5": str})
    county_order = co_df["fips5"].str.zfill(5).tolist()

    W_geo_all, gal_order = load_w_geo(county_order)
    assert gal_order == county_order, "GAL order mismatch"

    bank_variants = load_bank_variants(county_order, W_geo_all=W_geo_all)

    panel    = load_panel_with_credit()
    panel["fips5"] = panel["fips5"].astype(str).str.zfill(5)
    YEARS    = sorted(panel["year"].unique())
    year_pos = {yr: i for i, yr in enumerate(YEARS)}

    missing = [w for w in ALT_MATRICES if w not in bank_variants]
    if missing:
        raise KeyError(f"load_bank_variants() did not return: {missing}")

    # W_geo as null, each bank variant as alternative
    PAIRS = [
        ("W_geo", W_geo_all, w_name, W_alt)
        for w_name, W_alt in ((w, bank_variants[w]) for w in ALT_MATRICES)
    ]

    # ── Run every bank-variant pair × 3 samples ──────────────────────────────
    sample_lookup = dict(get_samples(panel))
    missing_samples = [s for s in SAMPLE_ORDER if s not in sample_lookup]
    if missing_samples:
        raise KeyError(f"get_samples() did not return: {missing_samples}")

    rows = []
    for sample_label in SAMPLE_ORDER:
        panel_sub = sample_lookup[sample_label]
        for w_null_label, W_null_all, w_alt_label, W_alt_all in PAIRS:
            print(f"  [{sample_label}] {w_null_label} vs {w_alt_label}")
            r = run_pair(
                panel_sub, county_order, W_null_all, W_alt_all,
                w_null_label, w_alt_label, sample_label, YEARS, year_pos,
            )
            rows.append(r)

    # ── Formatted table ───────────────────────────────────────────────────────
    def stars(p):
        return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else "   "))

    W = 108
    print()
    print("=" * W)
    print(
        "J-test (Davidson & MacKinnon 1981): "
        f"W_geo vs all {len(ALT_MATRICES)} bank weight matrices"
    )
    print("D1: H0=W_null, H1=W_alt  |  D2: H0=W_alt, H1=W_null  |  two-sided z")
    print("=" * W)
    print(
        f"{'Sample':<14} {'Pair':<33} {'N':>5}  "
        f"{'D1 J-coeff':>12} {'z':>7} {'p':>8}  "
        f"{'D2 J-coeff':>12} {'z':>7} {'p':>8}  "
        f"{'Conclusion'}"
    )
    print("-" * W)

    for r in rows:
        s1   = stars(r["d1_j_p"])
        s2   = stars(r["d2_j_p"])
        pair = f"{r['w_null']} vs {r['w_alt']}"
        print(
            f"{r['sample']:<14} {pair:<33} {r['n_counties']:>5}  "
            f"{r['d1_j_coeff']:>12.4f}{s1} {r['d1_j_z']:>7.3f} {r['d1_j_p']:>8.4f}  "
            f"{r['d2_j_coeff']:>12.4f}{s2} {r['d2_j_z']:>7.3f} {r['d2_j_p']:>8.4f}  "
            f"{r['conclusion']}"
        )

    print()
    print("Significance: *** p<0.01  ** p<0.05  * p<0.10")
    print("W_alt preferred:      D1 rejects H0=W_null, D2 does not reject H0=W_alt")
    print("W_null preferred:     D2 rejects H0=W_alt,  D1 does not reject H0=W_null")
    print("complementary:        both models contain independent spatial information")
    print("indeterminate:        insufficient power to distinguish specifications")
    print("=" * W)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if output_dir is not None:
        cols = [
            "sample", "pair", "w_null", "w_alt", "n_counties",
            "d1_j_coeff", "d1_j_se", "d1_j_z", "d1_j_p", "d1_reject",
            "d2_j_coeff", "d2_j_se", "d2_j_z", "d2_j_p", "d2_reject",
            "conclusion",
        ]
        df_out = pd.DataFrame(rows, columns=cols)
        expected_pairs = {(sample, w_alt) for sample in SAMPLE_ORDER for w_alt in ALT_MATRICES}
        observed_pairs = set(zip(df_out["sample"], df_out["w_alt"]))
        if len(df_out) != len(expected_pairs) or observed_pairs != expected_pairs:
            missing_pairs = sorted(expected_pairs - observed_pairs)
            extra_pairs = sorted(observed_pairs - expected_pairs)
            raise RuntimeError(
                f"Expected {len(expected_pairs)} J-test rows; got {len(df_out)}. "
                f"Missing={missing_pairs}, extra={extra_pairs}"
            )
        df_out.to_csv(
            output_dir / "jtest_credit_results.csv", index=False
        )
        print(f"\nSaved jtest_credit_results.csv to {output_dir}")

    return rows


if __name__ == '__main__':
    run(Path(__file__).parents[2] / 'output')
