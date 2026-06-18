"""
Panel_FE_Error estimates for restricted bank-network links.

Runs the credit-growth SEM against three restrictions of W_bank_avg:
non-geographic links, interstate links, and intrastate links. Each restriction
is compared with the W_geo benchmark for the full and border-MSA samples
(Favara & Imbs 2015, Appendix Tables A1/A2).

Outputs
-------
  output/bank_nongeo_credit_results.csv
  output/bank_nongeo_matrix_stats.csv
  output/bank_interstate_credit_results.csv
  output/bank_interstate_sparsity.csv
  output/bank_intrastate_credit_results.csv
  output/bank_intrastate_sparsity.csv
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse
import scipy.stats as stats
import spreg

sys.path.insert(0, str(Path(__file__).parents[1]))
import utils  # noqa: applies spreg compatibility patch
from panel_data import CREDIT_CONTROLS, load_panel_with_credit
from utils import row_standardize, sparse_to_pysal_w
from w_variants import load_w_geo, load_bank_variants


ROOT        = Path(__file__).parents[2]
COUNTY_PATH = ROOT / "data" / "county_order_Wgeo.csv"

DV = "Dl_nloans_b"
X_VARS = ["Linter_bra"] + CREDIT_CONTROLS


def build_wbank_nongeo(W_bank_sp, county_order, W_geo_sp):
    W_ng = W_bank_sp.toarray().astype(np.float64)
    W_geo = W_geo_sp.toarray().astype(np.float64)
    np.fill_diagonal(W_ng, 0.0)
    np.fill_diagonal(W_geo, 0.0)
    W_ng[W_geo > 0] = 0.0
    np.fill_diagonal(W_ng, 0.0)
    return row_standardize(scipy.sparse.csr_matrix(W_ng))


def build_wbank_interstate(W_bank_sp, county_order):
    states = np.array([c[:2] for c in county_order])
    W = W_bank_sp.toarray().astype(np.float64)
    np.fill_diagonal(W, 0.0)
    same_state = states[:, None] == states[None, :]
    np.fill_diagonal(same_state, False)
    W[same_state] = 0.0
    np.fill_diagonal(W, 0.0)
    return row_standardize(scipy.sparse.csr_matrix(W))


def build_wbank_intrastate(W_bank_sp, county_order):
    states = np.array([c[:2] for c in county_order])
    W = W_bank_sp.toarray().astype(np.float64)
    np.fill_diagonal(W, 0.0)
    diff_state = states[:, None] != states[None, :]
    np.fill_diagonal(diff_state, False)
    W[diff_state] = 0.0
    np.fill_diagonal(W, 0.0)
    return row_standardize(scipy.sparse.csr_matrix(W))


def sparsity_stats(W_sp, label):
    N = W_sp.shape[0]
    W_arr = W_sp.toarray()
    np.fill_diagonal(W_arr, 0.0)

    n_links = int(np.count_nonzero(W_arr))
    possible = N * (N - 1)
    density = n_links / possible
    sparsity = 1.0 - density
    avg_nbrs = n_links / N
    row_sums = W_sp.sum(axis=1).A1
    n_islands = int((row_sums == 0).sum())

    print(f"  {label:<25}  N={N}  links={n_links:>9,}  "
          f"density={density:.6f}  sparsity={sparsity:.6f}  "
          f"avg_nbrs={avg_nbrs:6.2f}  islands={n_islands}")

    return dict(
        label=label,
        N=N,
        n_links=n_links,
        possible=possible,
        density=density,
        sparsity=sparsity,
        avg_nbrs=avg_nbrs,
        n_islands=n_islands,
    )


def nongeo_matrix_stats(W_sp, label):
    N = W_sp.shape[0]
    W_arr = W_sp.toarray()
    np.fill_diagonal(W_arr, 0.0)
    n_pairs = int(np.count_nonzero(W_arr))
    possible = N * (N - 1)
    return dict(
        matrix=label,
        sparsity=1.0 - (n_pairs / possible),
        avg_nbrs=n_pairs / N,
        n_pairs=n_pairs,
    )


def usable_counties(panel_sub, county_order, W_all):
    any_nan = panel_sub.groupby("fips5")[[DV] + X_VARS].apply(
        lambda g: g.isna().any().any()
    )
    sub_co = set(panel_sub["fips5"].unique())

    full_rs = np.array(W_all.sum(axis=1)).flatten()
    islands = {county_order[i] for i, r in enumerate(full_rs) if r == 0}
    usable = [
        c for c in county_order
        if c in sub_co and not any_nan.get(c, True) and c not in islands
    ]
    return usable


def build_arrays(panel_sub, county_order, W_all, years, year_pos, sample_label,
                 usable=None):
    T = len(years)
    usable = list(usable) if usable is not None else usable_counties(
        panel_sub, county_order, W_all)
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

    y_long = df[DV].values.reshape(-1, 1)
    t_idx_vec = df["t_idx"].values
    year_dummies = np.column_stack([
        (t_idx_vec == year_pos[yr]).astype(np.float64)
        for yr in years[1:]
    ])
    x_long = np.hstack([df[X_VARS].values.astype(np.float64), year_dummies])

    assert not np.isnan(y_long).any(), f"NaN in y ({sample_label})"
    assert not np.isnan(x_long).any(), f"NaN in X ({sample_label})"
    assert y_long.shape == (N * T, 1), f"y shape mismatch: {y_long.shape}"
    assert x_long.shape == (N * T, len(X_VARS) + len(years) - 1), f"x shape mismatch: {x_long.shape}"

    idx = np.array([county_order.index(c) for c in usable])
    W_sub = row_standardize(W_all[idx, :][:, idx])
    return y_long, x_long, sparse_to_pysal_w(W_sub), N, tuple(usable)


def run_fe(y, x, w, w_label, ds_label, years):
    nx = X_VARS + [f"yr{yr}" for yr in years[1:]]
    return spreg.Panel_FE_Error(
        y, x, w,
        name_y=DV,
        name_x=nx,
        name_w=w_label,
        name_ds=ds_label,
    )


def extract(res, N, T, model_name):
    return dict(
        model=model_name,
        beta=float(res.betas[0, 0]),
        se_beta=float(res.std_err[0]),
        z_beta=float(res.z_stat[0][0]),
        p_beta=float(res.z_stat[0][1]),
        lam=float(res.lam),
        se_lam=float(res.std_err[-1]),
        z_lam=float(res.z_stat[-1][0]),
        p_lam=float(res.z_stat[-1][1]),
        n_co=N,
        n_obs=N * T,
    )


def stars(p):
    return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))


def gap_test(r_geo, r_alt):
    gap = r_alt["lam"] - r_geo["lam"]
    se_gap = np.sqrt(r_geo["se_lam"]**2 + r_alt["se_lam"]**2)
    z_gap = gap / se_gap
    p_gap = float(stats.norm.sf(z_gap))
    return gap, se_gap, z_gap, p_gap


def load_or_build(cache_path, label, build_func, W_bank_raw, county_order):
    if cache_path.exists():
        print(f"Loading cached {label} ...")
        return row_standardize(scipy.sparse.load_npz(cache_path))

    print(f"Building {label} ...")
    W_variant = build_func(W_bank_raw, county_order)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    scipy.sparse.save_npz(str(cache_path), W_variant)
    print(f"  Saved {label} -> {cache_path}")
    return W_variant


def estimate_variant(config, W_geo_all, W_bank_raw, W_variant_all,
                     panel, panel_contig, panel_noncontig, county_order, years, year_pos):
    T = len(years)
    rows = []
    print(f"\nEstimating {config['label']} link restriction ...")

    for sample_label, panel_sub in [
        ("Full",      panel),
        ("Contig",    panel_contig),
        ("NonContig", panel_noncontig),
    ]:
        w_label = config["w_label"]
        print(f"  {sample_label} x {w_label} ...", flush=True)
        y_alt, x_alt, w_alt, N_alt, usable = build_arrays(
            panel_sub, county_order, W_variant_all, years, year_pos, sample_label)
        res_alt = run_fe(y_alt, x_alt, w_alt, w_label, sample_label, years)

        print(f"  {sample_label} x W_geo (paired {w_label} sample) ...", flush=True)
        y_geo, x_geo, w_geo, N_geo, _ = build_arrays(
            panel_sub, county_order, W_geo_all, years, year_pos, sample_label,
            usable=usable)
        res_geo = run_fe(y_geo, x_geo, w_geo, "W_geo", sample_label, years)

        rows.append(extract(res_geo, N_geo, T, f"W_geo ({sample_label.lower()})"))
        rows.append(extract(res_alt, N_alt, T, f"{w_label} ({sample_label.lower()})"))

    print()
    print("=" * 90)
    print(f"Panel_FE_Error -- DV: {DV} | {config['w_label']} vs W_geo")
    print("=" * 90)
    print(f"{'Model':<38} {'N':>6}  {'beta':>8} {'SE':>6}  "
          f"{'lambda':>8} {'SE':>6}  {'p_lam':>8}")
    print("-" * 90)
    for r in rows:
        print(f"{r['model']:<38} {r['n_co']:>6}  "
              f"{r['beta']:>8.4f}{stars(r['p_beta'])} {r['se_beta']:>6.4f}  "
              f"{r['lam']:>8.4f}{stars(r['p_lam'])} {r['se_lam']:>6.4f}  "
              f"{r['p_lam']:>8.4f}")

    print()
    print(f"Lambda gap ({config['w_label']} vs W_geo):")
    for sample_label in ["Full", "Contig", "NonContig"]:
        r_geo = next((r for r in rows if r["model"] == f"W_geo ({sample_label.lower()})"), None)
        r_alt = next((r for r in rows if r["model"] == f"{config['w_label']} ({sample_label.lower()})"), None)
        if r_geo is None or r_alt is None:
            continue
        gap, se_gap, z_gap, p_gap = gap_test(r_geo, r_alt)
        print(f"  {sample_label:<14}: gap={gap:+.4f}  SE={se_gap:.4f}  "
              f"z={z_gap:.3f}  p={p_gap:.4f}{stars(p_gap)}")

    print("Note: SE(gap) ignores covariance between estimators.")
    return rows


def run(output_dir=None):
    output_dir = Path(output_dir or (ROOT / "output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    co_df        = pd.read_csv(COUNTY_PATH, dtype={"fips5": str})
    county_order = co_df["fips5"].str.zfill(5).tolist()

    W_geo_all, gal_order = load_w_geo(county_order)
    assert gal_order == county_order, "GAL order mismatch"

    # Load all bank variants (builds + caches any missing matrices)
    bank_variants = load_bank_variants(county_order, W_geo_all=W_geo_all)
    W_bank_raw    = bank_variants["W_bank"]

    variants = [
        {
            "key": "nongeo",
            "label": "nonGeo",
            "w_label": "W_bank_nonGeo",
            "W_variant": bank_variants["W_bank_nonGeo"],
            "results_csv": "bank_nongeo_credit_results.csv",
            "sparsity_csv": "bank_nongeo_matrix_stats.csv",
            "sparsity_kind": "nongeo",
        },
        {
            "key": "interstate",
            "label": "interstate",
            "w_label": "W_bank_interstate",
            "W_variant": bank_variants["W_bank_interstate"],
            "results_csv": "bank_interstate_credit_results.csv",
            "sparsity_csv": "bank_interstate_sparsity.csv",
            "sparsity_kind": "standard",
        },
        {
            "key": "intrastate",
            "label": "intrastate",
            "w_label": "W_bank_intrastate",
            "W_variant": bank_variants["W_bank_intrastate"],
            "results_csv": "bank_intrastate_credit_results.csv",
            "sparsity_csv": "bank_intrastate_sparsity.csv",
            "sparsity_kind": "standard",
        },
    ]

    panel = load_panel_with_credit()
    panel["fips5"] = panel["fips5"].astype(str).str.zfill(5)
    years = sorted(panel["year"].unique())
    year_pos = {yr: i for i, yr in enumerate(years)}
    panel_contig    = panel[panel["border"] == 1].copy()
    panel_noncontig = panel[panel["border"] == 0].copy()

    all_results = {}
    for config in variants:
        W_variant_all = config["W_variant"]

        print("\nSparsity diagnostics:")
        if config["sparsity_kind"] == "nongeo":
            sparse_rows = [
                nongeo_matrix_stats(W_geo_all, "W_geo"),
                nongeo_matrix_stats(W_bank_raw, "W_bank_avg"),
                nongeo_matrix_stats(W_variant_all, config["w_label"]),
            ]
        else:
            sparse_rows = [
                sparsity_stats(W_geo_all, "W_geo"),
                sparsity_stats(W_bank_raw, "W_bank (full)"),
                sparsity_stats(W_variant_all, config["w_label"]),
            ]

        rows = estimate_variant(
            config, W_geo_all, W_bank_raw, W_variant_all,
            panel, panel_contig, panel_noncontig, county_order, years, year_pos,
        )

        result_cols = [
            "model", "beta", "se_beta", "z_beta", "p_beta",
            "lam", "se_lam", "z_lam", "p_lam", "n_co", "n_obs",
        ]
        pd.DataFrame(rows)[result_cols].to_csv(
            output_dir / config["results_csv"], index=False
        )
        pd.DataFrame(sparse_rows).to_csv(
            output_dir / config["sparsity_csv"], index=False
        )
        print(f"\nSaved: {output_dir / config['results_csv']}")
        print(f"Saved: {output_dir / config['sparsity_csv']}")

        all_results[config["key"]] = (rows, sparse_rows)

    return all_results


if __name__ == "__main__":
    run(Path(__file__).parents[2] / "output")
