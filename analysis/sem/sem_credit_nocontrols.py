"""
sem_credit_nocontrols.py
=========================
Panel_FE_Error credit-growth robustness with no county controls.

The displayed W_geo rows are estimated on the exact base W_bank county set for
each sample, matching the main controlled SEM outputs.

Output: output/panel_fe_credit_results_nocontrols.csv
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
import sem_credit as base
from panel_data import load_panel_with_credit, get_samples
from w_variants import load_w_geo, load_bank_variants

ROOT        = Path(__file__).parents[2]
COUNTY_PATH = ROOT / "data" / "county_order_Wgeo.csv"
X_VARS      = ["Linter_bra"]


def run(output_dir=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    old_x_vars = base.X_VARS
    base.X_VARS = X_VARS
    try:
        co_df        = pd.read_csv(COUNTY_PATH, dtype={"fips5": str})
        county_order = co_df["fips5"].str.zfill(5).tolist()

        W_geo_all, gal_order = load_w_geo(county_order)
        assert gal_order == county_order, "GAL order mismatch"

        bank_variants = load_bank_variants(county_order, W_geo_all=W_geo_all)

        panel = load_panel_with_credit()
        panel["fips5"] = panel["fips5"].astype(str).str.zfill(5)
        years    = sorted(panel["year"].unique())
        T        = len(years)
        year_pos = {yr: i for i, yr in enumerate(years)}
        samples  = get_samples(panel)

        results = {}
        for sample_label, panel_sub in samples:
            for w_name, W_all in bank_variants.items():
                print(f"  Estimating no-controls: {sample_label} x {w_name} ...", flush=True)
                try:
                    y, x, w, N, usable = base._build_arrays_variant(
                        panel_sub, county_order, W_all, years, year_pos,
                        sample_label, w_name)
                    res = base.run_panel_fe(y, x, w, w_name, sample_label, years)
                    results[(sample_label, w_name)] = base.extract(res, N, T)
                    results[(sample_label, w_name)]["_usable_counties"] = usable
                except Exception as exc:
                    print(f"  [SKIP] {sample_label} x {w_name}: {exc}")
                    results[(sample_label, w_name)] = None

            base_bank = results.get((sample_label, "W_bank"))
            base_usable = None if base_bank is None else base_bank["_usable_counties"]
            print(
                f"  Estimating no-controls: {sample_label} x W_geo on W_bank counties ...",
                flush=True,
            )
            try:
                y, x, w, N, usable = base._build_arrays_variant(
                    panel_sub, county_order, W_geo_all, years, year_pos,
                    sample_label, "W_geo", usable=base_usable)
                res = base.run_panel_fe(y, x, w, "W_geo", sample_label, years)
                results[(sample_label, "W_geo")] = base.extract(res, N, T)
                results[(sample_label, "W_geo")]["_usable_counties"] = usable
            except Exception as exc:
                print(f"  [SKIP] {sample_label} x W_geo: {exc}")
                results[(sample_label, "W_geo")] = None

        if output_dir is not None:
            rows = []
            w_keys = ["W_geo"] + list(bank_variants.keys())
            for sample_label, _ in samples:
                for w_name in w_keys:
                    r = results.get((sample_label, w_name))
                    if r is None:
                        continue
                    rows.append(dict(
                        model    = f"FE {w_name} ({sample_label.lower()})",
                        sample   = sample_label,
                        w_matrix = w_name,
                        **{k: r[k] for k in
                           ["beta", "se_beta", "z_beta", "p_beta",
                            "lam", "se_lam", "z_lam", "p_lam",
                            "n_co", "n_obs"]},
                    ))

            out_path = output_dir / "panel_fe_credit_results_nocontrols.csv"
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(f"\nSaved panel_fe_credit_results_nocontrols.csv to {output_dir}")

        return results
    finally:
        base.X_VARS = old_x_vars


if __name__ == "__main__":
    run(ROOT / "output")
