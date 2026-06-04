"""
Pooled Spatial Error Model (GM) — W_geo vs W_bank

Estimator
---------
Within-demean the panel (two-way: county FE + year FE), stack into a
balanced N*T vector, build a block-diagonal W (T copies of the N x N
cross-sectional matrix), and run spreg.GM_Error.

Two W_bank variants are compared:
  binary : I_{c,h} = 1 if HC h has any branch in county c  (cosine similarity)
  count  : n_{c,h} = number of branches of HC h in county c

Results are printed for full sample and non-border (border == 0) subsample.

Note on lambda = 0.9900
-----------------------
The GM estimator clips lambda at +/-0.99 (soft bound). Values at exactly
0.9900 indicate the true spatial autocorrelation is at or above the bound.
See panel_fe_error.py (ML estimator) for interior estimates.
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
import scipy.sparse
from scipy.sparse import block_diag as sp_block_diag
import spreg

from utils import gal_to_W, row_standardize, sparse_to_pysal_w

ROOT        = Path(__file__).parent.parent
PANEL_PATH  = ROOT / "data" / "estimation_panel.csv"
GAL_PATH    = ROOT / "data" / "W_geo_queen.gal"
COUNTY_PATH = ROOT / "data" / "county_order_Wgeo.csv"
WBANK_BIN   = ROOT / "data" / "W_bank_avg.npz"
WBANK_CNT   = ROOT / "data" / "W_bank_count_avg.npz"

# ── Load shared inputs ────────────────────────────────────────────────────────
co_df        = pd.read_csv(COUNTY_PATH, dtype={"fips5": str})
county_order = co_df["fips5"].str.zfill(5).tolist()
N_ALL        = len(county_order)

W_geo_all, gal_order = gal_to_W(GAL_PATH, county_order)
assert gal_order == county_order, "GAL order != county_order_Wgeo.csv"
print(f"[OK] County order verified  |  N={N_ALL}")

W_bank_bin = row_standardize(scipy.sparse.load_npz(WBANK_BIN))
W_bank_cnt = row_standardize(scipy.sparse.load_npz(WBANK_CNT))

panel = pd.read_csv(PANEL_PATH)
panel["fips5"] = panel["fips5"].astype(str).str.zfill(5)
YEARS    = sorted(panel["year"].unique())
T        = len(YEARS)
year_idx = {yr: i for i, yr in enumerate(YEARS)}
print(f"Panel: {panel.shape[0]:,} rows | {panel['fips5'].nunique()} counties"
      f" | T={T} | {YEARS[0]}-{YEARS[-1]}\n")


# ── Core estimation function ──────────────────────────────────────────────────
def run_sem_pair(panel_sub, W_bank_all, label):
    """
    Within-demean panel_sub, build block-diagonal W, run GM_Error for
    W_geo and W_bank.  Returns (res_geo, res_bank, N_sub, n_obs).
    """
    na_all   = panel_sub.groupby("fips5")["Linter_ela"].apply(lambda s: s.isna().all())
    sub_co   = set(panel_sub["fips5"].unique())
    usable   = [c for c in county_order if c in sub_co and not na_all.get(c, True)]
    N_sub    = len(usable)
    n_obs    = N_sub * T
    usable_pos = {c: i for i, c in enumerate(usable)}

    print(f"  [{label}]  counties={N_sub}  n_obs={n_obs:,}"
          f"  (dropped {N_ALL - N_sub})")

    df = (panel_sub[panel_sub["fips5"].isin(set(usable))]
          .assign(t_idx=lambda d: d["year"].map(year_idx),
                  c_idx=lambda d: d["fips5"].map(usable_pos))
          .sort_values(["t_idx", "c_idx"]))

    y_mat = df["Linter_ela"].values.reshape(T, N_sub)
    x_mat = df["Linter_bra"].values.reshape(T, N_sub)

    def demean(M):
        return M - M.mean(axis=0) - M.mean(axis=1, keepdims=True) + M.mean()

    y_dem = demean(y_mat).reshape(-1, 1)
    x_dem = demean(x_mat).reshape(-1, 1)

    idx        = np.array([county_order.index(c) for c in usable])
    W_geo_sub  = row_standardize(W_geo_all[idx, :][:, idx])
    W_bank_sub = row_standardize(W_bank_all[idx, :][:, idx])

    W_geo_block  = sp_block_diag([W_geo_sub]  * T, format="csr")
    W_bank_block = sp_block_diag([W_bank_sub] * T, format="csr")

    w_geo_p  = sparse_to_pysal_w(W_geo_block)
    w_bank_p = sparse_to_pysal_w(W_bank_block)

    res_geo  = spreg.GM_Error(y_dem, x_dem, w=w_geo_p,
                               name_y="Linter_ela", name_x=["Linter_bra"],
                               name_ds=label)
    res_bank = spreg.GM_Error(y_dem, x_dem, w=w_bank_p,
                               name_y="Linter_ela", name_x=["Linter_bra"],
                               name_ds=label)
    return res_geo, res_bank, N_sub, n_obs


def extract(res, n_co, n_obs):
    """Pull beta, SE, lambda from a GM_Error result.
    betas layout: [const, Linter_bra, lambda]  -- lambda is the last element."""
    return dict(
        beta  = float(res.betas[1, 0]),
        se    = float(res.std_err[1]),
        lam   = float(res.betas[-1, 0]),
        n_co  = n_co,
        n_obs = n_obs,
    )


# ── Run both W_bank variants ──────────────────────────────────────────────────
panel_nb = panel[panel["border"] == 0].copy()

print("=" * 60)
print("BINARY W_bank (HC presence)")
print("=" * 60)
rg_f,  rb_f,  nc_f,  no_f  = run_sem_pair(panel,    W_bank_bin, "Full")
rg_nb, rb_nb, nc_nb, no_nb = run_sem_pair(panel_nb, W_bank_bin, "Non-border")
bin_results = {
    "W_geo  (full)"      : extract(rg_f,  nc_f,  no_f),
    "W_bank (full)"      : extract(rb_f,  nc_f,  no_f),
    "W_geo  (non-border)": extract(rg_nb, nc_nb, no_nb),
    "W_bank (non-border)": extract(rb_nb, nc_nb, no_nb),
}

print("\n" + "=" * 60)
print("COUNT W_bank (branch counts)")
print("=" * 60)
_, rc_f,  _, _ = run_sem_pair(panel,    W_bank_cnt, "Full")
_, rc_nb, _, _ = run_sem_pair(panel_nb, W_bank_cnt, "Non-border")
cnt_results = {
    "W_bank_count (full)"      : extract(rc_f,  nc_f,  no_f),
    "W_bank_count (non-border)": extract(rc_nb, nc_nb, no_nb),
}


# ── Results tables ────────────────────────────────────────────────────────────
HDR = "{:<26s}  {:>10s}  {:>8s}  {:>8s}  {:>9s}  {:>8s}"
ROW = "{:<26s}  {:>10.4f}  {:>8.4f}  {:>8.4f}  {:>9d}  {:>8d}"
SEP = "-" * 76

def print_table(title, results):
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)
    print(HDR.format("Model", "b(dereg)", "SE", "lambda", "N counties", "N obs"))
    print(SEP)
    for name, r in results.items():
        print(ROW.format(name, r["beta"], r["se"], r["lam"], r["n_co"], r["n_obs"]))
    print(SEP)

print_table(
    "MAIN RESULTS  --  Pooled GM_Error, two-way within-demeaned panel",
    bin_results
)
print_table(
    "ROBUSTNESS  --  Count-weighted W_bank  (same W_geo rows as above)",
    {**{k: bin_results[k] for k in ["W_geo  (full)", "W_geo  (non-border)"]},
     **cnt_results}
)

# ── W_bank lambda > W_geo lambda? ────────────────────────────────────────────
print("\nW_bank lambda > W_geo lambda?")
for sample in ("full", "non-border"):
    lg  = bin_results[f"W_geo  ({sample})"]["lam"]
    lb  = bin_results[f"W_bank ({sample})"]["lam"]
    lbc = cnt_results[f"W_bank_count ({sample})"]["lam"]
    tag = "[YES]" if lb > lg else "[NO] "
    print(f"  {sample:<12}: geo={lg:.4f}  bank_bin={lb:.4f}  bank_cnt={lbc:.4f}"
          f"  --> {tag}")
