"""
W_bank_nonGeo: bank transmission beyond geographic adjacency
============================================================
Builds a residual weight matrix by zeroing all W_bank_avg entries
between counties that are also geographic neighbours in W_geo.
What remains are purely non-geographic bank connections.

If lambda under W_bank_nonGeo is positive and significant, bank
networks transmit credit shocks through channels that geography
alone cannot capture.
"""
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import scipy.sparse
import pandas as pd
import spreg

import utils  # noqa: applies spreg Panel_FE_Error compatibility patch
from utils import gal_to_W, row_standardize, sparse_to_pysal_w

ROOT        = Path(__file__).parent.parent
PANEL_PATH  = ROOT / 'data' / 'estimation_panel.csv'
GAL_PATH    = ROOT / 'data' / 'W_geo_queen.gal'
COUNTY_PATH = ROOT / 'data' / 'county_order_Wgeo.csv'
WBANK_PATH  = ROOT / 'data' / 'W_bank_avg.npz'

LAM_GEO  = dict(full=0.8268, nb=0.8316)
LAM_BANK = dict(full=0.9879, nb=0.9873)


def sparsity_stats(arr, N):
    off = int(np.count_nonzero(arr)) - int(np.count_nonzero(np.diag(arr)))
    tot = N * (N - 1)
    return 1 - off / tot, off / N, off


def run(output_dir=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load & verify order ────────────────────────────────────────────────
    co_df        = pd.read_csv(COUNTY_PATH, dtype={'fips5': str})
    county_order = co_df['fips5'].str.zfill(5).tolist()
    N            = len(county_order)

    W_geo_sparse, gal_order = gal_to_W(GAL_PATH, county_order)
    assert gal_order == county_order, 'GAL order != county_order_Wgeo.csv'

    W_bank_sparse = row_standardize(scipy.sparse.load_npz(WBANK_PATH))
    assert W_bank_sparse.shape == (N, N), \
        f'W_bank shape {W_bank_sparse.shape} != ({N},{N})'

    # ── 2. Build W_bank_nonGeo ────────────────────────────────────────────────
    W_bank_arr = W_bank_sparse.toarray()
    W_geo_arr  = W_geo_sparse.toarray()
    np.fill_diagonal(W_bank_arr, 0.0)
    np.fill_diagonal(W_geo_arr,  0.0)

    W_nonGeo = W_bank_arr.copy()
    W_nonGeo[W_geo_arr > 0] = 0.0   # zero entries where geo neighbours
    np.fill_diagonal(W_nonGeo, 0.0)

    # Row-standardise
    rs = W_nonGeo.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        W_nonGeo_rs = np.where(rs > 0, W_nonGeo / rs, 0.0)

    W_nonGeo_sp = scipy.sparse.csr_matrix(W_nonGeo_rs)

    # ── 3. Sparsity diagnostics ───────────────────────────────────────────────
    sp_geo,  avg_geo,  nz_geo  = sparsity_stats(W_geo_arr,  N)
    sp_bank, avg_bank, nz_bank = sparsity_stats(W_bank_arr, N)
    sp_ng,   avg_ng,   nz_ng   = sparsity_stats(W_nonGeo,   N)

    # ── 4. Islands ────────────────────────────────────────────────────────────
    row_sums  = W_nonGeo.sum(axis=1)
    island_idx = np.where(row_sums == 0)[0]
    island_fips = {county_order[i] for i in island_idx}

    # ── 5. Panel arrays ───────────────────────────────────────────────────────
    panel = pd.read_csv(PANEL_PATH)
    panel['fips5'] = panel['fips5'].astype(str).str.zfill(5)
    YEARS    = sorted(panel['year'].unique())
    T        = len(YEARS)
    year_pos = {yr: i for i, yr in enumerate(YEARS)}

    def build_arrays(panel_sub, label):
        na_all = panel_sub.groupby('fips5')['Linter_ela'].apply(
            lambda s: s.isna().all())
        sub_co = set(panel_sub['fips5'].unique())
        usable = [c for c in county_order
                  if c in sub_co
                  and not na_all.get(c, True)
                  and c not in island_fips]
        N_u   = len(usable)
        u_pos = {c: i for i, c in enumerate(usable)}

        df = (panel_sub[panel_sub['fips5'].isin(set(usable))]
              .assign(t_idx=lambda d: d['year'].map(year_pos),
                      c_idx=lambda d: d['fips5'].map(u_pos))
              .sort_values(['t_idx', 'c_idx']))

        y   = df['Linter_ela'].values.reshape(-1, 1)
        t_v = df['t_idx'].values
        yd  = np.column_stack(
            [(t_v == year_pos[yr]).astype(float) for yr in YEARS[1:]])
        x   = np.hstack([df['Linter_bra'].values.reshape(-1, 1), yd])

        idx   = np.array([county_order.index(c) for c in usable])
        W_sub = row_standardize(
            scipy.sparse.csr_matrix(W_nonGeo[idx, :][:, idx]))

        w_pysal = sparse_to_pysal_w(W_sub)
        return y, x, w_pysal, N_u

    # ── 6. Estimation ─────────────────────────────────────────────────────────
    y_f,  x_f,  w_f,  N_f  = build_arrays(panel, 'Full')
    y_nb, x_nb, w_nb, N_nb = build_arrays(
        panel[panel['border'] == 0].copy(), 'Non-border')

    def run_fe(y, x, w):
        nx  = ['Linter_bra'] + [f'yr{yr}' for yr in YEARS[1:]]
        res = spreg.Panel_FE_Error(
            y, x, w, name_y='Linter_ela', name_x=nx)
        return dict(
            beta = float(res.betas[0, 0]),
            se_b = float(res.std_err[0]),
            p_b  = float(res.z_stat[0][1]),
            lam  = float(res.lam),
            se_l = float(res.std_err[-1]),
            p_l  = float(res.z_stat[-1][1]),
        )

    res_f  = run_fe(y_f,  x_f,  w_f)
    res_nb = run_fe(y_nb, x_nb, w_nb)

    # ── 7. Save outputs ───────────────────────────────────────────────────────
    if output_dir is not None:
        model_rows = [
            dict(model='W_bank_nonGeo (full)',
                 beta=res_f['beta'],  se_b=res_f['se_b'],  p_b=res_f['p_b'],
                 lam=res_f['lam'],    se_l=res_f['se_l'],  p_l=res_f['p_l']),
            dict(model='W_bank_nonGeo (non-border)',
                 beta=res_nb['beta'], se_b=res_nb['se_b'], p_b=res_nb['p_b'],
                 lam=res_nb['lam'],   se_l=res_nb['se_l'], p_l=res_nb['p_l']),
        ]
        pd.DataFrame(model_rows).to_csv(
            output_dir / "bank_nongeo_results.csv", index=False)

        matrix_rows = [
            dict(matrix='W_geo',         sparsity=sp_geo,  avg_nbrs=avg_geo,  n_pairs=nz_geo),
            dict(matrix='W_bank_avg',     sparsity=sp_bank, avg_nbrs=avg_bank, n_pairs=nz_bank),
            dict(matrix='W_bank_nonGeo',  sparsity=sp_ng,   avg_nbrs=avg_ng,   n_pairs=nz_ng),
        ]
        pd.DataFrame(matrix_rows).to_csv(
            output_dir / "bank_nongeo_matrix_stats.csv", index=False)

    return dict(res_f=res_f, res_nb=res_nb,
                sp_geo=sp_geo, sp_bank=sp_bank, sp_ng=sp_ng,
                avg_geo=avg_geo, avg_bank=avg_bank, avg_ng=avg_ng)


if __name__ == '__main__':
    run(Path(__file__).parent.parent / 'output')
