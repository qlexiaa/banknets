"""
spatial_multiplier_decomposition.py
=====================================
Spatial multiplier decomposition following LeSage & Pace (2009),
adapted to the Spatial Error Model (Panel_FE_Error).

Under the SEM:  u = lambda * W * u + eps
the reduced-form impact matrix is S = (I - lambda*W)^{-1}:
  S[c, c']  = total effect of a unit shock at c' that reaches county c
             through the spatial error process.

Power-series expansion:
  S = I + lambda*W + lambda^2*W^2 + ...
  k=0 (I)        : direct / own-county retention
  k>=1 (lam^k Wk): k-th order network transmission

NOTE: In the SEM, shocks propagate through the *error covariance*, not the
outcome itself (that would require a SAR model). S quantifies statistical
co-movement in the disturbance process, not structural causal transmission.

Outputs
-------
  output/spatial_multiplier_decomposition.csv
  output/spatial_multiplier_decay.csv
  output/hub_counties.csv
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.sparse
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from utils import row_standardize
from w_variants import load_w_geo, load_bank_variants
from panel_data import load_panel_with_credit

ROOT             = Path(__file__).parents[2]
COUNTY_PATH      = ROOT / "data" / "county_order_Wgeo.csv"
CREDIT_CSV       = ROOT / "output" / "panel_fe_credit_results.csv"
CENTROID_CACHE   = ROOT / "data" / "county_centroids.csv"

DECAY_THRESHOLD  = 1e-4    # stop power series when contribution < this
DECAY_MAX_K      = 600     # hard cap on expansion order
HUB_TOP_N        = 10      # number of hub counties to report

# Minimal FIPS-2 → state name lookup
STATE_NAMES = {
    "01":"Alabama","02":"Alaska","04":"Arizona","05":"Arkansas",
    "06":"California","08":"Colorado","09":"Connecticut","10":"Delaware",
    "11":"DC","12":"Florida","13":"Georgia","15":"Hawaii",
    "16":"Idaho","17":"Illinois","18":"Indiana","19":"Iowa",
    "20":"Kansas","21":"Kentucky","22":"Louisiana","23":"Maine",
    "24":"Maryland","25":"Massachusetts","26":"Michigan","27":"Minnesota",
    "28":"Mississippi","29":"Missouri","30":"Montana","31":"Nebraska",
    "32":"Nevada","33":"New Hampshire","34":"New Jersey","35":"New Mexico",
    "36":"New York","37":"North Carolina","38":"North Dakota","39":"Ohio",
    "40":"Oklahoma","41":"Oregon","42":"Pennsylvania","44":"Rhode Island",
    "45":"South Carolina","46":"South Dakota","47":"Tennessee","48":"Texas",
    "49":"Utah","50":"Vermont","51":"Virginia","53":"Washington",
    "54":"West Virginia","55":"Wisconsin","56":"Wyoming",
}


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_lambdas():
    """Read lambda estimates from panel_fe_credit_results.csv for all samples."""
    lam = {}
    cr  = pd.read_csv(CREDIT_CSV)

    for sample_tag in ["full", "contig", "noncontig"]:
        for w_tag in ["W_geo", "W_bank", "W_bank_knn3", "W_bank_knn4"]:
            mask = (cr["model"].str.contains(w_tag, regex=False) &
                    cr["model"].str.contains(f"({sample_tag})", regex=False))
            rows = cr.loc[mask, "lam"]
            if len(rows) > 0:
                lam[("credit", w_tag, sample_tag)] = float(rows.iloc[0])

    return lam


def load_centroids(county_order):
    """
    Return (lats, lons) arrays aligned to county_order.
    Downloads Census 2020 population-weighted county centroids on first call
    and caches to data/county_centroids.csv. Returns None on failure.
    """
    if CENTROID_CACHE.exists():
        df = pd.read_csv(CENTROID_CACHE, dtype={"fips5": str})
    else:
        try:
            url = ("https://www2.census.gov/geo/docs/reference/cenpop2020/"
                   "county/CenPop2020_Mean_CO.txt")
            raw = pd.read_csv(url)
            df = pd.DataFrame({
                "fips5": (raw["STATEFP"].astype(str).str.zfill(2) +
                          raw["COUNTYFP"].astype(str).str.zfill(3)),
                "lat": raw["LATITUDE"].astype(float),
                "lon": raw["LONGITUDE"].astype(float),
            })
            df.to_csv(CENTROID_CACHE, index=False)
            print("  Centroid cache written to data/county_centroids.csv")
        except Exception as exc:
            print(f"  [WARN] Centroid download failed ({exc}). "
                  "Reporting network-hop reach instead of geographic reach.")
            return None

    lkp  = df.set_index("fips5")[["lat", "lon"]]
    lats = np.array([lkp.loc[c, "lat"] if c in lkp.index else np.nan
                     for c in county_order])
    lons = np.array([lkp.loc[c, "lon"] if c in lkp.index else np.nan
                     for c in county_order])
    n_missing = int(np.isnan(lats).sum())
    if n_missing:
        print(f"  [WARN] {n_missing} counties missing centroids; filled with median.")
        lats = np.where(np.isnan(lats), np.nanmedian(lats), lats)
        lons = np.where(np.isnan(lons), np.nanmedian(lons), lons)
    return lats, lons


# ── Matrix helpers ────────────────────────────────────────────────────────────

def haversine_matrix(lats, lons):
    """Vectorised N×N great-circle distance matrix in kilometres."""
    R     = 6371.0
    la    = np.radians(lats)
    lo    = np.radians(lons)
    dlat  = la[:, None] - la[None, :]
    dlon  = lo[:, None] - lo[None, :]
    a     = (np.sin(dlat / 2) ** 2
             + np.cos(la[:, None]) * np.cos(la[None, :]) * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def hop_reach_from_sparse(W_sp, S_off):
    """
    Fallback reach metric (network hops) when geographic centroids unavailable.
    Uses BFS shortest-path distances in the binary W adjacency graph.
    hop_reach_c = weighted average hop distance from c to all reachable c'.
    """
    N     = W_sp.shape[0]
    # Binary adjacency (ignore weights, just topology)
    A     = (W_sp > 0).astype(np.float32)
    # scipy shortest_path on unweighted graph → hop distances
    hops  = scipy.sparse.csgraph.shortest_path(A, method="D", directed=False)
    hops  = np.where(np.isinf(hops), 0.0, hops)   # treat disconnected as 0

    off_row_sums = S_off.sum(axis=1)
    denom = np.where(off_row_sums > 0, off_row_sums, 1.0)
    reach = (S_off * hops).sum(axis=1) / denom
    return reach


# ── Core decomposition ────────────────────────────────────────────────────────

def drop_zero_rows(W_sp, county_order):
    """
    Remove counties whose row in W sums to zero (structural islands with no
    connections under this W specification). Returns the subsetted sparse
    matrix and the correspondingly filtered county_order list.
    """
    rs   = np.array(W_sp.sum(axis=1)).flatten()
    keep = np.where(rs > 0)[0]
    if len(keep) == W_sp.shape[0]:
        return W_sp, county_order          # nothing to drop
    W_sub  = W_sp[keep, :][:, keep].tocsr()
    co_sub = [county_order[i] for i in keep]
    print(f"  Dropped {W_sp.shape[0] - len(keep)} zero-row island counties "
          f"({W_sp.shape[0]} -> {len(keep)})")
    return W_sub, co_sub


def compute_inverse(W_dense, lam):
    """
    Compute S = (I - lam*W)^{-1} and verify the analytical row-sum identity.
    Requires W to be exactly row-standardised (all row sums = 1).
    """
    N        = W_dense.shape[0]
    S        = scipy.linalg.inv(np.eye(N) - lam * W_dense)
    analytic = 1.0 / (1.0 - lam)
    numeric  = S.sum(axis=1).mean()
    rel_err  = abs(numeric - analytic) / abs(analytic)
    if rel_err > 1e-4:
        print(f"  [WARN] Row-sum verification: numeric={numeric:.6f}, "
              f"analytic={analytic:.6f}, rel_err={rel_err:.2e}")
    else:
        print(f"  Verification OK: row-sum mean={numeric:.6f}  "
              f"(analytic {analytic:.6f}, rel_err={rel_err:.1e})")
    return S, analytic


def decay_series(W_sp, lam, threshold=DECAY_THRESHOLD, max_k=DECAY_MAX_K):
    """
    Contribution of each order k to average off-diagonal transmission.

    contribution_k = lam^k * mean_{i≠j}([W^k]_{ij})
                   = lam^k * (N - trace(W^k)) / (N*(N-1))

    trace(W^k) is computed via eigenvalues (O(N) per k after one O(N^3) eigval solve),
    making this efficient even for large k needed when lambda is close to 1.
    """
    N     = W_sp.shape[0]
    eigs  = np.linalg.eigvals(W_sp.toarray().astype(np.float64))

    rows  = []
    cumul = 0.0
    for k in range(1, max_k + 1):
        trace_k      = float(np.real(eigs ** k).sum())
        mean_off      = (N - trace_k) / (N * (N - 1))
        contrib       = (lam ** k) * mean_off
        cumul        += contrib
        rows.append({"order_k": k, "contribution": contrib, "_cumul": cumul})
        if k >= 3 and contrib < threshold:
            break

    # Normalise cumulative share to fraction of total decay (excluding k=0 own-effect)
    total = cumul
    for r in rows:
        r["cumulative_share"] = r["_cumul"] / total if total > 1e-15 else 0.0
        del r["_cumul"]

    return pd.DataFrame(rows)


def hub_counties(S, county_order, top_n=HUB_TOP_N):
    """
    Top `top_n` sender counties (largest off-diagonal row sums)
    and top `top_n` receiver counties (largest off-diagonal column sums).
    """
    S_off = S.copy()
    np.fill_diagonal(S_off, 0.0)

    row_tx  = S_off.sum(axis=1)    # outgoing: total shock sent to others
    col_rx  = S_off.sum(axis=0)    # incoming: total shock received from others

    records = []
    for direction, vals in [("sender", row_tx), ("receiver", col_rx)]:
        top_idx = np.argsort(vals)[::-1][:top_n]
        for rank, idx in enumerate(top_idx, 1):
            fips = county_order[idx]
            state = STATE_NAMES.get(fips[:2], "??")
            records.append({
                "direction": direction,
                "rank": rank,
                "fips5": fips,
                "state": state,
                "total_transmission": float(vals[idx]),
            })
    return pd.DataFrame(records)


# ── Master run ────────────────────────────────────────────────────────────────

def run(output_dir=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load shared inputs ────────────────────────────────────────────────────
    print("Loading lambda values ...")
    lam_dict = load_lambdas()
    for k, v in lam_dict.items():
        print(f"  {k[0]:8s} {k[1]:8s}  lambda = {v:.6f}  "
              f"total multiplier = {1/(1-v):.4f}x")

    print("\nLoading W matrices and panel ...")
    co_df        = pd.read_csv(COUNTY_PATH, dtype={"fips5": str})
    county_order = co_df["fips5"].str.zfill(5).tolist()
    N            = len(county_order)
    print(f"  N = {N} counties (full matrix)")

    W_geo_sp, _ = load_w_geo(county_order)
    bank_vars   = load_bank_variants(county_order, W_geo_all=W_geo_sp)
    W_bank_sp   = bank_vars["W_bank"]
    W_knn3_sp   = bank_vars["W_bank_knn3"]
    W_knn4_sp   = bank_vars["W_bank_knn4"]

    W_MATS = {
        "W_geo":       W_geo_sp,
        "W_bank":      W_bank_sp,
        "W_bank_knn3": W_knn3_sp,
        "W_bank_knn4": W_knn4_sp,
    }

    panel = load_panel_with_credit()
    panel["fips5"] = panel["fips5"].astype(str).str.zfill(5)

    # Usable county set per sample: drop any-NaN counties for each sub-panel
    def _usable_counties(panel_sub):
        any_nan = panel_sub.groupby("fips5")["Dl_nloans_b"].apply(
            lambda s: s.isna().any())
        sub_co = set(panel_sub["fips5"].unique())
        return [c for c in county_order
                if c in sub_co and not any_nan.get(c, True)]

    SAMPLE_COUNTIES = {
        "full":     _usable_counties(panel),
        "contig":   _usable_counties(panel[panel["border"] == 1]),
        "noncontig":_usable_counties(panel[panel["border"] == 0]),
    }
    for stag, co_list in SAMPLE_COUNTIES.items():
        print(f"  {stag}: {len(co_list)} usable counties")

    print("\nLoading / downloading county centroids ...")
    centroid_result = load_centroids(county_order)
    if centroid_result is not None:
        lats, lons   = centroid_result
        dist_matrix  = haversine_matrix(lats, lons)
        np.fill_diagonal(dist_matrix, 0.0)
        reach_label  = "km"
        print(f"  Distance matrix ready ({dist_matrix.shape}), "
              f"mean={dist_matrix[dist_matrix > 0].mean():.0f} km")
    else:
        dist_matrix  = None
        reach_label  = "hops"

    # ── Run decomposition for each combination ────────────────────────────────
    summary_rows = []
    all_decay    = []
    all_hubs     = []

    COMBOS = [
        ("credit", "W_geo",        "full"),
        ("credit", "W_bank",       "full"),
        ("credit", "W_bank_knn3",  "full"),
        ("credit", "W_bank_knn4",  "full"),
        ("credit", "W_geo",        "contig"),
        ("credit", "W_bank",       "contig"),
        ("credit", "W_bank_knn3",  "contig"),
        ("credit", "W_bank_knn4",  "contig"),
        ("credit", "W_geo",        "noncontig"),
        ("credit", "W_bank",       "noncontig"),
        ("credit", "W_bank_knn3",  "noncontig"),
        ("credit", "W_bank_knn4",  "noncontig"),
    ]

    for outcome, w_label, sample_tag in COMBOS:
        key = (outcome, w_label, sample_tag)
        if key not in lam_dict:
            print(f"  [SKIP] lambda not found for {key}")
            continue
        lam     = lam_dict[key]
        W_sp    = W_MATS[w_label]
        co_samp = SAMPLE_COUNTIES.get(sample_tag, county_order)

        print(f"\n{'='*60}")
        print(f"  {outcome.upper()} | {w_label} | {sample_tag}  lambda={lam:.6f}")
        print(f"{'='*60}")

        # Subset W to the sample counties first (so the centroid/reach lookup
        # uses the correct county set), then drop zero-row structural islands.
        # This is the fix for missing avg-reach in sample-specific combos:
        # previously drop_zero_rows used the full county_order, so for contig/
        # noncontig the reach was computed over all non-island counties rather
        # than just those in the sample.
        idx_samp    = np.array([county_order.index(c) for c in co_samp])
        W_sp_samp   = W_sp[idx_samp, :][:, idx_samp].tocsr()
        W_sp_sub, co_sub = drop_zero_rows(W_sp_samp, co_samp)
        N_sub   = len(co_sub)
        W_dense = W_sp_sub.toarray().astype(np.float64)
        row_sums = W_dense.sum(axis=1, keepdims=True)
        nonzero_rows = row_sums[:, 0] > 0
        W_dense[nonzero_rows] /= row_sums[nonzero_rows]

        # ── 1. Inverse ────────────────────────────────────────────────────────
        print(f"  Computing (I - lambda*W)^{{-1}} on {N_sub}x{N_sub} matrix ...")
        S, analytic_total = compute_inverse(W_dense, lam)
        diag_S            = np.diag(S)
        S_off             = S.copy(); np.fill_diagonal(S_off, 0.0)

        avg_direct   = float(diag_S.mean())
        avg_total    = float(S.sum(axis=1).mean())
        avg_indirect = avg_total - avg_direct
        indir_share  = avg_indirect / avg_total * 100.0

        print(f"  Verification: row-sum mean = {S.sum(axis=1).mean():.6f}  "
              f"(analytic 1/(1-lam) = {analytic_total:.6f})")
        print(f"  Total multiplier : {avg_total:.4f}x")
        print(f"  Avg direct       : {avg_direct:.4f}  "
              f"({avg_direct/avg_total*100:.1f}% of total)")
        print(f"  Avg indirect     : {avg_indirect:.4f}  "
              f"({indir_share:.1f}% of total)")

        # ── 2. Geographic / hop reach ─────────────────────────────────────────
        if dist_matrix is not None:
            # Align distance matrix to co_sub (subset of county_order)
            # county_order is the full master list; dist_matrix is also aligned to it
            idx_sub  = np.array([county_order.index(c) for c in co_sub])
            dist_sub = dist_matrix[np.ix_(idx_sub, idx_sub)]
            off_row_sums = S_off.sum(axis=1)
            denom        = np.where(off_row_sums > 0, off_row_sums, 1.0)
            reach_c      = (S_off * dist_sub).sum(axis=1) / denom
        else:
            print("  Computing network-hop reach (geographic fallback) ...")
            reach_c = hop_reach_from_sparse(W_sp_sub, S_off)

        avg_reach = float(reach_c.mean())
        med_reach = float(np.median(reach_c))
        print(f"  Avg reach        : {avg_reach:.1f} {reach_label}  "
              f"(median {med_reach:.1f})")

        # ── 3. Decay series ───────────────────────────────────────────────────
        print("  Computing decay series (eigenvalue method) ...")
        df_decay = decay_series(W_sp_sub, lam)
        n_terms  = len(df_decay)
        cumul_at_5  = float(df_decay[df_decay["order_k"] <= 5]["contribution"].sum()
                            / df_decay["contribution"].sum()) * 100
        print(f"  Decay: {n_terms} terms to threshold {DECAY_THRESHOLD:.0e}. "
              f"First 5 orders capture {cumul_at_5:.1f}% of indirect transmission.")
        df_decay.insert(0, "sample",   sample_tag)
        df_decay.insert(0, "w_matrix", w_label)
        df_decay.insert(0, "outcome",  outcome)
        all_decay.append(df_decay)

        # ── 4. Hub counties ───────────────────────────────────────────────────
        df_hubs = hub_counties(S, co_sub)
        df_hubs.insert(0, "sample",   sample_tag)
        df_hubs.insert(0, "w_matrix", w_label)
        df_hubs.insert(0, "outcome",  outcome)
        all_hubs.append(df_hubs)

        top_snd = df_hubs[df_hubs["direction"] == "sender"].head(5)
        top_rcv = df_hubs[df_hubs["direction"] == "receiver"].head(5)
        print("  Top 5 senders  (off-diag row sum):",
              ", ".join(f"{r.fips5}({r.state[:2]})" for r in top_snd.itertuples()))
        print("  Top 5 receivers (off-diag col sum):",
              ", ".join(f"{r.fips5}({r.state[:2]})" for r in top_rcv.itertuples()))

        summary_rows.append({
            "outcome":         outcome,
            "w_matrix":        w_label,
            "sample":          sample_tag,
            "lambda":          lam,
            "total_multiplier": avg_total,
            "avg_direct":      avg_direct,
            "avg_indirect":    avg_indirect,
            "indirect_share_pct": indir_share,
            f"avg_reach_{reach_label}": avg_reach,
            f"med_reach_{reach_label}": med_reach,
            "n_decay_terms":   n_terms,
        })

    # ── Print comparison table ────────────────────────────────────────────────
    W_TBL = 90
    print()
    print("=" * W_TBL)
    print("SPATIAL MULTIPLIER DECOMPOSITION SUMMARY")
    print("S = (I - lambda*W)^{-1}   |   NOTE: SEM error-process co-movement, not causal SAR")
    print("=" * W_TBL)
    hdr = (f"{'Outcome':<10} {'W':<12} {'Sample':<12} {'lambda':>8} {'Total mult':>11} "
           f"{'Avg direct':>11} {'Avg indir':>10} {'Indir %':>8} "
           f"{f'Avg reach ({reach_label})':>18}")
    print(hdr)
    print("-" * W_TBL)
    for r in summary_rows:
        reach_key = f"avg_reach_{reach_label}"
        print(f"{r['outcome']:<10} {r['w_matrix']:<12} {r['sample']:<12} {r['lambda']:>8.4f} "
              f"{r['total_multiplier']:>11.4f} {r['avg_direct']:>11.4f} "
              f"{r['avg_indirect']:>10.4f} {r['indirect_share_pct']:>7.2f}% "
              f"{r.get(reach_key, float('nan')):>18.1f}")
    print("=" * W_TBL)
    print("Total multiplier = 1/(1-lambda)  [analytic, verified numerically]")
    print("Direct effect    = mean diagonal of S (own-county retention)")
    print("Indirect effect  = total - direct  (network-mediated transmission)")
    print(f"Reach            = transmission-weighted avg {'geographic distance' if reach_label=='km' else 'network hops'}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    if output_dir is not None:
        # Main summary
        pd.DataFrame(summary_rows).to_csv(
            output_dir / "spatial_multiplier_decomposition.csv", index=False)
        print(f"\nSaved spatial_multiplier_decomposition.csv")

        # Decay series
        decay_cols = ["outcome", "w_matrix", "sample", "order_k", "contribution", "cumulative_share"]
        pd.concat(all_decay, ignore_index=True)[decay_cols].to_csv(
            output_dir / "spatial_multiplier_decay.csv", index=False)
        print("Saved spatial_multiplier_decay.csv")

        # Hub counties
        hub_cols = ["outcome", "w_matrix", "direction", "rank",
                    "fips5", "state", "total_transmission"]
        pd.concat(all_hubs, ignore_index=True)[hub_cols].to_csv(
            output_dir / "hub_counties.csv", index=False)
        print("Saved hub_counties.csv")

    return {"summary": summary_rows, "decay": all_decay, "hubs": all_hubs}


if __name__ == "__main__":
    run(Path(__file__).parents[2] / "output")
