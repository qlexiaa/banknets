"""
Bank-geography network overlap analysis
========================================
For each county, computes the fraction of its top-5 bank connections
(by weight in W_bank_avg) that are also geographic neighbours in W_geo.

Overlap fraction = |top5_bank_nbrs intersect geo_nbrs| / 5

Counties with 0 geographic neighbours (islands) are skipped.

Outputs:
  geo_bank_overlap_stats.csv        -- summary statistics (key-value pairs)
  geo_bank_overlap_distribution.csv -- histogram data (bins 0, 0.2, 0.4, 0.6, 0.8, 1.0)
  overlap_histogram.png             -- histogram plot
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import numpy as np
import scipy.sparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parents[1]))
import utils  # noqa: applies spreg compatibility patch
from utils import gal_to_W, row_standardize

ROOT        = Path(__file__).parents[2]
GAL_PATH    = ROOT / "data" / "W_geo_queen.gal"
COUNTY_PATH = ROOT / "data" / "county_order_Wgeo.csv"
WBANK_PATH  = ROOT / "data" / "W_bank_avg.npz"

TOP_K = 5   # number of top bank connections to consider per county


def run(output_dir=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load inputs ───────────────────────────────────────────────────────────
    co_df        = pd.read_csv(COUNTY_PATH, dtype={"fips5": str})
    county_order = co_df["fips5"].str.zfill(5).tolist()
    N            = len(county_order)

    W_geo_sparse, gal_order = gal_to_W(GAL_PATH, county_order)
    assert gal_order == county_order, "GAL order != county_order_Wgeo.csv"

    W_bank_sparse = row_standardize(scipy.sparse.load_npz(WBANK_PATH))
    assert W_bank_sparse.shape == (N, N)

    # Dense arrays for indexing
    W_bank = W_bank_sparse.toarray()
    W_geo  = W_geo_sparse.toarray()
    np.fill_diagonal(W_bank, 0.0)
    np.fill_diagonal(W_geo,  0.0)

    # ── Compute per-county overlap fraction ───────────────────────────────────
    overlaps = []
    for i in range(N):
        bank_row = W_bank[i]
        geo_row  = W_geo[i]

        geo_nbrs = set(np.where(geo_row > 0)[0])
        if len(geo_nbrs) == 0:
            continue  # skip islands

        # Top-5 bank connections
        nz = np.count_nonzero(bank_row)
        if nz == 0:
            overlaps.append(0.0)
            continue

        if nz <= TOP_K:
            top5_idx = set(np.where(bank_row > 0)[0])
        else:
            top5_idx = set(np.argpartition(bank_row, -TOP_K)[-TOP_K:])

        intersection = top5_idx & geo_nbrs
        overlaps.append(len(intersection) / TOP_K)

    overlaps = np.array(overlaps)
    n_counties = len(overlaps)

    # ── Summary statistics ────────────────────────────────────────────────────
    stats = {
        "n_counties":   n_counties,
        "mean":         float(np.mean(overlaps)),
        "median":       float(np.median(overlaps)),
        "p25":          float(np.percentile(overlaps, 25)),
        "p75":          float(np.percentile(overlaps, 75)),
        "min":          float(np.min(overlaps)),
        "max":          float(np.max(overlaps)),
        "std":          float(np.std(overlaps)),
        "n_overlap_0":  int(np.sum(overlaps == 0.0)),
        "n_overlap_1":  int(np.sum(overlaps == 1.0)),
        "n_overlap_ge02": int(np.sum(overlaps >= 0.2)),
        "n_overlap_ge04": int(np.sum(overlaps >= 0.4)),
        "n_overlap_ge06": int(np.sum(overlaps >= 0.6)),
        "n_overlap_ge08": int(np.sum(overlaps >= 0.8)),
    }

    # ── Histogram data ────────────────────────────────────────────────────────
    # Overlap fraction can only take 6 exact values: 0, 0.2, 0.4, 0.6, 0.8, 1.0
    bin_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_labels = ["0\n(0/5)", "0.2\n(1/5)", "0.4\n(2/5)",
                  "0.6\n(3/5)", "0.8\n(4/5)", "1.0\n(5/5)"]

    dist_rows = []
    for val, lbl in zip(bin_values, bin_labels):
        cnt = int(np.sum(np.isclose(overlaps, val)))
        dist_rows.append(dict(
            overlap_bin=lbl,
            n_counties=cnt,
            pct=100.0 * cnt / n_counties if n_counties > 0 else 0.0,
        ))

    # ── Save outputs ──────────────────────────────────────────────────────────
    if output_dir is not None:
        # Stats CSV (key-value)
        stats_rows = [{"metric": k, "value": v} for k, v in stats.items()]
        pd.DataFrame(stats_rows).to_csv(
            output_dir / "geo_bank_overlap_stats.csv", index=False)

        # Distribution CSV
        pd.DataFrame(dist_rows).to_csv(
            output_dir / "geo_bank_overlap_distribution.csv", index=False)

        # Histogram plot
        fig, ax = plt.subplots(figsize=(8, 5))
        bar_counts = [r["n_counties"] for r in dist_rows]

        ax.bar(bin_values, bar_counts, width=0.15, color='steelblue', alpha=0.8,
               edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Overlap fraction (top-5 bank nbrs in geo nbrs)', fontsize=12)
        ax.set_ylabel('Number of counties', fontsize=12)
        ax.set_title(
            f'Bank-geography network overlap\n'
            f'(top-{TOP_K} bank connections vs queen contiguity neighbours)\n'
            f'N={n_counties} counties  |  mean={stats["mean"]:.3f}  '
            f'median={stats["median"]:.3f}',
            fontsize=11)
        ax.set_xticks(bin_values)
        ax.set_xticklabels(bin_labels)
        ax.grid(axis='y', alpha=0.35)

        # Annotate bars with percentages
        for x_pos, cnt in zip(bin_values, bar_counts):
            pct = 100.0 * cnt / n_counties if n_counties > 0 else 0.0
            if cnt > 0:
                ax.text(x_pos, cnt + max(bar_counts) * 0.01,
                        f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_dir / 'overlap_histogram.png', dpi=150)
        plt.close()

    return dict(overlaps=overlaps, stats=stats, dist_rows=dist_rows)


if __name__ == '__main__':
    run(ROOT / 'output')
