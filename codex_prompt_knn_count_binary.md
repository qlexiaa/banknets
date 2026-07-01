# Codex Prompt: Add KNN-3/4 Variants for W_bank_count and W_bank_binary

## Goal

Add four new weight matrices and run them through the full SEM pipeline so
the thesis can compare whether the type of bank-network connection
(continuous strength vs. count vs. binary) matters once geographic reach
(via KNN truncation) is held constant.

New matrices:
  - W_bank_count_knn3 : top-3 connections by count weight, row-standardised
  - W_bank_count_knn4 : top-4 connections by count weight, row-standardised
  - W_bank_binary_knn3: top-3 connections by count ranking, binary (0/1) weights, row-standardised
  - W_bank_binary_knn4: top-4 connections by count ranking, binary (0/1) weights, row-standardised

The conceptual distinction: W_bank_knn3/4 keep the top-k by *continuous*
branch-weighted strength. The new variants isolate whether the count of
shared banks (W_bank_count_knn3/4) or mere presence/absence of connection
(W_bank_binary_knn3/4) — rather than continuous strength — drives the
spatial error parameter.

---

## Step 1 — `analysis/w_variants.py`

### 1a. Add a `_build_knn_from_count` helper

Add a function that takes the raw count matrix (`W_bank_count_raw`) as the
ranking criterion but preserves either count weights or binary weights:

```python
def _build_knn_from_count(W_count_sp, k, binary=False):
    """Keep top-k connections ranked by count; weights = count or 1."""
    W = W_count_sp.toarray().astype(np.float64)
    np.fill_diagonal(W, 0.0)
    N = W.shape[0]
    W_knn = np.zeros_like(W)
    for i in range(N):
        row = W[i]
        nz  = np.count_nonzero(row)
        if nz == 0:
            continue
        if nz <= k:
            W_knn[i] = 1.0 if binary else row
        else:
            top_k = np.argpartition(row, -k)[-k:]
            W_knn[i, top_k] = 1.0 if binary else row[top_k]
    np.fill_diagonal(W_knn, 0.0)
    return scipy.sparse.csr_matrix(W_knn)
```

### 1b. Add cache paths

In the `PATHS` dict add:
```python
"W_bank_count_knn3"  : DATA / "W_bank_count_knn3.npz",
"W_bank_count_knn4"  : DATA / "W_bank_count_knn4.npz",
"W_bank_binary_knn3" : DATA / "W_bank_binary_knn3.npz",
"W_bank_binary_knn4" : DATA / "W_bank_binary_knn4.npz",
```

### 1c. Add to `load_bank_variants()`

After the existing knn4 block, add:

```python
# W_bank_count_knn3/knn4 (top-k by count weight)
W_count_knn3_raw = _load_or_build(
    "W_bank_count_knn3",
    lambda: _build_knn_from_count(W_count_raw, k=3, binary=False)
)
variants["W_bank_count_knn3"] = row_standardize(W_count_knn3_raw)

W_count_knn4_raw = _load_or_build(
    "W_bank_count_knn4",
    lambda: _build_knn_from_count(W_count_raw, k=4, binary=False)
)
variants["W_bank_count_knn4"] = row_standardize(W_count_knn4_raw)

# W_bank_binary_knn3/knn4 (top-k by count ranking, binary weights)
W_bin_knn3_raw = _load_or_build(
    "W_bank_binary_knn3",
    lambda: _build_knn_from_count(W_count_raw, k=3, binary=True)
)
variants["W_bank_binary_knn3"] = row_standardize(W_bin_knn3_raw)

W_bin_knn4_raw = _load_or_build(
    "W_bank_binary_knn4",
    lambda: _build_knn_from_count(W_count_raw, k=4, binary=True)
)
variants["W_bank_binary_knn4"] = row_standardize(W_bin_knn4_raw)
```

### 1d. Add to `ALL_VARIANTS` list

```python
ALL_VARIANTS = [
    "W_bank", "W_bank_count", "W_bank_binary",
    "W_bank_knn3", "W_bank_knn4",
    "W_bank_count_knn3", "W_bank_count_knn4",
    "W_bank_binary_knn3", "W_bank_binary_knn4",
    "W_bank_nonGeo", "W_bank_interstate", "W_bank_intrastate",
]
```

---

## Step 2 — `analysis/sem/sem_w_variants.py`

Re-run with all new matrices included so they appear in
`output/four_w_comparison_credit.csv`.

- The script already loops over `load_bank_variants()` which now returns
  the four new matrices — no other changes needed to the loop logic.
- After running, verify `four_w_comparison_credit.csv` has rows for
  `W_bank_count_knn3`, `W_bank_count_knn4`, `W_bank_binary_knn3`,
  `W_bank_binary_knn4` for all three samples (Full, Contig, NonContig).

```bash
python analysis/sem/sem_w_variants.py
```

---

## Step 3 — `analysis/update_thesis_latex_values.py`

### 3a. Add labels to `W_LABELS`

```python
"W_bank_count_knn3"  : r"$W_{\text{bank,count,knn3}}$",
"W_bank_count_knn4"  : r"$W_{\text{bank,count,knn4}}$",
"W_bank_binary_knn3" : r"$W_{\text{bank,bin,knn3}}$",
"W_bank_binary_knn4" : r"$W_{\text{bank,bin,knn4}}$",
```

### 3b. Add a new builder: `build_knn_type_comparison()`

Add a new table builder that groups by connection type for a direct
comparison. The table shows λ and z-gap for all KNN-3 and KNN-4 variants
side-by-side, for the Full sample:

| Matrix | Type | k | λ | SE(λ) | Δλ vs W_geo (z) |
|---|---|---|---|---|---|
| W_bank_knn3      | Continuous | 3 | ... | ... | ... |
| W_bank_count_knn3| Count      | 3 | ... | ... | ... |
| W_bank_binary_knn3| Binary    | 3 | ... | ... | ... |
| W_bank_knn4      | Continuous | 4 | ... | ... | ... |
| W_bank_count_knn4| Count      | 4 | ... | ... | ... |
| W_bank_binary_knn4| Binary    | 4 | ... | ... | ... |

```python
def build_knn_type_comparison() -> str:
    rows = read_csv(OUT / "four_w_comparison_credit.csv")
    order = [
        ("W_bank_knn3",       "Continuous", "3"),
        ("W_bank_count_knn3", "Count",      "3"),
        ("W_bank_binary_knn3","Binary",      "3"),
        ("W_bank_knn4",       "Continuous", "4"),
        ("W_bank_count_knn4", "Count",      "4"),
        ("W_bank_binary_knn4","Binary",      "4"),
    ]
    lines = []
    for w, wtype, k in order:
        if k == "4" and lines:
            lines.append(r"    \addlinespace")
        r = row_by(rows, sample="Full", w_matrix=w)
        gap = rf"${fmt(f(r,'gap_vs_geo'),3)}\ (z{{=}}{fmt(f(r,'z_gap'),2)}){stars(f(r,'p_gap_onesided'))}$"
        lines.append(
            rf"    {wtype:<12} & $k={k}$ & "
            rf"{coef_se_spaced(f(r,'lam'), f(r,'p_lam'), f(r,'se_lam'))} & {gap} \\"
        )
    return rf"""
\begin{{table}}[htbp]
  \centering
  \caption{{Spatial error parameter by connection type and density: KNN-3 vs KNN-4,
  continuous vs count vs binary weights (full sample).}}
  \label{{tab:knn-type}}
  \begin{{tabular}}{{llcc}}
    \toprule
    Weight type & Density & $\hat\lambda$ & $\Delta\hat\lambda$ vs.\ $W_{{\text{{geo}}}}$ ($z$) \\
    \midrule
{chr(10).join(lines)}
    \bottomrule
  \end{{tabular}}
  \begin{{tablenotes}}\footnotesize
  \item \emph{{Notes.}} All matrices truncated to their top-$k$ connections per county
  (ranked by continuous branch strength) then row-standardised. ``Continuous'': weights
  proportional to branch strength. ``Count'': weights proportional to number of shared
  banks. ``Binary'': unweighted (0/1). $^{{*}}p<0.10$, $^{{**}}p<0.05$, $^{{***}}p<0.01$.
  \end{{tablenotes}}
\end{{table}}
"""
```

### 3c. Register the new table

Add to the `builders` list in `update_latex()`:
```python
("tab:knn-type", build_knn_type_comparison),
```

### 3d. Update `build_sem_full()` to include new matrices

In `build_sem_full()`, extend the `matrices` list:
```python
matrices = [
    "W_geo",
    "W_bank",
    "W_bank_count",
    "W_bank_binary",
    "W_bank_knn3",
    "W_bank_knn4",
    "W_bank_count_knn3",   # NEW
    "W_bank_count_knn4",   # NEW
    "W_bank_binary_knn3",  # NEW
    "W_bank_binary_knn4",  # NEW
    "W_bank_nonGeo",
    "W_bank_interstate",
    "W_bank_intrastate",
]
```

---

## Step 4 — Insert `tab:knn-type` into `thesis_updated.tex`

After the existing `tab:knn` table environment in the appendix, add:

```latex
% --- NEW TABLE: KNN type comparison ---
\input{...}   % or paste the generated environment directly
```

The exact placement should be in the robustness appendix section, after the
KNN sweep table (`tab:knn`) and before the SAR full table (`tab:sar-full`).
In `thesis_updated.tex`, find the line:
```latex
\label{tab:knn}
```
and insert the new `\begin{table}...\end{table}` block for `tab:knn-type`
immediately after the `\end{table}` that closes `tab:knn`.

---

## Step 5 — Narrative addition

In the section that discusses KNN robustness (search for "knn" or "nearest"
in the thesis), add a sentence pointing to the new table, e.g.:

> Table~\ref{tab:knn-type} isolates the role of weighting scheme: across both
> $k=3$ and $k=4$, the continuous, count, and binary specifications deliver
> almost identical $\hat\lambda$ and gap statistics, suggesting that the
> number of shared-bank links matters but the exact calibration of link
> strength does not.

*(Adjust the wording once you see the actual results.)*

---

## Verification

After all steps:
```bash
python analysis/sem/sem_w_variants.py        # regenerates four_w_comparison_credit.csv
python analysis/update_thesis_latex_values.py thesis_updated.tex -o thesis_updated.tex
```

Check that:
- `four_w_comparison_credit.csv` has 39 rows (13 matrices × 3 samples)
- `SKIP tab:knn-type` does NOT appear in the update script output
- The new table renders correctly in the thesis
