# Codex Prompts: Fix Incomplete Output CSVs

Each section is a self-contained prompt. Run them in order — later scripts
(fgls, sar) read lambda values that come from the SEM run which is already
complete (`four_w_comparison_credit.csv` is now fully populated).

---

## 1. `analysis/sar/sar_credit.py` → `output/sar_robustness_credit.csv`

**Problem:** The CSV has only 3 rows (Full/W_geo with missing delta/z,
Full/W_bank complete, Full/W_bank_count empty). Needs 27 rows:
Full/Contig/NonContig × 9 matrices.

**Prompt:**

```
Fix analysis/sar/sar_credit.py so it produces all 27 rows in
output/sar_robustness_credit.csv: three samples (Full, Contig, NonContig)
× nine matrices (W_geo, W_bank, W_bank_count, W_bank_binary, W_bank_knn3,
W_bank_knn4, W_bank_nonGeo, W_bank_interstate, W_bank_intrastate).

Context:
- The script already has a working run_panel_fe_lag() and extract_lag()
  function and loops over samples returned by get_samples(panel).
- The current loop only covers W_geo and W_bank; extend it to all nine
  matrices from load_bank_variants() (which returns all variants).
- For W_geo the gap columns (delta_rho, delta_rho_se, z_stat) should be
  empty/NaN (W_geo is the baseline, no gap to compute).
- For all other matrices compute delta_rho = rho_w - rho_geo,
  delta_rho_se = sqrt(rho_se_w**2 + rho_se_geo**2),
  z_stat = delta_rho / delta_rho_se  (one-sided, same convention as SEM).
- Use the same panel construction as sem_credit.py: load_panel_with_credit(),
  get_samples(), row_standardize(), sparse_to_pysal_w().
- W_bank_nonGeo, W_bank_interstate, W_bank_intrastate are loaded from
  load_bank_variants() — make sure those keys are included in the loop.
- Output CSV columns must be: sample, w_matrix, n_counties, n_obs, rho,
  rho_se, beta_D, beta_D_se, delta_rho, delta_rho_se, z_stat
- Run python analysis/sar/sar_credit.py from the repo root and verify
  output/sar_robustness_credit.csv has 27 non-empty rows.
```

---

## 2. `analysis/inference/fgls.py` → `output/fgls_comparison.csv`

**Problem:** Has 9 rows (Full×5, Contig×3 + 1 empty junk row). Needs 15 rows:
Full/Contig/NonContig × 5 estimators (OLS, FGLS W_geo, FGLS W_bank,
FGLS W_bank_knn3, FGLS W_bank_knn4). The 9th row has `estimator='FGLS '`
with empty beta — this is a corrupt partial row that must be removed.

**Prompt:**

```
Fix analysis/inference/fgls.py so output/fgls_comparison.csv has exactly
15 rows: samples Full, Contig, NonContig × estimators
  "OLS (Favara-Imbs)", "FGLS W_geo", "FGLS W_bank",
  "FGLS W_bank_knn3", "FGLS W_bank_knn4"
in that order.

Context:
- The existing code already works for Full (5 rows) and partially for
  Contig (OLS, FGLS W_geo, FGLS W_bank confirmed correct).
- The 9th row in the current CSV has estimator "FGLS " (trailing space,
  empty beta) — this is a corrupt partial run. Delete it.
- The lambda values for the FGLS filters are read from
  output/panel_fe_credit_results.csv (which has Full/Contig/NonContig ×
  W_geo/W_bank/W_bank_knn3/W_bank_knn4 lambdas — it is complete).
- Use get_samples(panel) to loop over all three samples.
- State-clustered SEs use unfiltered residuals as the score and the
  filtered regressor as the bread (as already implemented for Full).
- CSV columns: sample, estimator, beta, se, ci_lower, ci_upper, t_stat,
  p_value (two-sided), hausman_chi2, hausman_p (for W_bank row only,
  NaN elsewhere).
- Overwrite output/fgls_comparison.csv with all 15 rows.
- Run python analysis/inference/fgls.py and confirm 15 rows, no empty beta.
```

---

## 3. `analysis/sar/multiplier_decomposition.py` → `output/spatial_multiplier_decomposition.csv`

**Problem:** The W_bank_knn4 row has lambda=0.26357 and
total_multiplier=1.35790 correctly filled in, but avg_indirect,
indirect_share_pct, avg_reach_km, med_reach_km, n_decay_terms are all None.

**Prompt:**

```
Fix analysis/sar/multiplier_decomposition.py so the W_bank_knn4 row in
output/spatial_multiplier_decomposition.csv is fully populated.

Context:
- The script does a power-series expansion S = sum_{k=0}^K (lambda*W)^k
  and from that computes:
    avg_indirect     = mean of off-diagonal elements of S
    indirect_share_pct = (avg_indirect / avg_diagonal) * 100
                       = (1 - 1/total_multiplier) * 100  [approximately]
    avg_reach_km     = transmission-weighted average great-circle distance
    med_reach_km     = median of the same distribution
    n_decay_terms    = number of power-series terms retained (until
                       cumulative share > 0.999 or |lambda|^k < 1e-6)
- For W_bank_knn4: lambda = 0.26357, total_multiplier = 1.35790 (already
  correct). The decomposition is failing to compute the reach statistics,
  likely because the W_bank_knn4 matrix object is not being loaded or the
  loop is exiting early.
- W_bank_knn4 is loaded via load_bank_variants()["W_bank_knn4"] from
  w_variants.py; confirm it is row-standardised before use.
- The full sample has n_counties=877 for W_geo/W_bank but W_bank_knn4
  uses 853 counties (those with ≥4 bank-network neighbours); make sure
  the county coordinate lookup is restricted to those same 853 counties
  when computing great-circle distances.
- Re-run python analysis/sar/multiplier_decomposition.py and verify all 4
  rows in output/spatial_multiplier_decomposition.csv have non-null values
  for avg_indirect, indirect_share_pct, avg_reach_km, med_reach_km,
  n_decay_terms.
- Also verify output/spatial_multiplier_decay.csv has decay rows for
  W_bank_knn4.
```

---

## 4. `analysis/extensions/slx_exposure.py` → `output/slx_exposure_results.csv`

**Problem:** Has 10 rows (Full×5, Contig×5). NonContig (5 rows) is entirely
absent. Also rows 1 and 6 (the "base" regression without E) have
beta_E/se_E/t_E empty — that is expected for the base spec; confirm the
builder handles None gracefully.

**Prompt:**

```
Fix analysis/extensions/slx_exposure.py so output/slx_exposure_results.csv
includes all three samples: Full, Contig, NonContig.

Context:
- The script currently only loops over Full and Contig. Add NonContig
  (the 639 non-contiguous counties, returned as the third element of
  get_samples(panel)).
- The SLX exposure variable E uses the row-standardised interstate
  bank-network matrix W_bank_interstate. For NonContig counties many
  cross-state bank links are absent; if the exposure has zero variance
  for NonContig, record NaN for beta_E and note it in the 'spec' column.
- The base regression rows (spec="base") deliberately have no E column —
  beta_E/se_E/t_E should be empty/NaN there. This is correct; leave as-is.
- For the permutation p-value: use seed=42 and 999 permutations, same as
  Full and Contig.
- Output CSV must have 15 rows (3 samples × 5 specs).
- Overwrite output/slx_exposure_results.csv.
- Run python analysis/extensions/slx_exposure.py and confirm 15 rows.
```

---

## 5. `analysis/sem/sem_knn_sweep.py` → `output/knn_sweep_credit_results.csv`

**Problem:** k=11 row has empty lam_bank_knn_contig, gap_contig,
n_co_contig, n_obs_contig (and likely noncontig too). The build_knn_table()
function iterates all rows and fails on the empty contig value.

**Prompt:**

```
Fix analysis/sem/sem_knn_sweep.py so output/knn_sweep_credit_results.csv
has no empty cells, and update_thesis_latex_values.py can render tab:knn.

Two options — implement whichever is correct for the data:

Option A (preferred if k=11 genuinely fails for Contig):
  After writing results, drop any row where lam_bank_knn_contig is NaN/empty
  before saving the CSV. The thesis table will then show only k=1..10.
  Add a comment in the script: "k=11 excluded: contiguous sample has <11
  average bank neighbours, making knn11 degenerate."

Option B (if k=11 can be computed for Contig):
  Debug why k=11 fails for the contiguous sample. Likely cause: the
  contiguous subsample (238 counties along MSA borders) does not have ≥11
  bank-network neighbours for all counties, so W_bank_knn11 is not
  fully connected. If spreg returns a singular matrix error, fall back to
  recording NaN for that cell and apply Option A.

After the fix:
- Run python analysis/sem/sem_knn_sweep.py
- Confirm output/knn_sweep_credit_results.csv has no empty lam_bank_knn_contig cells.
- Run python analysis/update_thesis_latex_values.py thesis_updated.tex -o thesis_updated.tex
  and verify tab:knn no longer raises KeyError.
```

---

## 6. `analysis/model_selection/jtest.py` → `output/jtest_credit_results.csv`

**Problem:** Only Full × 6 matrices (W_bank through W_bank_nonGeo) present.
Missing: Full W_bank_interstate + W_bank_intrastate, all Contig rows (×8),
all NonContig rows (×8). Needs 24 rows (3 samples × 8 matrices).

**Prompt:**

```
Fix analysis/model_selection/jtest.py so output/jtest_credit_results.csv
has 24 rows: samples Full, Contig, NonContig × alt matrices
  W_bank, W_bank_count, W_bank_binary, W_bank_knn3, W_bank_knn4,
  W_bank_nonGeo, W_bank_interstate, W_bank_intrastate
(each tested against W_geo as the null).

Context:
- The current script loops over Full and "Border" samples with a subset of
  matrices. Rename "Border" → "Contig" (consistent with the rest of the
  codebase) and add "NonContig".
- Use get_samples(panel) to obtain Full/Contig/NonContig consistently.
- For each (sample, w_alt) pair, run both directions of the J-test:
    Direction 1: W_geo null, w_alt alternative
    Direction 2: w_alt null, W_geo alternative
  The J-test mechanism is already implemented for W_bank; extend the loop.
- W_bank_interstate and W_bank_intrastate are in load_bank_variants().
  For NonContig, W_bank_interstate may have many zero rows (non-contiguous
  counties have fewer cross-state connections); spreg should still run.
- Conclusion classification: 'W_null preferred' if only D2 rejects,
  'W_alt preferred' if only D1 rejects, 'complementary' if both reject,
  'indeterminate' if neither.
- CSV columns: sample, pair, w_null, w_alt, n_counties,
  d1_j_coeff, d1_j_se, d1_j_z, d1_j_p, d1_reject,
  d2_j_coeff, d2_j_se, d2_j_z, d2_j_p, d2_reject, conclusion
- Overwrite output/jtest_credit_results.csv with all 24 rows.
- Run python analysis/model_selection/jtest.py and confirm 24 rows.
```

---

## 7. `analysis/sar/sar_iv_credit.py` → `output/sar_iv_results.csv`

**Problem:** Has 3 rows (Full/W_geo/KP-2SLS, Full/W_geo/SDM-IV,
Full/W_bank/KP-2SLS). Full/W_bank/KP-2SLS has first_stage_F_cluster=None.
Missing: Full/W_bank/SDM-IV and all 4 Contig rows.
`build_ivsar_table()` needs 8 rows:
Full/W_geo/KP-2SLS, Full/W_geo/SDM-IV, Full/W_bank/KP-2SLS, Full/W_bank/SDM-IV,
Contig/W_geo/KP-2SLS, Contig/W_geo/SDM-IV, Contig/W_bank/KP-2SLS, Contig/W_bank/SDM-IV.

**Prompt:**

```
Fix analysis/sar/sar_iv_credit.py so output/sar_iv_results.csv has 8 rows:
  Full  × {W_geo, W_bank} × {KP-2SLS, SDM-IV}
  Contig × {W_geo, W_bank} × {KP-2SLS, SDM-IV}

Context:
- The script already implements KP-2SLS (two instruments: W*D and W²*D)
  and SDM-IV (adds W*X controls). Extend the sample loop to include Contig
  using get_samples(panel).
- For Full/W_bank/KP-2SLS the first_stage_F_cluster is None — this means
  the state-clustered first-stage F was not computed. The formula is:
    F_cluster = (pi_hat_WD / SE_cluster_WD)^2
  where pi_hat_WD is the coefficient on the first instrument (WD) in the
  first stage and SE_cluster_WD uses the state-clustered sandwich.
  Compute and record it.
- Output CSV columns (existing schema): sample, W, spec, rho, rho_se,
  rho_ci_lower, rho_ci_upper, beta, beta_se, beta_ci_lower, beta_ci_upper,
  theta_WD, theta_WD_se, first_stage_F, first_stage_F_cluster,
  corr_q1q2, corr_q2q3, cond_num_instr, residual_moran_i_mean,
  N_counties, N_obs
- Overwrite output/sar_iv_results.csv with all 8 rows.
- Run python analysis/sar/sar_iv_credit.py and confirm 8 rows,
  no None in first_stage_F_cluster.
```

---

## After all scripts are fixed

Run the update script to regenerate all thesis tables:

```bash
cd C:\Users\alxmc\Projects\Thesis
python analysis/update_thesis_latex_values.py thesis_updated.tex -o thesis_updated.tex
```

Expected: zero SKIP lines in output. Then upload `thesis_updated.tex` to
Prism (prism.openai.com) to replace `main.tex`.
