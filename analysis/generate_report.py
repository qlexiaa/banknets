"""
generate_report.py
==================
Auto-generates output/results_report.md from all result CSVs in output/.

Run any time after running new analyses to refresh the report:
    python analysis/generate_report.py

For each analysis the report documents:
  1. What regression / analysis was run and why
  2. Every model equation and formula used (with academic references)
  3. Key estimation assumptions
  4. Results (formatted tables)

New result CSVs are picked up automatically — sections whose CSV is missing
are skipped and noted at the bottom.
"""

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "output"
DIAG = OUT / "diagnostics"

# ── helpers ───────────────────────────────────────────────────────────────────

def fmt(x, decimals=4):
    if pd.isna(x):
        return "—"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if abs(f) == 0:
        return "0"
    if abs(f) < 0.001:
        return f"{f:.3e}"
    return f"{f:.{decimals}f}"

def stars(p):
    try:
        p = float(p)
        if p < 0.01:  return "***"
        if p < 0.05:  return "**"
        if p < 0.10:  return "*"
        return ""
    except (TypeError, ValueError):
        return ""

def pval_str(p):
    try:
        p = float(p)
        if p < 0.001: return "<0.001"
        return f"{p:.3f}"
    except (TypeError, ValueError):
        return str(p)

def load(path):
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p)

def section(title, level=2):
    return f"\n{'#' * level} {title}\n"

def md_table(df):
    lines = ["| " + " | ".join(str(c) for c in df.columns) + " |",
             "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines) + "\n"

def md_table_custom(df, formatters):
    """formatters: dict col_name -> callable(val) -> str"""
    lines = ["| " + " | ".join(str(c) for c in df.columns) + " |",
             "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    for _, row in df.iterrows():
        cells = []
        for col, val in row.items():
            fn = formatters.get(col, fmt)
            cells.append(fn(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def sec_panel_fe_credit():
    df = load(OUT / "panel_fe_credit_results.csv")
    if df is None:
        return ""
    s = section("1. Main SEM — Credit Growth")

    s += """
**Source:** `analysis/sem/sem_credit.py`

**References:** Anselin (1988) *Spatial Econometrics*; Anselin et al. (2008) *A Local Indicator of Spatial Association*; Favara & Imbs (2015) *Credit Supply and the Price of Housing*

---

**Structural equation (Spatial Error Model):**

```
Δln(loans)_it = β · Linter_bra_it + α_i + τ_t + u_it        ...(1)
u_it = λ · (W u)_it + ε_it                                    ...(2)
```

where:
- `Δln(loans)_it` = log-change in commercial-bank mortgage loans (Favara-Imbs HMDA), county *i*, year *t*
- `Linter_bra_it` = cumulative interstate branching deregulation index (sum of indicator variables for each bilateral pair state deregulated)
- `α_i` = county fixed effect; `τ_t` = year fixed effect
- `W` = row-standardised spatial weight matrix (two variants compared)
- `λ` = spatial error autocorrelation parameter; `ε_it` ~ i.i.d.(0, σ²)

Reduced form obtained by substituting (2) into (1):

```
Δln(loans)_it = β · Linter_bra_it + α_i + τ_t + (I - λW)⁻¹ ε_it
```

**Estimation:** Maximum likelihood via `spreg.Panel_FE_Error` (Anselin et al. 2008). Log-likelihood for a single period under normality:

```
ln L = -(NT/2) ln(2πσ²) + T ln|I - λW| - (1/2σ²) ε'ε
```

where `ε = (I - λW) · ũ` and `ũ` are two-way FE within-demeaned residuals.

**Two-way within (FE) transformation** (applied before ML, following Mundlak 1978):

```
z̃_it = z_it - z̄_i - z̄_t + z̄      ...(3)
```

where `z̄_i` = county mean, `z̄_t` = year mean, `z̄` = grand mean.

**Gap test** (H₁: λ_bank > λ_geo, one-sided):

```
z_gap = (λ̂_bank - λ̂_geo) / sqrt(SE(λ̂_bank)² + SE(λ̂_geo)²)    ...(4)
p = Φ(-z_gap)   (normal tail)
```

Note: SE(gap) ignores the covariance between estimators (conservative).

**Spatial weight matrices:**
- **W_geo**: queen contiguity (binary adjacency, row-standardised)
- **W_bank**: time-averaged cosine similarity of county bank holding-company portfolios (Favara & Imbs 2015, Data Appendix)

**NaN policy:** Counties with any missing year in `Δln(loans)` are dropped entirely (67 counties), ensuring a strictly balanced panel required by `Panel_FE_Error`.

**Results:**

"""
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Model": r.get("model", f"{r.get('w_matrix','')} ({r.get('sample','')})"),
            "β (deregulation)": f"{fmt(r.get('beta', r.get('beta_D','')))} ({fmt(r.get('se_beta', r.get('se_beta_D','')))}){stars(r.get('p_beta', 0.001))}",
            "λ (spatial)": f"{fmt(r['lam'])} ({fmt(r['se_lam'])}){stars(r['p_lam'])}",
            "N counties": fmt(int(r["n_co"])),
            "N obs": fmt(int(r["n_obs"])),
        })
    s += md_table(pd.DataFrame(rows))
    s += "\n*Standard errors in parentheses. \\*p<0.10, \\*\\*p<0.05, \\*\\*\\*p<0.01*\n"
    return s


def sec_panel_fe_hpi():
    df = load(OUT / "panel_fe_hpi_results.csv")
    if df is None:
        return ""
    s = section("2. Main SEM — House Price Growth")

    s += """
**Source:** `analysis/sem/sem_hpi.py`

**References:** Favara & Imbs (2015); Anselin et al. (2008)

---

**Structural equation:**

```
Δln(HPI)_it = β_D · Linter_bra_it + β_Dη · Linter_ela_it
             + γ · Controls_it + α_i + τ_t + u_it        ...(5)
u_it = λ · (W u)_it + ε_it                               ...(6)
```

where:
- `Δln(HPI)_it` = log-change in house price index (OFHEO/FHFA)
- `Linter_bra_it` = interstate deregulation index (same as §1)
- `Linter_ela_it` = within-state (intrastate) deregulation index
- `Controls_it` ∈ {LΔln(HPI), Δln(pop), LΔln(pop), Δln(inc), LΔln(inc), Δln(hsosf), LΔln(hsosf)}
- Prefix L = one-year lag

**Estimation:** Same ML as §1 (eq. ln L above). Estimated **unweighted** — Favara & Imbs (2015) weight HPI regressions by 1/(counties per state) but `spreg.Panel_FE_Error` does not support observation weights, so this SEM is unweighted.

**Two-way within transformation:** Equation (3) above, applied to all variables.

**Lambda gap test:** Equation (4) above, `Δλ = λ_bank - λ_geo`.

**Results:**

"""
    rows = []
    for _, r in df.iterrows():
        gap_str = ""
        if pd.notna(r.get("delta_lambda", np.nan)) and not pd.isna(r.get("delta_lambda")):
            gap_str = f"{fmt(r['delta_lambda'])} ({fmt(r['delta_lambda_se'])}){stars(0.0001)}"
        rows.append({
            "Sample / W": f"{r['sample']} / {r['w_matrix']}",
            "β_D": f"{fmt(r['beta_D'])} ({fmt(r['beta_D_se'])}){stars(0.001)}",
            "β_Dη": f"{fmt(r['beta_Deta'])} ({fmt(r['beta_Deta_se'])}){stars(0.001)}",
            "λ": f"{fmt(r['lambda'])} ({fmt(r['lambda_se'])}){stars(0.001)}",
            "Δλ (bank−geo)": gap_str if gap_str else "—",
            "N counties": fmt(int(r["n_counties"])),
        })
    s += md_table(pd.DataFrame(rows))
    s += "\n*Standard errors in parentheses.*\n"
    return s


def sec_four_w():
    df = load(OUT / "four_w_comparison_credit.csv")
    if df is None:
        return ""
    s = section("3. Four-W Comparison — Credit Growth")

    s += """
**Source:** `analysis/sem/sem_w_variants.py`

**References:** Anselin (1988); Favara & Imbs (2015)

---

**Model:** Same as §1, equations (1)–(2).

**Gap test** (one-sided z, same as equation (4)):
```
z_gap = (λ̂_alt - λ̂_geo) / sqrt(SE(λ̂_alt)² + SE(λ̂_geo)²)
p_gap = Φ(-z_gap)
```

**Likelihood-ratio test** (also reported):
```
LR = 2 · (logLL_alt - logLL_geo) ~ χ²(1) under H₀
```

**Spatial weight matrices compared:**
- **W_geo**: queen contiguity benchmark
- **W_bank**: time-averaged cosine HHC similarity (continuous, row-standardised)
- **W_bank_count**: branch-count-weighted HHC overlap (row-standardised)
- **W_bank_binary**: binary HHC overlap (row-standardised)
- **W_bank_knn4**: top-4 W_bank links per county (row-standardised)
- **W_bank_nonGeo**: W_bank with geographic-neighbour entries zeroed out (row-standardised)
- **W_bank_interstate**: W_bank restricted to cross-state links only
- **W_bank_intrastate**: W_bank restricted to within-state links only

**Results:**

"""
    rows = []
    for _, r in df.iterrows():
        gap_str = f"{fmt(r['gap_vs_geo'])} (z={fmt(r['z_gap'], 2)}){stars(r['p_gap_onesided'])}" \
            if pd.notna(r.get("gap_vs_geo", np.nan)) and not pd.isna(r.get("gap_vs_geo")) else "—"
        rows.append({
            "Sample": r["sample"],
            "W matrix": r["w_matrix"],
            "β": f"{fmt(r['beta'])} ({fmt(r['se_beta'])}){stars(r['p_beta'])}",
            "λ": f"{fmt(r['lam'])} ({fmt(r['se_lam'])}){stars(r['p_lam'])}",
            "Δλ vs W_geo": gap_str,
            "N obs": fmt(int(r["n_obs"])),
        })
    s += md_table(pd.DataFrame(rows))
    s += "\n*Standard errors in parentheses. Gap is one-sided z-test (H₁: λ_alt > λ_geo).*\n"
    return s


def sec_composite_credit():
    opt = load(OUT / "composite_w_credit_optima.csv")
    if opt is None:
        return ""
    s = section("4. Composite W Profile — Credit Growth")

    s += """
**Source:** `analysis/model_selection/composite_w.py`

**References:** LeSage & Pace (2009) *Introduction to Spatial Econometrics*

---

**Profile likelihood over convex combinations of W matrices:**

```
W_combo(α) = α · W_geo + (1 - α) · W_alt       ...(7)
```

where α ∈ {0, 0.05, 0.10, …, 1.0} (21-point grid). For each α, `W_combo` is row-standardised before estimation.

The model at each grid point is the same SEM as §1, equations (1)–(2), but with `W = W_combo(α)`.

**Optimum:** α* = argmax_α logLL(α)

**Log-likelihood improvement:**
```
ΔlogLL = logLL(α*) - max(logLL_geo, logLL_bank)     ...(8)
```

Interpretation: α* = 1 → pure geographic; α* = 0 → pure bank-network.

**Pairs profiled:**
- **Pair A**: W_geo vs W_bank (continuous cosine)
- **Pair B**: W_geo vs W_bank_count (branch-count weighted)
- **Pair C**: W_geo vs W_bank_nonGeo (non-geographic bank links)

**Optimal mixing weights:**

"""
    rows = []
    for _, r in opt.iterrows():
        rows.append({
            "Pair": r["pair"],
            "W_alt": r["w_alt"],
            "Sample": r["sample"],
            "α*": fmt(r["alpha_star"]),
            "λ at α*": f"{fmt(r['lam_at_star'])} ({fmt(r['se_lam_at_star'])})",
            "β at α*": f"{fmt(r['beta_at_star'])} ({fmt(r['se_beta_at_star'])})",
            "ΔlogLL": fmt(r["logll_improvement"], 1),
        })
    s += md_table(pd.DataFrame(rows))
    s += "\n*α* = 1: pure W_geo; α* = 0: pure W_bank. ΔlogLL = improvement over best single-W endpoint.*\n"
    return s


def sec_jtest():
    df = load(OUT / "jtest_credit_results.csv")
    if df is None:
        return ""
    s = section("5. J-Test — Non-Nested Model Selection")

    s += """
**Source:** `analysis/model_selection/jtest.py`

**References:** Davidson & MacKinnon (1981) *Several Tests for Model Specification in the Presence of Alternative Hypotheses*

---

**Purpose:** Tests whether the spatial prediction component of one W (the alternative) has additional explanatory power when the model is estimated under another W (the null).

**Spatial prediction component** (from fitted SEM residuals):

```
m̂ = λ̂ · W · û = û - ê_filtered      ...(9)
```

where `û` are within-demeaned OLS residuals and `ê_filtered = (I - λ̂W)û` are the spatially-filtered residuals stored in `res.e_filtered`.

**J-test direction** (H₀: W_null, H₁: W_alt):

```
Step 1: Estimate SEM under W_alt → extract m̂_alt via (9)
Step 2: Augment regressor matrix: X_aug = [X | m̂_alt]     ...(10)
Step 3: Estimate SEM under W_null with (y, X_aug)
Step 4: J-statistic = z-ratio on the coefficient of m̂_alt in Step 3
```

Both directions are tested for each W pair (H₀=W_null, H₁=W_alt) and (H₀=W_alt, H₁=W_null).

**Decision rule:**
- Direction 1 rejects only → W_alt preferred
- Direction 2 rejects only → W_null preferred
- Both reject → Indeterminate (both W specifications capture independent spatial structure)
- Neither rejects → Indeterminate (neither is informative)

**Results:**

"""
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Sample": r["sample"],
            "Null W": r["w_null"],
            "Alt W": r["w_alt"],
            "Dir 1 z": f"{fmt(r['d1_j_z'], 2)}{stars(r['d1_j_p'])}",
            "Dir 2 z": f"{fmt(r['d2_j_z'], 2)}{stars(r['d2_j_p'])}",
            "Conclusion": r["conclusion"],
        })
    s += md_table(pd.DataFrame(rows))
    s += "\n*\\*\\*\\* p<0.01. Both directions significant → neither W matrix dominates.*\n"
    return s


def sec_lm_diag():
    df = load(OUT / "lm_diagnostics_credit.csv")
    if df is None:
        return ""
    s = section("6. LM Diagnostics — Credit OLS Residuals")

    s += """
**Source:** `analysis/model_selection/lm_tests.py`

**References:** Anselin (1988) §12; Anselin et al. (1996) *Simple Diagnostic Tests for Spatial Dependence*; Burridge (1980)

---

**Pre-estimation:** Apply the **two-way within transformation** (eq. 3) to all variables so that OLS residuals match the two-way FE mean specification in §1.

**OLS residuals (two-way FE):**

```
Δln(loans)̃_it = β̂_OLS · L̃inter_bra_it + ũ_it
```

**LM error statistic** (Anselin 1988, eq. 12.7):

```
LM_error = [ũ'Wu / σ̂²]² / tr(W'W + W²)      ...(11)
```

under H₀: λ = 0, LM_error ~ χ²(1).

**LM lag statistic** (Anselin et al. 1996):

```
LM_lag = [ũ'Wỹ / σ̂²]² / [(WXβ̂)'M(Wβ̂X)/σ̂² + tr(W'W + W²)]      ...(12)
```

under H₀: ρ = 0, LM_lag ~ χ²(1).

**Robust versions** (Anselin et al. 1996): rLM_error and rLM_lag correct each statistic for the presence of the other spatial misspecification.

**Panel implementation:** The block-diagonal panel weight matrix `W_NT = I_T ⊗ W` (Kronecker product) is passed to `spreg.panel_LMerror` and `spreg.panel_LMlag`. Implemented in `scipy.sparse.kron(I_T, W)`.

**Decision tree** (Anselin 1988, p. 103):
1. If LM_error sig, LM_lag not → SEM preferred
2. If LM_lag sig, LM_error not → SAR preferred
3. If both sig → use rLM_error vs rLM_lag; larger robust statistic indicates preferred model
4. If neither sig → no spatial dependence

**Results:**

"""
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Sample": r["sample"],
            "W": r["w_matrix"],
            "LM_error": f"{fmt(r['LM_error'], 1)}{stars(r['p_LM_error'])}",
            "LM_lag": f"{fmt(r['LM_lag'], 1)}{stars(r['p_LM_lag'])}",
            "rLM_error": f"{fmt(r['rLM_error'], 1)}{stars(r['p_rLM_error'])}",
            "rLM_lag": f"{fmt(r['rLM_lag'], 1)}{stars(r['p_rLM_lag'])}",
            "Decision": r["decision"],
        })
    s += md_table(pd.DataFrame(rows))
    s += "\n*\\*p<0.10, \\*\\*p<0.05, \\*\\*\\*p<0.01*\n"
    return s


def sec_conley():
    df = load(OUT / "conley_se_comparison.csv")
    if df is None:
        return ""
    s = section("7. Conley SE Comparison")

    s += """
**Source:** `analysis/inference/conley_se.py`

**References:** Conley (1999) *GMM Estimation with Cross Sectional Dependence*; Colella, Lalive, Seyed & Tschopp (2019) *Inference with Arbitrary Clustering*

---

**Model** (same point estimate β across all four estimators):

```
Δln(loans)̃_it = β · L̃inter_bra_it + ũ_it     (two-way FE OLS)
```

**General sandwich variance** (White 1980):

```
Var(β̂) = (X'X)⁻¹ · B̂ · (X'X)⁻¹ · df_correction    ...(13)
```

For the scalar k=1 case: `Var(β̂) = B̂ / (X'X)² · df_correction`

---

**Estimator 1 — State-clustered (Favara & Imbs 2015 baseline):**

```
B_state = Σ_s score_s²
score_s = Σ_{c∈s} Σ_t ũ_ct · x̃_ct              ...(14)
df_corr = G / (G - 1)   (G = number of states)
```

---

**Estimator 2 & 3 — Spatial HAC** (Conley 1999):

```
B_spatial = Σ_t  v_t' W v_t
where v_t[c] = ũ_ct · x̃_ct   (element-wise product)     ...(15)
```

Equivalently: `B_spatial = Σ_t Σ_c Σ_{c'} w_{cc'} ũ_ct ũ_{c't} x̃_ct x̃_{c't}`

`df_corr = NT / (NT - k - 1)` where k = number of non-FE regressors

Applied with W = W_geo (estimator 2) and W = W_bank (estimator 3).

---

**Estimator 4 — Two-way (State + Spatial)** (Colella et al. 2019):

```
B_twoway = B_state + B_spatial(W_bank) - B_OLS     ...(16)
B_OLS    = Σ_{c,t} ũ²_ct · x̃²_ct               (HC1 overlap term)
```

Interpretation: `B_twoway` avoids double-counting the diagonal contribution that appears in both `B_state` and `B_spatial`.

---

**95% CI and t-statistic:**

```
CI = [β̂ - 1.96 · SE, β̂ + 1.96 · SE]
t  = β̂ / SE
p  = 2 · Φ(-|t|)     (two-sided normal)          ...(17)
```

**Results:**

"""
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Sample": r["sample"],
            "Estimator": r["estimator"],
            "β": fmt(r["beta"]),
            "SE": fmt(r["se"]),
            "95% CI": f"[{fmt(r['ci_lower'])}, {fmt(r['ci_upper'])}]",
            "t-stat": f"{fmt(r['t_stat'], 2)}{stars(r['p_value'])}",
        })
    s += md_table(pd.DataFrame(rows))
    return s


def sec_fgls():
    df  = load(OUT / "fgls_comparison.csv")
    ht  = load(OUT / "fgls_hausman_moran.csv")
    if df is None:
        return ""
    s = section("8. FGLS Comparison")

    s += """
**Source:** `analysis/inference/fgls.py`

**References:** Feasible GLS: Greene (2003) *Econometric Analysis*, Ch. 10; Hausman (1978); Conley (1999); Moran (1950)

---

**Model:**

```
Δln(loans)_it = β · Linter_bra_it + α_i + τ_t + ε_it
```

**Spatial filter** (Applied period-by-period):

```
A = I - λ · W                                       ...(18)
ỹ_A[t] = A · ỹ[t],   X̃_A[t] = A · X̃[t]
```

where `ỹ[t]` and `X̃[t]` are the two-way FE within-demeaned arrays for period t (eq. 3).

**Stability condition** (required before applying filter):

```
|λ| < 1 / ρ(W)    where ρ(W) = max eigenvalue of W  ...(19)
```

(Gershgorin bound used as fallback: `ρ(W) ≤ max_i Σ_j |w_ij|`)

**FGLS point estimate** (scalar k=1 case):

```
β_FGLS = (X̃_A' X̃_A)⁻¹ X̃_A' ỹ_A = (X̃_A' ỹ_A) / (X̃_A' X̃_A)    ...(20)
```

**State-clustered SE (FGLS)** — unfiltered residuals, filtered regressor as bread:

```
ξ̂_it = ỹ_it - x̃_it · β_FGLS             [unfiltered residuals]
bread = Σ_{c,t} (X̃_A,ct)²
score_s = Σ_{c∈s,t} X̃_A,ct · ξ̂_ct
meat = Σ_s score_s²
Var(β̂_FGLS) = meat / bread² · G/(G-1)        ...(21)
```

**Four estimators:**
- **OLS:** A = I (no filter), λ = 0
- **FGLS W_geo:** λ = λ̂_geo from Panel_FE_Error (§1)
- **FGLS W_bank:** λ = λ̂_bank from Panel_FE_Error (§1)
- **FGLS W\*:** Composite W* = 0.20·W_geo + 0.80·W_bank (Pair A optimal α* from §4), λ = λ̂ at α* = 0.20

**Hausman test** (H₀: OLS = FGLS W_bank, i.e., no spatial misspecification bias):

```
H = (β_OLS - β_FGLS)² / (SE_OLS² - SE_FGLS²) ~ χ²(1)    ...(22)
```

Fallback when denominator ≤ 0 (non-positive definite variance difference):

```
t_diff = (β_OLS - β_FGLS) / sqrt(SE_OLS² + SE_FGLS²)     ...(23)
```

**Residual Moran's I** on FGLS W_bank filtered residuals (year-by-year), using permutation test (eq. 25 below).

**FGLS β estimates:**

"""
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Sample": r["sample"],
            "Estimator": r["estimator"],
            "β": f"{fmt(r['beta'])} ({fmt(r['se'])}){stars(r['p_value'])}",
            "95% CI": f"[{fmt(r['ci_lower'])}, {fmt(r['ci_upper'])}]",
            "N counties": fmt(int(r["N"])),
        })
    s += md_table(pd.DataFrame(rows))

    if ht is not None:
        s += "\n**Hausman test and residual Moran's I:**\n\n"
        ht_rows = []
        for _, r in ht.iterrows():
            ht_rows.append({
                "Sample": r["sample"],
                "Hausman note": r["hausman_note"],
                "t_diff": f"{fmt(r['t_diff'], 3)}{stars(r['p_diff'])}",
                "Mean Moran's I (resid)": fmt(r["moran_mean_I"]),
                "p (Moran, mean)": pval_str(r["moran_mean_p"]),
            })
        s += md_table(pd.DataFrame(ht_rows))
    return s


def sec_sar_robustness():
    df = load(OUT / "sar_robustness_credit.csv")
    if df is None:
        return ""
    s = section("9. SAR Robustness Check")

    s += """
**Source:** `analysis/sar/sar_credit.py`

**References:** Anselin (1988) Ch. 6; Cliff & Ord (1981) *Spatial Processes*

---

**SAR (Spatial Lag) model:**

```
Δln(loans)_it = ρ · (W Δln(loans))_it + β · Linter_bra_it + α_i + τ_t + u_it    ...(24)
```

where the spatial lag `(WΔln(loans))_it` is the **outcome variable** spatially lagged, as opposed to the SEM's error lag.

Estimated via `spreg.Panel_FE_Lag`, which instruments `Wy` internally using spatial lags of the exogenous regressors (Kelejian & Prucha 1998).

**Rho gap test** (same structure as eq. 4):

```
Δρ = ρ̂_bank - ρ̂_geo
z = Δρ / sqrt(SE(ρ̂_bank)² + SE(ρ̂_geo)²)
```

**Results:**

"""
    rows = []
    for _, r in df.iterrows():
        gap_str = f"{fmt(r['delta_rho'])} (z={fmt(r['z_stat'], 2)}){stars(0.001)}" \
            if pd.notna(r.get("delta_rho", np.nan)) and not pd.isna(r.get("delta_rho")) else "—"
        rows.append({
            "Sample": r["sample"],
            "W": r["w_matrix"],
            "ρ (spatial lag)": f"{fmt(r['rho'])} ({fmt(r['rho_se'])}){stars(0.001)}",
            "β (deregulation)": f"{fmt(r['beta_D'])} ({fmt(r['beta_D_se'])}){stars(0.001)}",
            "Δρ (bank−geo)": gap_str,
            "N obs": fmt(int(r["n_obs"])),
        })
    s += md_table(pd.DataFrame(rows))
    s += "\n*SAR ρ is analogous to SEM λ but governs the outcome (not the error) process.*\n"
    return s


def sec_sar_iv():
    df = load(OUT / "sar_iv_results.csv")
    if df is None:
        return ""
    s = section("10. IV-SAR Estimation")

    s += """
**Source:** `analysis/sar/sar_iv_credit.py`

**References:** Kelejian & Prucha (1998) *A Generalised Spatial Two-Stage Least Squares Procedure*; Moran (1950)

---

**Model** (same as SAR, eq. 24):

```
y_it = ρ · (W y)_it + β · Linter_bra_it + α_i + τ_t + ξ_it
```

The spatial lag `(Wy)` is **endogenous** — correlated with ξ through the feedback loop.

**Instruments** (Kelejian & Prucha 1998):

```
q1_it = (W · Linter_bra)_it      (first spatial lag of deregulation)
q2_it = (W² · Linter_bra)_it     (second spatial lag of deregulation)
```

**2SLS estimation steps** (all on within-demeaned data, eq. 3):

```
Step 1: two-way within-transform all variables → ỹ, ẑ = Wỹ, x̃, q̃1, q̃2
Step 2: First stage OLS (instrument relevance):
         ẑ ~ [x̃, q̃1, q̃2]  →  ẑ_hat
         F-stat tests H₀: α_WD = α_W2D = 0    ...(25)
Step 3: Second stage OLS:
         ỹ ~ [ẑ_hat, x̃]  →  (ρ̂_IV, β̂_IV)    ...(26)
Step 4: 2SLS SEs using ORIGINAL spatial lag (not ẑ_hat) for residuals:
         ξ̂ = ỹ - ẑ · (ρ̂, β̂)'   [uses Z_orig, not Z_hat]
Step 5: State-clustered sandwich (eq. 14) applied to Z_hat as bread:
         score_s = Σ_{obs∈s} Z_hat_obs' · ξ̂_obs
         B = Σ_s score_s · score_s'
         V = (Z_hat' Z_hat)⁻¹ · B · (Z_hat' Z_hat)⁻¹ · G/(G-1)    ...(27)
Step 6: Moran's I on SAR residuals (year-by-year, eq. 28 below)
```

**Results:**

"""
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Sample": r["sample"],
            "W": r["W"],
            "ρ (IV)": f"{fmt(r['rho'])} ({fmt(r['rho_se'])})",
            "β": f"{fmt(r['beta'])} ({fmt(r['beta_se'])})",
            "1st stage F": fmt(r["first_stage_F"], 1),
            "Moran I (resid, mean)": fmt(r["residual_moran_i_mean"]),
            "N obs": fmt(int(r["N_obs"])),
        })
    s += md_table(pd.DataFrame(rows))
    s += "\n*IV-SAR can be unstable when first-stage F < 10. Non-border sample instruments are weak.*\n"
    return s


def sec_bank_interstate():
    df = load(OUT / "bank_interstate_credit_results.csv")
    if df is None:
        return ""
    s = section("11. Endogeneity Check — Interstate Bank Links Only")

    s += """
**Source:** `analysis/sem/sem_link_restrictions.py`

**References:** Favara & Imbs (2015, Appendix A); Rice & Strahan (2010)

---

**Model:** Same as §1, equations (1)–(2), using `W_bank_interstate`.

**Matrix construction:**

```
W_bank_interstate[i,j] = W_bank[i,j]  if state(i) ≠ state(j)
                        = 0            if state(i) = state(j)     ...(28)
```

followed by row-standardisation.

**Motivation:** Within-state bank links may reflect economic sorting (endogeneity). Cross-state links isolate post-IBBEA deregulation and are less subject to within-state omitted-variable bias.

**Identification prediction:** If λ_interstate ≈ 0 in the border-MSA sample, the estimated bank-network channel is spurious (driven by geographic proximity or within-state sorting rather than cross-state deregulation).

**Results:**

"""
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Model": r["model"],
            "β": f"{fmt(r.get('beta', r.get('beta_D','')))} ({fmt(r.get('se_beta', r.get('se_beta_D','')))}){stars(r.get('p_beta', 0.001))}",
            "λ": f"{fmt(r['lam'])} ({fmt(r['se_lam'])}){stars(r['p_lam'])}",
            "N obs": fmt(int(r["n_obs"])),
        })
    s += md_table(pd.DataFrame(rows))
    return s


def sec_bank_intrastate():
    df = load(OUT / "bank_intrastate_credit_results.csv")
    if df is None:
        return ""
    s = section("12. Placebo Check — Intrastate Bank Links Only")

    s += """
**Source:** `analysis/sem/sem_link_restrictions.py`

**References:** Favara & Imbs (2015); Rice & Strahan (2010)

---

**Model:** Same as §1, equations (1)–(2), using `W_bank_intrastate`.

**Matrix construction** (complement of eq. 28):

```
W_bank_intrastate[i,j] = W_bank[i,j]  if state(i) = state(j)
                        = 0            if state(i) ≠ state(j)     ...(29)
```

followed by row-standardisation.

**Placebo logic:** If the bank-network channel captures genuine cross-state deregulation spillovers, then `W_bank_intrastate` (which only preserves within-state links) should yield λ_intrastate < λ_bank. A large λ_intrastate would suggest the bank channel proxies intra-state economic correlation rather than the IBBEA deregulation shock.

**Results:**

"""
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Model": r["model"],
            "β": f"{fmt(r.get('beta', r.get('beta_D','')))} ({fmt(r.get('se_beta', r.get('se_beta_D','')))}){stars(r.get('p_beta', 0.001))}",
            "λ": f"{fmt(r['lam'])} ({fmt(r['se_lam'])}){stars(r['p_lam'])}",
            "N obs": fmt(int(r["n_obs"])),
        })
    s += md_table(pd.DataFrame(rows))
    return s


def sec_bank_nongeo():
    df = load(OUT / "bank_nongeo_credit_results.csv")
    if df is None:
        return ""
    s = section("13. Non-Geographic Bank Network")

    s += """
**Source:** `analysis/sem/sem_link_restrictions.py`

**References:** Anselin (1988); Favara & Imbs (2015)

---

**Model:** Same as §1, equations (1)–(2), using `W_bank_nonGeo`.

**Matrix construction:**

```
W_bank_nonGeo[i,j] = W_bank[i,j]  if W_geo[i,j] = 0
                   = 0             if W_geo[i,j] > 0          ...(30)
```

followed by row-standardisation. This removes all bank-network links that coincide with geographic adjacency, isolating the purely non-geographic component.

**Purpose:** Tests whether the bank-network spatial channel is genuinely distinct from geographic proximity. If λ_nonGeo is large and significant, bank spillovers operate independently of geographic contiguity.

**Results:**

"""
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Model": r["model"],
            "β": f"{fmt(r.get('beta', r.get('beta_D','')))} ({fmt(r.get('se_beta', r.get('se_beta_D','')))}){stars(r.get('p_beta', 0.001))}",
            "λ": f"{fmt(r['lam'])} ({fmt(r['se_lam'])}){stars(r['p_lam'])}",
            "N obs": fmt(int(r["n_obs"])),
        })
    s += md_table(pd.DataFrame(rows))
    return s


def sec_knn():
    df = load(OUT / "knn_sweep_credit_results.csv")
    if df is None:
        return ""
    # Detect column naming convention (nb vs border)
    has_nb     = "lam_geo_nb"     in df.columns
    has_border = "lam_geo_border" in df.columns
    lam_nb_col = "lam_geo_nb"     if has_nb else "lam_geo_border"
    knn_nb_col = "lam_bank_knn_nb" if "lam_bank_knn_nb" in df.columns else "lam_bank_knn_border"
    gap_nb_col = "gap_nb"         if "gap_nb" in df.columns else "gap_border"

    s = section("14. KNN Crossover Sweep")

    s += """
**Source:** `analysis/sem/sem_knn_sweep.py`

**References:** Anselin (1988); LeSage & Pace (2009)

---

**Model:** Same as §1, equations (1)–(2), replacing W_bank with `W_bank_knn_k`.

**KNN matrix construction:**

```
W_bank_knn_k[i, :] retains only the top-k entries of W_bank[i, :]
                   (by weight magnitude), sets all others to zero     ...(31)
```

followed by row-standardisation. k sweeps from 1 to 20.

**Purpose:** Finds the crossover k where `λ_knn(k) > λ_geo`, establishing the bank-network density at which spatial error dependence matches geographic contiguity. W_geo has density ≈ 0.33% (mean ~3.4 neighbours per county).

**Density** = non-zero off-diagonal links / (N · (N-1))

**Reference lambdas:** λ_geo (full) = 0.1801, λ_geo (border) = 0.1701 (from §1).

**Results (selected k values):**

"""
    show_ks = [1, 2, 3, 5, 8, 10, 15, 20]
    sub = df[df["k"].isin(show_ks)].copy()
    rows = []
    for _, r in sub.iterrows():
        rows.append({
            "k": int(r["k"]),
            "Density (%)": fmt(r["density"] * 100, 3),
            "λ_knn (full)": fmt(r["lam_bank_knn"]),
            "gap (full)": fmt(r["gap"]),
            "λ_knn (border)": fmt(r[knn_nb_col]),
            "gap (border)": fmt(r[gap_nb_col]),
        })
    s += md_table(pd.DataFrame(rows))

    cf = df[df["gap"] > 0]["k"].min() if (df["gap"] > 0).any() else ">20"
    cb = df[df[gap_nb_col] > 0]["k"].min() if (df[gap_nb_col] > 0).any() else ">20"
    s += f"\n**Crossover k:** Full sample = **k={cf}**; Border sample = **k={cb}**\n"
    return s


def sec_spatial_multiplier():
    df = load(OUT / "spatial_multiplier_decomposition.csv")
    if df is None:
        return ""
    s = section("15. SEM Spatial Multiplier Decomposition")

    s += """
**Source:** `analysis/sar/multiplier_decomposition.py`

**References:** LeSage & Pace (2009) *Introduction to Spatial Econometrics*, Ch. 2

---

**Framework:** Under the SEM (eq. 2), the reduced-form error covariance satisfies:

```
u = (I - λW)⁻¹ ε  ≡  S · ε
```

The **impact matrix** `S = (I - λW)⁻¹` captures total statistical co-movement (not structural causal transmission) in the disturbance process between county pairs.

**Power-series expansion** (converges when |λ| < 1/ρ(W)):

```
S = I + λW + λ²W² + λ³W³ + ...     ...(32)
```

Decomposition:
- k=0 term (I): **direct / own-county retention**
- k≥1 terms (λᵏWᵏ): **k-th order network transmission**

**Scalar summaries:**
```
Total multiplier  = Σ_k λᵏ tr(Wᵏ)/N  ≈ 1/(1-λ) for symmetric W
Avg indirect effect = (Total multiplier) - (Avg direct)
Indirect share (%) = 100 · Avg indirect / Total multiplier     ...(33)
```

**Average reach (km):** Weighted average distance between counties, with weights proportional to the k-th order impact share. Requires county centroid coordinates.

Series terminates when `||λᵏWᵏ||_∞ < 10⁻⁴` (convergence threshold) or k > 600.

**Results:**

"""
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Outcome": r["outcome"],
            "W": r["w_matrix"],
            "λ": fmt(r["lambda"]),
            "Total multiplier": fmt(r["total_multiplier"], 3),
            "Avg direct": fmt(r["avg_direct"], 3),
            "Avg indirect": fmt(r["avg_indirect"], 3),
            "Indirect share (%)": fmt(r["indirect_share_pct"], 1),
            "Avg reach (km)": fmt(r["avg_reach_km"], 0),
        })
    s += md_table(pd.DataFrame(rows))
    return s


def sec_sar_multiplier():
    df = load(OUT / "multiplier_counterfactual.csv")
    if df is None:
        return ""
    s = section("16. SAR Multiplier Counterfactual")

    s += """
**Source:** `analysis/extensions/multiplier_counterfactual.py`

**References:** LeSage & Pace (2009), Ch. 2; Kelejian & Prucha (1998)

---

**Framework:** For the SAR model (eq. 24) with row-stochastic W (W·1 = 1), the scalar total multiplier is:

```
M = 1 / (1 - ρ)     ...(34)
```

**Effect decomposition** for a unit deregulation shock (ΔD = 1):

```
Direct effect   = β_D                       (own-county, no network)
Total effect    = β_D / (1 - ρ) = β_D · M   (own-county + network)
Indirect effect = Total - Direct = β_D · ρ / (1 - ρ)    ...(35)
```

**Parametric bootstrap CI** (1000 draws, seed = 0):

```
(ρ*, β*) ~ N([ρ̂, β̂], diag([SE_ρ², SE_β²]))    ...(36)
```

Draws with |ρ*| ≥ 1 are discarded (SAR stability condition requires |ρ| < 1/ρ(W) ≤ 1). CIs = [5th, 95th] percentile of accepted draws.

**Results (selected W matrices):**

"""
    key_ws = ["W_geo", "W_bank", "W_bank_count", "W_bank_binary", "W_bank_knn4"]
    sub = df[df["w_matrix"].isin(key_ws)].copy()
    rows = []
    for _, r in sub.iterrows():
        if abs(float(r["rho"])) >= 1:
            continue
        rows.append({
            "Sample": r["sample"],
            "W": r["w_matrix"],
            "ρ": fmt(r["rho"]),
            "M = 1/(1-ρ)": fmt(r["multiplier"], 3),
            "Direct (β)": fmt(r["point_direct"]),
            "Indirect": fmt(r["point_indirect"]),
            "Total": fmt(r["point_total"]),
            "Total 90% CI": f"[{fmt(r['p5_tot'])}, {fmt(r['p95_tot'])}]",
        })
    s += md_table(pd.DataFrame(rows))
    s += "\n*Bootstrap 90% CIs from 1000 draws. Draws with |rho| >= 1 discarded.*\n"
    return s


def sec_slx():
    df = load(OUT / "slx_exposure_results.csv")
    if df is None:
        return ""
    s = section("17. SLX Exposure — Bank-Network Deregulation Spillover")
    s += """
**Source:** `analysis/extensions/slx_exposure.py`

**References:** Halleck Vega & Elhorst (2015) *The SLX Model*; Conley (1999); Colella et al. (2019)

---

**SLX exposure variable:**

```
E_{c,t} = sum_{c'} w^{interstate}_{cc'} * D_{s(c'),t}     ...(37)
```

where `D = Linter_bra` (state-level deregulation index) and `w^{interstate}` is the row-standardised
interstate bank-network weight matrix (cross-state links only, eq. 28).

**Regression specifications** (two-way county + year FE, within estimator):

```
(1) Base:    delta_ln(loans)_it = beta_D * D_it + eps_it
(2) SLX:     delta_ln(loans)_it = beta_D * D_it + beta_E * E_it + eps_it     ...(38)
(3) Placebo: delta_ln(loans_pl)_it = beta_D * D_it + beta_E * E_it + eps_it
```

where `loans_pl` = non-bank mortgage lending.

**Permutation test** for beta_E (999 permutations, seed=42):

```
p_perm = #{|beta^pi_E| >= |beta_E_obs|} / 999     ...(39)
```

**Results:**

"""
    rows = []
    for _, r in df.iterrows():
        beta_e_str = (f"{fmt(r['beta_E'])} ({fmt(r['se_E'])}){stars(r.get('p_E', 1.0))}"
                      if pd.notna(r.get("beta_E", np.nan)) else "—")
        perm_str = fmt(r["p_perm"]) if pd.notna(r.get("p_perm", np.nan)) else "—"
        rows.append({
            "Sample": r["sample"],
            "Spec": r["spec"],
            "DV": r["dv"],
            "beta_D (SE)": f"{fmt(r['beta_D'])} ({fmt(r['se_D'])}){stars(r['p_D'])}",
            "beta_E (SE)": beta_e_str,
            "p_perm": perm_str,
            "N obs": fmt(int(r["n_obs"])),
        })
    s += md_table(pd.DataFrame(rows))
    s += "\n*Standard errors in parentheses. p_perm: permutation-based p-value for beta_E.*\n"
    return s


def sec_moran_diagnostics():
    df = load(DIAG / "moran_i_results.csv")
    if df is None:
        return ""
    s = section("18. Moran's I — Pre-Estimation Diagnostics")
    s += """
**Source:** `analysis/diagnostics/moran_baseline.py`

**References:** Moran (1950) *Notes on Continuous Stochastic Phenomena*; Cliff & Ord (1981) *Spatial Processes*

---

**Baseline OLS** (two-way FE, replicates Favara & Imbs 2015):

```
delta_ln(loans)_it = beta * Linter_bra_it + controls + a_i + tau_t + u_it   ...(40)
```

**Moran's I statistic** (Moran 1950):

```
I_t = (N / S_0) * (u_t' W u_t) / (u_t' u_t)     ...(41)
```

S_0 = sum_i sum_j w_ij.

**Expected value and z-score** under H0 (no spatial autocorrelation):

```
E[I] = -1 / (N - 1)
z_I  = (I - E[I]) / sqrt(Var[I])                 ...(42)
```

Permutation p-values (999 draws):

```
p_perm = #{|I^pi| >= |I_obs|} / 999              ...(43)
```

**Summary (county-level, mean over years):**

"""
    summary = (df[df["level"] == "county"]
               .groupby("outcome")
               .agg(mean_I=("moran_I", "mean"),
                    sig_years=("significant", "sum"),
                    total_years=("significant", "count"))
               .reset_index())
    rows = [{"Outcome": r["outcome"],
             "Mean Moran's I": fmt(r["mean_I"]),
             "Significant years / total": f"{int(r['sig_years'])}/{int(r['total_years'])}"}
            for _, r in summary.iterrows()]
    s += md_table(pd.DataFrame(rows))

    s += "\n**Year-by-year (county level):**\n\n"
    sub = df[df["level"] == "county"][
        ["year", "outcome", "moran_I", "z_score", "p_value", "significant"]].copy()
    rows2 = [{"Year": int(r["year"]), "Outcome": r["outcome"],
               "Moran's I": fmt(r["moran_I"]), "z": fmt(r["z_score"], 2),
               "p": pval_str(r["p_value"]), "Sig": "Y" if r["significant"] else ""}
             for _, r in sub.iterrows()]
    s += md_table(pd.DataFrame(rows2))
    return s


def sec_moran_composite():
    df = load(DIAG / "moran_i_composite_credit_summary.csv")
    if df is None:
        return ""
    s = section("19. Moran's I Under Composite W — Credit Residuals")
    s += """
**Source:** `analysis/diagnostics/moran_bank_variants.py`

**References:** Moran (1950); LeSage & Pace (2009)

---

Moran's I (eq. 41) on credit SEM residuals under composite W matrices from section 4.
A lower Moran's I at alpha* validates the composite weighting approach.

**Summary:**

"""
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Sample": r["sample"],
            "W matrix": r["w_matrix"],
            "alpha": fmt(r["alpha"]),
            "Mean Moran's I": fmt(r["mean_moran_I"]),
            "Sig. years": f"{int(r['significant_years'])}/{int(r['n_years'])}",
        })
    s += md_table(pd.DataFrame(rows))
    return s


def sec_moran_wbank():
    df = load(DIAG / "moran_i_wbank_summary.csv")
    if df is None:
        return ""
    s = section("20. Moran's I — Bank Network W Variants")
    s += """
**Source:** `analysis/diagnostics/moran_bank_variants.py`

**References:** Moran (1950); Anselin (1988)

---

Moran's I (eq. 41) on SEM residuals for all W_bank specifications and KNN truncations k=1...20.
Lower Moran's I = better residual whitening.

**Summary (non-KNN matrices):**

"""
    non_knn = df[~df["w_matrix"].str.contains("knn", na=False)].copy()
    rows = [{"Outcome": r["outcome"], "W matrix": r["w_matrix"],
              "Mean Moran's I": fmt(r["mean_moran_I"]),
              "Median Moran's I": fmt(r["median_moran_I"]),
              "Sig. years": f"{int(r['significant_years'])}/{int(r['n_years'])}",
              "Density": fmt(r["density"] * 100, 3) + "%"}
             for _, r in non_knn.iterrows()]
    s += md_table(pd.DataFrame(rows))
    return s


def sec_bank_overlap():
    stats = load(OUT / "geo_bank_overlap_stats.csv")
    dist  = load(OUT / "geo_bank_overlap_distribution.csv")
    if stats is None:
        return ""
    s = section("21. Bank-Geography Network Overlap")
    s += """
**Source:** `analysis/diagnostics/network_overlap.py`

**References:** Favara & Imbs (2015, Data Appendix)

---

**Overlap fraction** per county i:

```
overlap_i = |top5_bank_nbrs(i) ∩ geo_nbrs(i)| / 5     ...(44)
```

where `top5_bank_nbrs(i)` = 5 counties with highest W_bank weight from i.
Islands (zero geographic neighbours) excluded.
Motivates the W_bank_nonGeo construction (eq. 30).

**Summary statistics:**

"""
    stat_col = ("statistic" if "statistic" in stats.columns
                else stats.columns[0])
    rows = [{"Statistic": r[stat_col], "Value": fmt(r["value"])}
            for _, r in stats.iterrows()]
    s += md_table(pd.DataFrame(rows))

    if dist is not None:
        s += "\n**Distribution of overlap fractions:**\n\n"
        rows2 = [{c: str(r[c]) for c in dist.columns} for _, r in dist.iterrows()]
        s += md_table(pd.DataFrame(rows2))
    return s


def sec_lambda_time():
    df = load(OUT / "lambda_time_yearspecific_credit_results.csv")
    tr = load(OUT / "lambda_time_yearspecific_credit_trends.csv")
    if df is None:
        return ""
    s = section("22. Time-Varying Lambda — Year-Specific W_bank")
    s += """
**Source:** `analysis/deprecated/lambda_time_yearspecific_credit.py`

**References:** IBBEA (1994); Rice & Strahan (2010); Favara & Imbs (2015)

---

**Model:** Year-specific cross-sectional ML_Error estimated independently for each year t:

```
delta_ln(loans)_{i,t} = beta_t * Linter_bra_{i,t} + u_{i,t}
u_{i,t} = lambda_t * (W_t u_t) + eps_{i,t}     ...(45)
```

W_t = year-specific W_bank matrix.

**Trend regression:**

```
lambda_t = a + b * t + eta_t     ...(46)
```

**Gap (bank minus geo):**

```
gap_t = lambda_bank,t - lambda_geo,t     ...(47)
```

**Results (lambda by year):**

"""
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Year": int(r["year"]),
            "lambda_geo (SE)": f"{fmt(r['lam_geo'])} ({fmt(r['se_lam_geo'])})",
            "lambda_bank (SE)": f"{fmt(r['lam_bank'])} ({fmt(r['se_lam_bank'])})",
            "Gap": fmt(r["gap"]),
            "N geo": fmt(int(r["n_geo"])),
            "N bank": fmt(int(r["n_bank"])),
        })
    s += md_table(pd.DataFrame(rows))

    if tr is not None:
        s += "\n**Trend regression slopes (eq. 46):**\n\n"
        trows = [{"Series": r["series"],
                  "Slope b": fmt(r["slope"]),
                  "Intercept a": fmt(r["intercept"]),
                  "R2": fmt(r["r2"], 3),
                  "p(H0: b=0)": pval_str(r["p_value"])}
                 for _, r in tr.iterrows()]
        s += md_table(pd.DataFrame(trows))
    return s


# ══════════════════════════════════════════════════════════════════════════════
# MASTER REPORT
# ══════════════════════════════════════════════════════════════════════════════

def build_report():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    toc = (
        "## Table of Contents\n\n"
        "1.  [Main SEM — Credit Growth](#1-main-sem--credit-growth)\n"
        "2.  [Main SEM — House Price Growth](#2-main-sem--house-price-growth)\n"
        "3.  [Four-W Comparison](#3-four-w-comparison--credit-growth)\n"
        "4.  [Composite W Profile — Credit](#4-composite-w-profile--credit-growth)\n"
        "5.  [J-Test — Model Selection](#5-j-test--non-nested-model-selection)\n"
        "6.  [LM Diagnostics](#6-lm-diagnostics--credit-ols-residuals)\n"
        "7.  [Conley SE Comparison](#7-conley-se-comparison)\n"
        "8.  [FGLS Comparison](#8-fgls-comparison)\n"
        "9.  [SAR Robustness Check](#9-sar-robustness-check)\n"
        "10. [IV-SAR Estimation](#10-iv-sar-estimation)\n"
        "11. [Endogeneity Check — Interstate Links](#11-endogeneity-check--interstate-bank-links-only)\n"
        "12. [Placebo — Intrastate Links](#12-placebo-check--intrastate-bank-links-only)\n"
        "13. [Non-Geographic Bank Network](#13-non-geographic-bank-network)\n"
        "14. [KNN Crossover Sweep](#14-knn-crossover-sweep)\n"
        "15. [SEM Spatial Multiplier](#15-sem-spatial-multiplier-decomposition)\n"
        "16. [SAR Multiplier Counterfactual](#16-sar-multiplier-counterfactual)\n"
        "17. [SLX Exposure](#17-slx-exposure--bank-network-deregulation-spillover)\n"
        "18. [Moran's I — Pre-Estimation](#18-morans-i--pre-estimation-diagnostics)\n"
        "19. [Moran's I — Composite W](#19-morans-i-under-composite-w--credit-residuals)\n"
        "20. [Moran's I — Bank Variants](#20-morans-i--bank-network-w-variants)\n"
        "21. [Bank-Geography Overlap](#21-bank-geography-network-overlap)\n"
        "22. [Time-Varying Lambda](#22-time-varying-lambda--year-specific-wbank)\n\n"
    )

    parts = [
        "# Thesis Results Report\n\n",
        f"*Auto-generated by `analysis/generate_report.py` on {ts}.*\n",
        "*Re-run `python analysis/generate_report.py` any time to incorporate new results.*\n\n",
        "---\n\n",
        toc,
        "---\n",
    ]

    builders = [
        sec_panel_fe_credit,
        sec_panel_fe_hpi,
        sec_four_w,
        sec_composite_credit,
        sec_jtest,
        sec_lm_diag,
        sec_conley,
        sec_fgls,
        sec_sar_robustness,
        sec_sar_iv,
        sec_bank_interstate,
        sec_bank_intrastate,
        sec_bank_nongeo,
        sec_knn,
        sec_spatial_multiplier,
        sec_sar_multiplier,
        sec_slx,
        sec_moran_diagnostics,
        sec_moran_composite,
        sec_moran_wbank,
        sec_bank_overlap,
        sec_lambda_time,
    ]

    found = 0
    for fn in builders:
        try:
            chunk = fn()
        except Exception as e:
            chunk = f"\n> **Error in {fn.__name__}:** {e}\n"
        if chunk:
            found += 1
            parts.append(chunk)
            parts.append("\n---\n")

    parts.append(f"\n*Report covers {found}/22 analyses.*\n")
    return "".join(parts)


if __name__ == "__main__":
    report = build_report()
    out_path = OUT / "results_report.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"Report written to {out_path}")
    print(f"  {len(report):,} characters, {report.count(chr(10)):,} lines")
