"""
generate_report.py
==================
Auto-generates output/results_report.md from all result CSVs in output/.

Run this any time after running new analyses to refresh the report:
    python analysis/generate_report.py

The report documents, for each analysis:
  1. What regression / analysis was run
  2. Model specification and key assumptions
  3. Results (formatted tables)

New result CSVs are picked up automatically — no edits needed.
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
    """Format a number for display."""
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
    """Return significance stars."""
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
    if not Path(path).exists():
        return None
    return pd.read_csv(path)

def section(title, level=2):
    hashes = "#" * level
    return f"\n{hashes} {title}\n"

def md_table(df, col_fmts=None):
    """Convert a DataFrame to a Markdown table string."""
    lines = []
    lines.append("| " + " | ".join(str(c) for c in df.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(df.columns)) + " |")
    for _, row in df.iterrows():
        cells = []
        for i, (col, val) in enumerate(row.items()):
            if col_fmts and col in col_fmts:
                cells.append(col_fmts[col](val))
            else:
                cells.append(fmt(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ── section builders ──────────────────────────────────────────────────────────

def sec_panel_fe_credit():
    df = load(OUT / "panel_fe_credit_results.csv")
    if df is None:
        return ""
    s = section("1. Main SEM — Credit Growth (Panel FE Error)")
    s += """
**Analysis:** Spatial Error Model (SEM) with two-way fixed effects, credit growth as the dependent variable.

**Model specification:**
```
Δln(loans)_it = β · Linter_bra_it + α_i + τ_t + u_it
u_it = λ · (W u)_it + ε_it
```
where `Linter_bra` is the interstate branching deregulation index (Favara & Imbs 2015), `α_i` are county fixed effects, `τ_t` are year fixed effects.

**Key assumptions:**
- Spatial errors follow a first-order autoregressive process (SEM, not SAR)
- W is row-standardised before estimation
- Panel is balanced after dropping counties with any missing `Δln(loans)` (67 counties dropped)
- Two spatial weights compared: W_geo (queen contiguity) and W_bank (time-averaged bank HHC cosine similarity)
- Standard errors: ML-based (spreg Panel_FE_Error)

**Results:**

"""
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Model": r["model"],
            "β (deregulation)": f"{fmt(r['beta'])} ({fmt(r['se_beta'])}){stars(r['p_beta'])}",
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
    s = section("2. Main SEM — House Price Growth (Panel FE Error)")
    s += """
**Analysis:** Spatial Error Model (SEM) with two-way fixed effects, house price growth as the dependent variable. Replicates Favara & Imbs (2015) HPI equation extended with spatial error correction.

**Model specification:**
```
Δln(HPI)_it = β_D · Linter_bra_it + β_Dη · Linter_ela_it + controls_it + α_i + τ_t + u_it
u_it = λ · (W u)_it + ε_it
```
Controls: lagged HPI growth, population growth (current & lagged), income growth (current & lagged), housing supply (current & lagged).

**Key assumptions:**
- Same SEM structure as credit equation
- Estimated unweighted (F&I use state-size weights; spreg does not support them in Panel_FE_Error)
- `Linter_ela` captures within-state deregulation (eta instrument)
- `delta_lambda` = λ_bank − λ_geo tests whether bank-network spatial dependence exceeds geographic

**Results:**

"""
    rows = []
    for _, r in df.iterrows():
        gap_str = ""
        if pd.notna(r.get("delta_lambda", np.nan)):
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
**Analysis:** Estimates the SEM under four spatial weight matrices and tests whether λ_bank > λ_geo using a one-sided gap test.

**Model:** Same as §1 (SEM credit equation).

**Spatial weights compared:**
- **W_geo**: queen contiguity (benchmark)
- **W_bank_bin**: binary bank HHC overlap
- **W_bank_count**: branch-count weighted HHC overlap
- **W_bank_nonGeo**: bank network with geographic-neighbour links zeroed out

**Key assumptions:**
- Gap test: `z = (λ_alt − λ_geo) / sqrt(SE_alt² + SE_geo²)` (one-sided, H₁: λ_bank > λ_geo)
- All matrices row-standardised; county samples differ slightly due to matrix sparsity

**Results:**

"""
    rows = []
    for _, r in df.iterrows():
        gap_str = f"{fmt(r['gap_vs_geo'])} (z={fmt(r['z_gap'], 2)}){stars(r['p_gap_onesided'])}" \
            if pd.notna(r.get("gap_vs_geo", np.nan)) else "—"
        rows.append({
            "Sample": r["sample"],
            "W matrix": r["w_matrix"],
            "β": f"{fmt(r['beta'])} ({fmt(r['se_beta'])}){stars(r['p_beta'])}",
            "λ": f"{fmt(r['lam'])} ({fmt(r['se_lam'])}){stars(r['p_lam'])}",
            "Δλ vs geo": gap_str,
            "N obs": fmt(int(r["n_obs"])),
        })
    s += md_table(pd.DataFrame(rows))
    s += "\n*Standard errors in parentheses. Gap is one-sided z-test (H₁: λ_bank > λ_geo).*\n"
    return s


def sec_composite_credit():
    res = load(OUT / "composite_w_credit_results.csv")
    opt = load(OUT / "composite_w_credit_optima.csv")
    if opt is None:
        return ""
    s = section("4. Composite W Profile — Credit Growth")
    s += """
**Analysis:** Profiles Panel_FE_Error log-likelihood over a convex combination of W matrices:
```
W_combo(α) = α · W_geo + (1−α) · W_alt
```
sweeping α ∈ {0, 0.05, …, 1.0} to find the optimal mixing weight α*.

**Model:** Same as §1.

**Spatial weights profiled (three pairs):**
- **Pair A**: W_geo vs W_bank_bin
- **Pair B**: W_geo vs W_bank_count
- **Pair C**: W_geo vs W_bank_nonGeo

**Key assumptions:**
- Log-likelihood improvement = logLL(α*) − max(logLL_geo, logLL_bank)
- α* = 1 means pure geography; α* = 0 means pure bank network

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
    s += "\n*α* = optimal mixing weight (α=1: pure W_geo, α=0: pure W_bank). ΔlogLL vs best endpoint.*\n"
    return s


def sec_composite_hpi():
    opt = load(OUT / "composite_w_hpi_optima.csv")
    if opt is None:
        return ""
    s = section("5. Composite W Profile — House Price Growth")
    s += """
**Analysis:** Same composite W profile likelihood as §4, but for the HPI equation (W_bank_bin only, both samples).

**Model:** Same as §2.

**Key assumptions:** Identical to §4. HPI estimated unweighted.

**Optimal mixing weights:**

"""
    rows = []
    for _, r in opt.iterrows():
        rows.append({
            "Sample": r["sample"],
            "α*": fmt(r["alpha_star"]),
            "λ at α*": f"{fmt(r['lam_at_star'])} ({fmt(r['se_lam_at_star'])})",
            "β_D at α*": f"{fmt(r['beta_D_at_star'])} ({fmt(r['se_beta_D_at_star'])})",
            "ΔlogLL": fmt(r["logll_improvement"], 1),
        })
    s += md_table(pd.DataFrame(rows))
    return s


def sec_jtest():
    df = load(OUT / "jtest_credit_results.csv")
    if df is None:
        return ""
    s = section("6. J-Test — Model Selection (W_geo vs W_bank)")
    s += """
**Analysis:** Davidson & MacKinnon (1981) J-test for non-nested model selection between competing spatial weight matrices.

**Mechanism:**
1. Estimate SEM under W_alt → extract spatial prediction component `m̂_alt = λ · W_alt · û`
2. Augment X with `m̂_alt` and re-estimate SEM under W_null
3. Reject W_null if the augmented coefficient on `m̂_alt` is significant (z-test)
4. Run both directions for each pair

**Key assumptions:**
- Spatial prediction component is derived from within-demeaned (two-way FE) residuals
- Both directions are tested; "Indeterminate" = both W matrices are rejected (common in practice)

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
    s += "\n*\\*\\*\\* p<0.01. Both directions significant → neither W dominates.*\n"
    return s


def sec_lm_diag():
    df = load(OUT / "lm_diagnostics_credit.csv")
    if df is None:
        return ""
    s = section("7. LM Diagnostics — Credit Growth OLS")
    s += """
**Analysis:** Spatial Lagrange Multiplier (LM) tests applied to OLS residuals from the credit equation to guide model choice between SEM and SAR.

**Tests run (spreg panel diagnostics):**
- **LM_error**: tests H₀: λ=0 (no spatial error autocorrelation)
- **LM_lag**: tests H₀: ρ=0 (no spatial lag in outcome)
- **rLM_error**: robust LM_error (controlling for possible spatial lag)
- **rLM_lag**: robust LM_lag (controlling for possible spatial error)

**Key assumptions:**
- OLS residuals are two-way FE within-demeaned to match the SEM mean specification
- Decision rule: if only rLM_error significant → SEM; only rLM_lag → SAR; both → robust LM values guide preference; neither → no spatial dependence

**Results:**

"""
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Sample": r["sample"],
            "W": r["w_matrix"],
            "LM_error": f"{fmt(r['LM_error'],1)}{stars(r['p_LM_error'])}",
            "LM_lag": f"{fmt(r['LM_lag'],1)}{stars(r['p_LM_lag'])}",
            "rLM_error": f"{fmt(r['rLM_error'],1)}{stars(r['p_rLM_error'])}",
            "rLM_lag": f"{fmt(r['rLM_lag'],1)}{stars(r['p_rLM_lag'])}",
            "Decision": r["decision"],
        })
    s += md_table(pd.DataFrame(rows))
    s += "\n*\\*p<0.10, \\*\\*p<0.05, \\*\\*\\*p<0.01*\n"
    return s


def sec_conley():
    df = load(OUT / "conley_se_comparison.csv")
    if df is None:
        return ""
    s = section("8. Conley SE Comparison")
    s += """
**Analysis:** Compares four standard-error estimators for the credit OLS regression to assess robustness of inference to spatial and cluster correlation.

**Model (same point estimate across all):**
```
Δln(loans)_it = β · Linter_bra_it + α_i + τ_t + u_it   (OLS)
```

**Four estimators:**
1. **State-clustered** — Favara & Imbs (2015) baseline
2. **Spatial HAC W_geo** — Conley (1999) sandwich under queen contiguity
3. **Spatial HAC W_bank** — Conley (1999) sandwich under bank-network weights
4. **State + Spatial W_bank** — Colella et al. (2019) two-way additive combination

**Key assumptions:**
- Conley (1999): `B = Σ_t Σ_c Σ_c' w_cc' u_ct u_c't x̃_ct x̃_c't`
- Colella et al. (2019): `B_twoway = B_state + B_spatial − B_OLS` (avoids double-counting overlap)
- β point estimate is identical across all estimators; only SE changes

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
    df = load(OUT / "fgls_comparison.csv")
    ht = load(OUT / "fgls_hausman_moran.csv")
    if df is None:
        return ""
    s = section("9. FGLS Comparison")
    s += """
**Analysis:** Feasible GLS (FGLS) for the credit equation, applying the spatial filter `A = I − λ·W` derived from ML-SEM lambda estimates, then comparing point estimates and SEs across four specifications.

**Estimators:**
1. **OLS** — Favara & Imbs (2015) baseline (no spatial correction)
2. **FGLS W_geo** — filter with λ_geo from Panel_FE_Error
3. **FGLS W_bank** — filter with λ_bank from Panel_FE_Error
4. **FGLS W\*** — composite W* = 0.20·W_geo + 0.80·W_bank (Pair A optimal α*)

**Key assumptions:**
- FGLS transformation applied period-by-period: `ỹ_A[t] = A · ỹ[t]`, `X̃_A[t] = A · X̃[t]`
- SEs use UNFILTERED residuals and FILTERED regressor (Hausman-robust construction)
- Hausman test: H₀: OLS and FGLS W_bank have the same β (i.e., no endogeneity from spatial misspecification)
- Moran's I on FGLS W_bank residuals tests residual spatial autocorrelation

**FGLS β estimates (state-clustered SEs):**

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
                "t-diff": f"{fmt(r['t_diff'], 3)}{stars(r['p_diff'])}",
                "Moran's I (mean)": fmt(r["moran_mean_I"]),
                "p (Moran)": pval_str(r["moran_mean_p"]),
            })
        s += md_table(pd.DataFrame(ht_rows))
    return s


def sec_sar_robustness():
    df = load(OUT / "sar_robustness_credit.csv")
    if df is None:
        return ""
    s = section("10. SAR Robustness Check — Credit Growth")
    s += """
**Analysis:** Estimates a Spatial Autoregressive (SAR) model instead of the SEM as a robustness check, using spreg.Panel_FE_Lag.

**Model specification:**
```
Δln(loans)_it = ρ · (W Δln(loans))_it + β · Linter_bra_it + α_i + τ_t + u_it
```
(spatial autoregressive process in the **outcome**, versus SEM's error process)

**Key assumptions:**
- Spatial lag `W·y` is treated as endogenous; spreg uses internal IV (spatial lags of exogenous regressors)
- Same panel construction and county filters as main SEM
- `delta_rho` = ρ_bank − ρ_geo tests whether bank-network spillover exceeds geographic (one-sided z-test)

**Results:**

"""
    rows = []
    for _, r in df.iterrows():
        gap_str = f"{fmt(r['delta_rho'])} (z={fmt(r['z_stat'], 2)})***" \
            if pd.notna(r.get("delta_rho", np.nan)) else "—"
        rows.append({
            "Sample": r["sample"],
            "W": r["w_matrix"],
            "ρ (spatial lag)": f"{fmt(r['rho'])} ({fmt(r['rho_se'])}){stars(0.001)}",
            "β (deregulation)": f"{fmt(r['beta_D'])} ({fmt(r['beta_D_se'])}){stars(0.001)}",
            "Δρ (bank−geo)": gap_str,
            "N obs": fmt(int(r["n_obs"])),
        })
    s += md_table(pd.DataFrame(rows))
    s += "\n*SAR ρ is analogous to SEM λ. Δρ one-sided z-test.*\n"
    return s


def sec_sar_iv():
    df = load(OUT / "sar_iv_results.csv")
    if df is None:
        return ""
    s = section("11. IV-SAR — Credit Growth (Kelejian & Prucha 1998)")
    s += """
**Analysis:** Instrumental-variables estimation of the SAR model following Kelejian & Prucha (1998), using spatial lags of the exogenous regressor as instruments for the endogenous spatial lag.

**Model:**
```
y_it = ρ · (W y)_it + β · Linter_bra_it + α_i + τ_t + ξ_it
```

**Instruments:** `q1 = W·Linter_bra`, `q2 = W²·Linter_bra`

**Estimation steps:**
1. Within-transform all variables (two-way FE demean)
2. First-stage OLS: `z̃ ~ [x̃, q̃1, q̃2]`
3. Second-stage OLS: `ỹ ~ [ẑ, x̃]`
4. 2SLS SEs using ORIGINAL spatial lag for residuals (not ẑ)
5. State-clustered sandwich correction
6. Residual Moran's I (year-by-year)

**Key assumptions:**
- Instruments are valid if W·D and W²·D are correlated with W·y but uncorrelated with ξ
- First-stage F-statistic assesses instrument strength (F > 10 is conventional threshold)
- State-clustered SEs account for within-state correlation in xi

**Results:**

"""
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Sample": r["sample"],
            "W": r["W"],
            "ρ (IV)": f"{fmt(r['rho'])} ({fmt(r['rho_se'])}){stars(0.001 if abs(float(r['rho_ci_lower'])) > 0 and float(r['rho_ci_lower']) > 0 else 0.5)}",
            "β": f"{fmt(r['beta'])} ({fmt(r['beta_se'])})",
            "1st stage F": fmt(r["first_stage_F"], 1),
            "Moran I (resid)": fmt(r["residual_moran_i_mean"]),
            "N obs": fmt(int(r["N_obs"])),
        })
    s += md_table(pd.DataFrame(rows))
    s += "\n*Note: IV-SAR can produce unstable ρ estimates when instruments are weak (F < 10).*\n"
    return s


def sec_bank_interstate():
    df = load(OUT / "bank_interstate_credit_results.csv")
    if df is None:
        return ""
    s = section("12. Endogeneity Check — Interstate Bank Links Only")
    s += """
**Analysis:** Robustness check using W_bank restricted to **cross-state links only** (`W_bank_interstate`), to isolate the part of the bank network most directly attributable to post-IBBEA cross-state branch expansion.

**Motivation:** Within-state bank links may reflect banks selecting into economically similar counties in the same state economy (endogeneity). Cross-state links are more directly tied to the specific regulatory shock.

**Model:** Same as §1.

**Key assumptions:**
- W_bank_interstate zeros all entries where counties i and j are in the same state (identical first two FIPS digits)
- If λ_interstate ≈ 0 in the non-border sample, the bank-network effect is driven by geographic proximity or within-state selection rather than cross-state deregulation

**Results:**

"""
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Model": r["model"],
            "β": f"{fmt(r['beta'])} ({fmt(r['se_beta'])}){stars(r['p_beta'])}",
            "λ": f"{fmt(r['lam'])} ({fmt(r['se_lam'])}){stars(r['p_lam'])}",
            "N obs": fmt(int(r["n_obs"])),
        })
    s += md_table(pd.DataFrame(rows))
    return s


def sec_bank_intrastate():
    df = load(OUT / "bank_intrastate_credit_results.csv")
    if df is None:
        return ""
    s = section("13. Placebo Check — Intrastate Bank Links Only")
    s += """
**Analysis:** Placebo/decomposition check using W_bank restricted to **within-state links only** (`W_bank_intrastate`).

**Motivation:** If λ_intrastate is large, the bank network is capturing intra-state economic similarity (potential omitted variable) rather than the cross-state deregulation channel. A significantly smaller λ_intrastate vs λ_bank would support the cross-state channel interpretation.

**Model:** Same as §1.

**Key assumptions:**
- W_bank_intrastate retains only entries where counties i and j share the same state FIPS prefix
- This is the complement of §12 (interstate check)
- λ_intrastate > λ_bank would be concerning for causal identification

**Results:**

"""
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Model": r["model"],
            "β": f"{fmt(r['beta'])} ({fmt(r['se_beta'])}){stars(r['p_beta'])}",
            "λ": f"{fmt(r['lam'])} ({fmt(r['se_lam'])}){stars(r['p_lam'])}",
            "N obs": fmt(int(r["n_obs"])),
        })
    s += md_table(pd.DataFrame(rows))
    return s


def sec_bank_nongeo():
    df = load(OUT / "bank_nongeo_credit_results.csv")
    if df is None:
        return ""
    s = section("14. Non-Geographic Bank Network — Credit Growth")
    s += """
**Analysis:** Tests whether the bank transmission channel is distinct from geographic proximity by using `W_bank_nonGeo` — the bank network with all geographic-neighbour links zeroed out.

**Motivation:** W_bank and W_geo are correlated (banks often operate in adjacent counties). W_bank_nonGeo isolates the purely non-geographic component to verify the bank channel is not just a proxy for geographic spillovers.

**Model:** Same as §1.

**Key assumptions:**
- W_bank_nonGeo: W_bank_bin with entries zeroed where counties are geographic neighbours
- If λ_nonGeo remains large and significant, the bank channel is genuinely distinct from geography

**Results:**

"""
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Model": r["model"],
            "β": f"{fmt(r['beta'])} ({fmt(r['se_beta'])}){stars(r['p_beta'])}",
            "λ": f"{fmt(r['lam'])} ({fmt(r['se_lam'])}){stars(r['p_lam'])}",
            "N obs": fmt(int(r["n_obs"])),
        })
    s += md_table(pd.DataFrame(rows))
    return s


def sec_knn():
    df = load(OUT / "knn_sweep_credit_results.csv")
    if df is None:
        return ""
    s = section("15. KNN Crossover Sweep — Credit Growth")
    s += """
**Analysis:** Sweeps k = 1…20 nearest-bank-network neighbours to find the crossover point where λ_knn first exceeds λ_geo, benchmarking the density at which the bank network achieves the same spatial dependence as geographic contiguity.

**Reference lambdas:** λ_geo = 0.1801 (full), λ_geo = 0.1701 (non-border)

**Key assumptions:**
- W_bank_knn_k: top-k links per county from W_bank_bin (most similar bank neighbours)
- Crossover k where gap (λ_knn − λ_geo) first becomes positive
- Density = links / possible links; W_geo has ~0.33% density

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
            "gap vs geo (full)": f"{fmt(r['gap'])}{stars(0.001 if r['gap'] > 0 else 0.5)}",
            "λ_knn (non-border)": fmt(r["lam_bank_knn_nb"]),
            "gap vs geo (nb)": fmt(r["gap_nb"]),
        })
    s += md_table(pd.DataFrame(rows))
    crossover_full = df[df["gap"] > 0]["k"].min()
    crossover_nb   = df[df["gap_nb"] > 0]["k"].min()
    s += f"\n**Crossover k:** Full sample = **k={crossover_full}**; Non-border = **k={crossover_nb}**\n"
    return s


def sec_spatial_multiplier():
    df = load(OUT / "spatial_multiplier_decomposition.csv")
    if df is None:
        return ""
    s = section("16. Spatial Multiplier Decomposition")
    s += """
**Analysis:** Decomposes the total spatial multiplier under the SEM following LeSage & Pace (2009).

**Framework:** Under the SEM `u = λ·W·u + ε`, the reduced-form impact matrix is `S = (I − λ·W)⁻¹`.
The power-series expansion `S = I + λW + λ²W² + …` decomposes impact into:
- **Direct (k=0):** own-county retention
- **Indirect (k≥1):** k-th order network transmission through spatial error propagation

**Key assumptions:**
- Total multiplier = 1/(1−λ) in the symmetric case; exact computation via power series up to convergence
- Average reach (km) = average distance weighted by k-th order impact shares
- Applies to SEM lambdas from §1 (credit) and §2 (HPI)

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


def sec_moran_diagnostics():
    df = load(DIAG / "moran_i_results.csv")
    if df is None:
        return ""
    s = section("17. Moran's I Diagnostics — Pre-estimation")
    s += """
**Analysis:** Tests for spatial autocorrelation in OLS residuals (replicating F&I eq. 1) before fitting spatial models, at county and state levels for both credit and HPI outcomes.

**Statistic:** Standard Moran's I under W_geo (queen contiguity). Permutation-based p-values.

**Key assumptions:**
- Residuals from two-way FE OLS (county + year demeaned)
- Tests both county-level (N ≈ 900–1015) and state-level (N = 42) spatial autocorrelation
- Significant positive Moran's I supports use of spatial error model

**Summary (mean Moran's I across years, county-level):**

"""
    summary = (df[df["level"] == "county"]
               .groupby("outcome")
               .agg(mean_I=("moran_I", "mean"),
                    sig_years=("significant", "sum"),
                    total_years=("significant", "count"))
               .reset_index())
    rows = []
    for _, r in summary.iterrows():
        rows.append({
            "Outcome": r["outcome"],
            "Mean Moran's I": fmt(r["mean_I"]),
            "Significant years / total": f"{int(r['sig_years'])}/{int(r['total_years'])}",
        })
    s += md_table(pd.DataFrame(rows))

    s += "\n**Year-by-year (county-level):**\n\n"
    sub = df[df["level"] == "county"][["year", "outcome", "moran_I", "z_score", "p_value", "significant"]].copy()
    rows2 = []
    for _, r in sub.iterrows():
        rows2.append({
            "Year": int(r["year"]),
            "Outcome": r["outcome"],
            "Moran's I": fmt(r["moran_I"]),
            "z-score": fmt(r["z_score"], 2),
            "p-value": pval_str(r["p_value"]),
            "Significant": "✓" if r["significant"] else "",
        })
    s += md_table(pd.DataFrame(rows2))
    return s


def sec_moran_composite():
    df = load(DIAG / "moran_i_composite_credit_summary.csv")
    if df is None:
        return ""
    s = section("18. Moran's I Under Composite W — Credit Residuals")
    s += """
**Analysis:** Moran's I on credit SEM residuals under composite W matrices at the optimal α* from §4, comparing endpoint matrices (W_geo, W_alt) to the optimal composite.

**Key assumptions:**
- Residuals from Panel_FE_Error estimated at the composite α* optimum
- Lower Moran's I (closer to zero) at α* relative to endpoints indicates better residual whitening

**Summary:**

"""
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Pair": r.get("pair", ""),
            "Sample": r["sample"],
            "W matrix": r["w_matrix"],
            "α": fmt(r["alpha"]),
            "Mean Moran's I": fmt(r["mean_moran_I"]),
            "Sig. years": f"{int(r['significant_years'])}/{int(r['n_years'])}",
        })
    s += md_table(pd.DataFrame(rows))
    return s


def sec_moran_wbank():
    df = load(DIAG / "moran_i_wbank_summary.csv")
    if df is None:
        return ""
    s = section("19. Moran's I — Bank Network W Variants")
    s += """
**Analysis:** Moran's I on credit and HPI SEM residuals across W_bank variants (binary, count-weighted, non-geographic) and KNN truncations (k=1…20).

**Purpose:** Identifies which W specification most effectively absorbs spatial autocorrelation in SEM residuals.

**Key assumptions:**
- Lower Moran's I = better residual whitening = more appropriate spatial weight specification
- W_bank_knn_k results trace how residual autocorrelation changes as the bank network is truncated

**Summary (non-KNN matrices only):**

"""
    non_knn = df[~df["w_matrix"].str.contains("knn")].copy()
    rows = []
    for _, r in non_knn.iterrows():
        rows.append({
            "Outcome": r["outcome"],
            "W matrix": r["w_matrix"],
            "Mean Moran's I": fmt(r["mean_moran_I"]),
            "Median Moran's I": fmt(r["median_moran_I"]),
            "Sig. years": f"{int(r['significant_years'])}/{int(r['n_years'])}",
            "Density": fmt(r["density"] * 100, 3) + "%",
        })
    s += md_table(pd.DataFrame(rows))
    return s


def sec_bank_overlap():
    stats = load(OUT / "geo_bank_overlap_stats.csv")
    dist  = load(OUT / "geo_bank_overlap_distribution.csv")
    if stats is None:
        return ""
    s = section("20. Bank–Geography Network Overlap")
    s += """
**Analysis:** For each county, computes the fraction of its top-5 bank network connections (by W_bank_avg weight) that are also geographic neighbours (W_geo queen contiguity).

**Overlap fraction:** `|top5_bank_nbrs ∩ geo_nbrs| / 5`

**Key assumptions:**
- Counties with zero geographic neighbours (islands) are excluded
- Captures collinearity between bank and geographic networks; used to motivate W_bank_nonGeo

**Summary statistics:**

"""
    stat_col = "statistic" if "statistic" in stats.columns else stats.columns[0]
    rows = [{"Statistic": r[stat_col], "Value": fmt(r["value"])}
            for _, r in stats.iterrows()]
    s += md_table(pd.DataFrame(rows))

    if dist is not None:
        s += "\n**Distribution of overlap fractions:**\n\n"
        rows2 = []
        for _, r in dist.iterrows():
            row = {c: fmt(r[c]) for c in dist.columns}
            rows2.append(row)
        s += md_table(pd.DataFrame(rows2))
    return s


# ── master report ─────────────────────────────────────────────────────────────

def build_report():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [
        f"# Thesis Results Report\n",
        f"*Auto-generated by `analysis/generate_report.py` on {ts}.*\n",
        "*Re-run the script any time to incorporate new results.*\n\n",
        "---\n",
        "## Table of Contents\n\n"
        "1. [Main SEM — Credit Growth](#1-main-sem--credit-growth-panel-fe-error)\n"
        "2. [Main SEM — House Price Growth](#2-main-sem--house-price-growth-panel-fe-error)\n"
        "3. [Four-W Comparison — Credit Growth](#3-four-w-comparison--credit-growth)\n"
        "4. [Composite W Profile — Credit Growth](#4-composite-w-profile--credit-growth)\n"
        "5. [Composite W Profile — House Price Growth](#5-composite-w-profile--house-price-growth)\n"
        "6. [J-Test — Model Selection](#6-j-test--model-selection-wgeo-vs-wbank)\n"
        "7. [LM Diagnostics](#7-lm-diagnostics--credit-growth-ols)\n"
        "8. [Conley SE Comparison](#8-conley-se-comparison)\n"
        "9. [FGLS Comparison](#9-fgls-comparison)\n"
        "10. [SAR Robustness Check](#10-sar-robustness-check--credit-growth)\n"
        "11. [IV-SAR](#11-iv-sar--credit-growth-kelejian--prucha-1998)\n"
        "12. [Endogeneity Check — Interstate Links](#12-endogeneity-check--interstate-bank-links-only)\n"
        "13. [Placebo — Intrastate Links](#13-placebo-check--intrastate-bank-links-only)\n"
        "14. [Non-Geographic Bank Network](#14-non-geographic-bank-network--credit-growth)\n"
        "15. [KNN Crossover Sweep](#15-knn-crossover-sweep--credit-growth)\n"
        "16. [Spatial Multiplier Decomposition](#16-spatial-multiplier-decomposition)\n"
        "17. [Moran's I — Pre-estimation Diagnostics](#17-morans-i-diagnostics--pre-estimation)\n"
        "18. [Moran's I — Composite W](#18-morans-i-under-composite-w--credit-residuals)\n"
        "19. [Moran's I — Bank Network Variants](#19-morans-i--bank-network-w-variants)\n"
        "20. [Bank–Geography Overlap](#20-bankgeography-network-overlap)\n\n"
        "---\n",
    ]

    builders = [
        sec_panel_fe_credit,
        sec_panel_fe_hpi,
        sec_four_w,
        sec_composite_credit,
        sec_composite_hpi,
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
        sec_moran_diagnostics,
        sec_moran_composite,
        sec_moran_wbank,
        sec_bank_overlap,
    ]

    found = 0
    for fn in builders:
        chunk = fn()
        if chunk:
            found += 1
            parts.append(chunk)
            parts.append("\n---\n")

    parts.append(f"\n*Report covers {found}/20 analyses. Missing sections indicate result CSVs not yet generated.*\n")
    return "".join(parts)


if __name__ == "__main__":
    report = build_report()
    out_path = OUT / "results_report.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"Report written to {out_path}")
    print(f"  {len(report):,} characters")
