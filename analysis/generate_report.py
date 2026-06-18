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

New result CSVs are picked up automatically -- sections whose CSV is missing
are skipped and noted at the bottom.
"""

from pathlib import Path
from datetime import datetime
import re
import pandas as pd
import numpy as np

from panel_data import CREDIT_CONTROLS, PLACEBO_CONTROLS

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "output"
DIAG = OUT / "diagnostics"
CREDIT_X_SPEC = "Linter_bra + " + " + ".join(CREDIT_CONTROLS)
PLACEBO_X_SPEC = "Linter_bra + " + " + ".join(PLACEBO_CONTROLS)

# ── helpers ───────────────────────────────────────────────────────────────────

def fmt(x, decimals=4):
    if pd.isna(x):
        return "--"
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
        pf = float(p)
        if pd.isna(pf):
            return ""
        if pf < 0.01:
            return "***"
        if pf < 0.05:
            return "**"
        if pf < 0.10:
            return "*"
        return ""
    except (TypeError, ValueError):
        return ""

def pval_str(p):
    try:
        pf = float(p)
        if pd.isna(pf):
            return "--"
        if pf < 0.001:
            return "<0.001"
        return f"{pf:.3f}"
    except (TypeError, ValueError):
        return str(p)

def fmti(x):
    """Safe integer formatter -- returns '--' for NaN/None."""
    try:
        if pd.isna(x):
            return "--"
        return f"{int(x):,}"
    except (TypeError, ValueError):
        return str(x)

# alias
safe_int = fmti

def load(path):
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df.dropna(how="all").reset_index(drop=True)

def section(title, level=2):
    return f"\n{'#' * level} {title}\n"

def first_section_heading(chunk):
    for line in chunk.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            return stripped[3:].strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return None

def heading_anchor(title):
    anchor = title.lower().replace(" -- ", "--")
    anchor = re.sub(r"[^\w\s-]", "", anchor)
    anchor = re.sub(r"\s+", "-", anchor).strip("-")
    return anchor

def toc_label(title):
    return re.sub(r"^\d+\.\s*", "", title).strip()

def build_toc(toc_entries):
    lines = ["## Table of Contents", ""]
    for i, title in enumerate(toc_entries, start=1):
        lines.append(f"{i}.  [{toc_label(title)}](#{heading_anchor(title)})")
    lines.append("")
    return "\n".join(lines) + "\n"

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
    s = section("1. Main SEM -- Credit Growth")

    s += """
**Source:** `analysis/sem/sem_credit.py`

**References:** Anselin (1988) *Spatial Econometrics*; Anselin et al. (2008) *A Local Indicator of Spatial Association*; Favara & Imbs (2015)

---

**Structural equation (Spatial Error Model):**

```
delta_ln(loans)_it = beta * Linter_bra_it + gamma' X_ct + a_i + tau_t + u_it        ...(1)
u_it = lambda * (W u)_it + eps_it                                                   ...(2)
```

where:
- `delta_ln(loans)_it` = log-change in commercial-bank mortgage loans (Favara-Imbs HMDA), county i, year t
- `Linter_bra_it` = cumulative interstate branching deregulation index
- `X_ct` = `{CREDIT_X_SPEC}`
- `a_i` = county fixed effect; `tau_t` = year fixed effect
- `W` = row-standardised spatial weight matrix (two variants compared)
- `lambda` = spatial error autocorrelation parameter; `eps_it` ~ i.i.d.(0, sigma^2)

Reduced form obtained by substituting (2) into (1):

```
delta_ln(loans)_it = beta * Linter_bra_it + gamma' X_ct + a_i + tau_t + (I - lambdaW)^(-1) eps_it
```

**Estimation:** Maximum likelihood via `spreg.Panel_FE_Error` (Anselin et al. 2008). Log-likelihood under normality:

```
ln L = -(NT/2) ln(2 pi sigma^2) + T ln|I - lambda W| - (1/2 sigma^2) eps'eps
```

where `eps = (I - lambda W) * u_tilde` and `u_tilde` are two-way FE within-demeaned residuals.

**Two-way within (FE) transformation** (Mundlak 1978):

```
z_tilde_it = z_it - z_bar_i - z_bar_t + z_bar      ...(3)
```

where `z_bar_i` = county mean, `z_bar_t` = year mean, `z_bar` = grand mean.

**Gap test** (H1: lambda_bank > lambda_geo, one-sided):

```
z_gap = (lambda_hat_bank - lambda_hat_geo) / sqrt(SE(lambda_hat_bank)^2 + SE(lambda_hat_geo)^2)    ...(4)
p = Phi(-z_gap)   (normal tail)
```

Note: SE(gap) ignores the covariance between estimators (conservative).

**Spatial weight matrices:**
- **W_geo**: queen contiguity (binary adjacency, row-standardised)
- **W_bank**: time-averaged cosine similarity of county bank holding-company portfolios (Favara & Imbs 2015, Data Appendix)

**Sample definitions:**
- **Full**: all counties with complete data
- **Contig**: counties sharing a geographic border with at least one other county (border==1)
- **NonContig**: non-contiguous counties (border==0)

**Results:**

"""
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Model": r.get("model", f"{r.get('w_matrix','')} ({r.get('sample','')})"),
            "beta (deregulation)": f"{fmt(r.get('beta', r.get('beta_D','')))} ({fmt(r.get('se_beta', r.get('se_beta_D','')))}){stars(r.get('p_beta', 0.001))}",
            "lambda (spatial)": f"{fmt(r['lam'])} ({fmt(r['se_lam'])}){stars(r['p_lam'])}",
            "N counties": safe_int(r["n_co"]),
            "N obs": safe_int(r["n_obs"]),
        })
    s += md_table(pd.DataFrame(rows))
    s += "\n*Standard errors in parentheses. \\*p<0.10, \\*\\*p<0.05, \\*\\*\\*p<0.01*\n"
    return s


def sec_four_w():
    df = load(OUT / "four_w_comparison_credit.csv")
    if df is None:
        return ""
    s = section("2. Four-W Comparison -- Credit Growth")

    s += """
**Source:** `analysis/sem/sem_w_variants.py`

**References:** Anselin (1988); Favara & Imbs (2015)

---

**Model:** Same as section 1, equations (1)-(2).

**Gap test** (one-sided z, same as equation (4)):
```
z_gap = (lambda_hat_alt - lambda_hat_geo) / sqrt(SE(lambda_hat_alt)^2 + SE(lambda_hat_geo)^2)
p_gap = Phi(-z_gap)
```

For bank-W rows, `lambda_hat_geo` is re-estimated on the same usable county
list as the bank-W model before computing the gap.

**Likelihood-ratio test** (also reported):
```
LR = 2 * (logLL_alt - logLL_geo) ~ chi^2(1) under H0
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
            if pd.notna(r.get("gap_vs_geo", np.nan)) and not pd.isna(r.get("gap_vs_geo")) else "--"
        rows.append({
            "Sample": r["sample"],
            "W matrix": r["w_matrix"],
            "beta": f"{fmt(r['beta'])} ({fmt(r['se_beta'])}){stars(r['p_beta'])}",
            "lambda": f"{fmt(r['lam'])} ({fmt(r['se_lam'])}){stars(r['p_lam'])}",
            "paired lambda_geo": "--" if r["w_matrix"] == "W_geo"
            else f"{fmt(r.get('lam_geo', np.nan))} ({safe_int(r.get('n_counties_geo', np.nan))})",
            "delta_lambda vs W_geo": gap_str,
            "N obs": safe_int(r["n_obs"]),
        })
    s += md_table(pd.DataFrame(rows))
    s += "\n*Standard errors in parentheses. Paired lambda_geo reports lambda with geo N in parentheses. Gap is one-sided z-test (H1: lambda_alt > lambda_geo).*\n"
    return s


def sec_jtest():
    df = load(OUT / "jtest_credit_results.csv")
    if df is None:
        return ""
    s = section("4. J-Test -- Non-Nested Model Selection")

    s += """
**Source:** `analysis/model_selection/jtest.py`

**References:** Davidson & MacKinnon (1981) *Several Tests for Model Specification in the Presence of Alternative Hypotheses*

---

**Purpose:** Tests whether the spatial prediction component of one W (the alternative) has additional explanatory power when the model is estimated under another W (the null).

**Spatial prediction component** (from fitted SEM residuals):

```
m_hat = lambda_hat * W * u_hat = u_hat - e_filtered      ...(9)
```

where `u_hat` are within-demeaned OLS residuals and `e_filtered = (I - lambda_hat W)u_hat` are the spatially-filtered residuals.

**J-test direction** (H0: W_null, H1: W_alt):

```
Step 1: Estimate SEM under W_alt -> extract m_hat_alt via (9)
Step 2: Augment regressor matrix: X_aug = [X | m_hat_alt]     ...(10)
Step 3: Estimate SEM under W_null with (y, X_aug)
Step 4: J-statistic = z-ratio on the coefficient of m_hat_alt in Step 3
```

Both directions are tested for each W pair.

**Decision rule:**
- Direction 1 rejects only -> W_alt preferred
- Direction 2 rejects only -> W_null preferred
- Both reject -> Indeterminate
- Neither rejects -> Indeterminate

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
    s += "\n*\\*\\*\\* p<0.01. Both directions significant -> neither W matrix dominates.*\n"
    return s


def sec_lm_diag():
    df = load(OUT / "lm_diagnostics_credit.csv")
    if df is None:
        return ""
    s = section("5. LM Diagnostics -- Credit OLS Residuals")

    s += """
**Source:** `analysis/model_selection/lm_tests.py`

**References:** Anselin (1988) section 12; Anselin et al. (1996) *Simple Diagnostic Tests for Spatial Dependence*; Burridge (1980)

---

**Pre-estimation:** Apply the **two-way within transformation** (eq. 3) to all variables so that OLS residuals match the two-way FE mean specification in section 1.

**LM error statistic** (Anselin 1988, eq. 12.7):

```
LM_error = [u_tilde'Wu / sigma_hat^2]^2 / tr(W'W + W^2)      ...(11)
```

under H0: lambda = 0, LM_error ~ chi^2(1).

**LM lag statistic** (Anselin et al. 1996):

```
LM_lag = [u_tilde'Wy_tilde / sigma_hat^2]^2 / [(WXbeta_hat)'M(Wbeta_hat X)/sigma_hat^2 + tr(W'W + W^2)]      ...(12)
```

under H0: rho = 0, LM_lag ~ chi^2(1).

**Robust versions** (Anselin et al. 1996): rLM_error and rLM_lag correct each statistic for the presence of the other spatial misspecification.

**Panel implementation:** Block-diagonal panel weight matrix `W_NT = I_T kron W` passed to `spreg.panel_LMerror` and `spreg.panel_LMlag`.

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
    s = section("6. Conley SE Comparison")

    s += """
**Source:** `analysis/inference/conley_se.py`

**References:** Conley (1999) *GMM Estimation with Cross Sectional Dependence*; Colella, Lalive, Seyed & Tschopp (2019) *Inference with Arbitrary Clustering*

---

**Model** (same point estimate beta across all four estimators):

```
delta_ln(loans)_tilde_it = beta * L_tilde_inter_bra_it + gamma' X_tilde_ct + u_tilde_it
```

Controlled X matrix: `{CREDIT_X_SPEC}`.

**General sandwich variance** (White 1980):

```
Var(beta_hat) = (X'X)^{-1} * B_hat * (X'X)^{-1} * df_correction    ...(13)
```

---

**Estimator 1 -- State-clustered (Favara & Imbs 2015 baseline):**

```
B_state = sum_s score_s^2
score_s = sum_{c in s} sum_t u_tilde_ct * x_tilde_ct              ...(14)
df_corr = G / (G - 1)   (G = number of states)
```

---

**Estimator 2 & 3 -- Spatial HAC** (Conley 1999):

```
B_spatial = sum_t  v_t' W v_t
where v_t[c] = u_tilde_ct * x_tilde_ct   (element-wise product)     ...(15)
```

Applied with W = W_geo (estimator 2) and W = W_bank (estimator 3).

---

**Estimator 4 -- Two-way (State + Spatial)** (Colella et al. 2019):

```
B_twoway = B_state + B_spatial(W_bank) - B_overlap     ...(16)
```

where `B_overlap` = sum over within-state AND spatially-linked county pairs:

```
B_overlap = sum_t sum_{c,c'} w_{cc'} * 1{state(c)==state(c')} * u_ct u_{c't} x_ct x_{c't}     ...(16b)
```

This is the true double-counted region (pairs that appear in both B_state and B_spatial).
The old formula used B_OLS (diagonal sum) as the overlap term, which is incorrect because W has zero diagonal.
The `se_twoway_old` column shows the old formula for comparison.

---

**Results:**

"""
    rows = []
    for _, r in df.dropna(subset=["beta"]).iterrows():
        row_d = {
            "Sample": r["sample"],
            "Estimator": r["estimator"],
            "beta": fmt(r["beta"]),
            "SE": fmt(r["se"]),
            "95% CI": f"[{fmt(r['ci_lower'])}, {fmt(r['ci_upper'])}]",
            "t-stat": f"{fmt(r['t_stat'], 2)}{stars(r['p_value'])}",
        }
        if "se_twoway_old" in r and pd.notna(r.get("se_twoway_old", np.nan)):
            row_d["SE (old formula)"] = fmt(r["se_twoway_old"])
        rows.append(row_d)
    s += md_table(pd.DataFrame(rows))
    return s


def sec_fgls():
    df  = load(OUT / "fgls_comparison.csv")
    ht  = load(OUT / "fgls_hausman_moran.csv")
    if df is None:
        return ""
    s = section("7. FGLS Comparison")

    s += """
**Source:** `analysis/inference/fgls.py`

**References:** Feasible GLS: Greene (2003) *Econometric Analysis*, Ch. 10; Hausman (1978); Conley (1999); Moran (1950)

---

**Model:**

```
delta_ln(loans)_it = beta * Linter_bra_it + gamma' X_ct + a_i + tau_t + eps_it
```

Controlled X matrix: `{CREDIT_X_SPEC}`.

**Spatial filter** (applied period-by-period):

```
A = I - lambda * W                                       ...(18)
y_tilde_A[t] = A * y_tilde[t],   X_tilde_A[t] = A * X_tilde[t]
```

where `y_tilde[t]` and `X_tilde[t]` are the two-way FE within-demeaned arrays for period t (eq. 3).

**FGLS point estimate** (scalar k=1 case):

```
beta_FGLS = (X_tilde_A' X_tilde_A)^{-1} X_tilde_A' y_tilde_A     ...(20)
```

**State-clustered SE (FGLS)** -- unfiltered residuals, filtered regressor as bread:

```
xi_hat_it = y_tilde_it - x_tilde_it * beta_FGLS             [unfiltered residuals]
bread = sum_{c,t} (X_tilde_A,ct)^2
score_s = sum_{c in s, t} X_tilde_A,ct * xi_hat_ct
meat = sum_s score_s^2
Var(beta_hat_FGLS) = meat / bread^2 * G/(G-1)        ...(21)
```

**Five estimators:**
- **OLS:** A = I (no filter), lambda = 0
- **FGLS W_geo:** lambda = lambda_hat_geo from Panel_FE_Error (section 1)
- **FGLS W_bank:** lambda = lambda_hat_bank from Panel_FE_Error (section 1)
- **FGLS W_bank_knn3:** lambda = lambda_hat_knn3 from Panel_FE_Error (falls back to lambda_bank if absent)
- **FGLS W_bank_knn4:** lambda = lambda_hat_knn4 from Panel_FE_Error (section 1)

**Hausman test** (H0: OLS = FGLS W_bank, i.e., no spatial misspecification bias):

```
H = (beta_OLS - beta_FGLS)^2 / (SE_OLS^2 - SE_FGLS^2) ~ chi^2(1)    ...(22)
```

Fallback when denominator <= 0 (non-positive definite variance difference):

```
t_diff = (beta_OLS - beta_FGLS) / sqrt(SE_OLS^2 + SE_FGLS^2)     ...(23)
```

**FGLS beta estimates:**

"""
    rows = []
    for _, r in df.dropna(subset=["beta"]).iterrows():
        rows.append({
            "Sample": r["sample"],
            "Estimator": r["estimator"],
            "beta": f"{fmt(r['beta'])} ({fmt(r['se'])}){stars(r['p_value'])}",
            "95% CI": f"[{fmt(r['ci_lower'])}, {fmt(r['ci_upper'])}]",
            "N counties": safe_int(r["N"]),
        })
    s += md_table(pd.DataFrame(rows))

    if ht is not None:
        s += "\n**Hausman test and residual Moran's I:**\n\n"
        ht_rows = []
        for _, r in ht.dropna(subset=["t_diff"]).iterrows():
            ht_rows.append({
                "Sample": r["sample"],
                "Hausman note": fmt(r["hausman_note"]),
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
    s = section("8. SAR Robustness Check")

    s += """
**Source:** `analysis/sar/sar_credit.py`

**References:** Anselin (1988) Ch. 6; Cliff & Ord (1981) *Spatial Processes*

---

**SAR (Spatial Lag) model:**

```
delta_ln(loans)_it = rho * (W delta_ln(loans))_it + beta * Linter_bra_it + gamma' X_ct + a_i + tau_t + u_it    ...(24)
```

Controlled X matrix: `{CREDIT_X_SPEC}`.

Estimated via `spreg.Panel_FE_Lag`, which instruments `Wy` internally using spatial lags of the exogenous regressors (Kelejian & Prucha 1998).

**Rho gap test:**

```
delta_rho = rho_hat_bank - rho_hat_geo
z = delta_rho / sqrt(SE(rho_hat_bank)^2 + SE(rho_hat_geo)^2)
```

**Results:**

"""
    rows = []
    for _, r in df.iterrows():
        gap_str = f"{fmt(r['delta_rho'])} (z={fmt(r['z_stat'], 2)}){stars(0.001)}" \
            if pd.notna(r.get("delta_rho", np.nan)) and not pd.isna(r.get("delta_rho")) else "--"
        rows.append({
            "Sample": r["sample"],
            "W": r["w_matrix"],
            "rho (spatial lag)": f"{fmt(r['rho'])} ({fmt(r['rho_se'])}){stars(0.001)}",
            "beta (deregulation)": f"{fmt(r['beta_D'])} ({fmt(r['beta_D_se'])}){stars(0.001)}",
            "delta_rho (bank-geo)": gap_str,
            "N obs": safe_int(r["n_obs"]),
        })
    s += md_table(pd.DataFrame(rows))
    s += "\n*SAR rho is analogous to SEM lambda but governs the outcome (not the error) process.*\n"
    return s


def sec_bank_interstate():
    df = load(OUT / "bank_interstate_credit_results.csv")
    if df is None:
        return ""
    s = section("9. Endogeneity Check -- Interstate Bank Links Only")

    s += """
**Source:** `analysis/sem/sem_link_restrictions.py`

**References:** Favara & Imbs (2015, Appendix A); Rice & Strahan (2010)

---

**Model:** Same as section 1, equations (1)-(2), using `W_bank_interstate`.

**Matrix construction:**

```
W_bank_interstate[i,j] = W_bank[i,j]  if state(i) != state(j)
                        = 0            if state(i) = state(j)     ...(28)
```

followed by row-standardisation.

**Motivation:** Within-state bank links may reflect economic sorting (endogeneity). Cross-state links isolate post-IBBEA deregulation and are less subject to within-state omitted-variable bias.

**Results:**

"""
    rows = []
    for _, r in df.dropna(subset=["lam"]).iterrows():
        rows.append({
            "Model": r["model"],
            "beta": f"{fmt(r.get('beta', r.get('beta_D','')))} ({fmt(r.get('se_beta', r.get('se_beta_D','')))}){stars(r.get('p_beta', 0.001))}",
            "lambda": f"{fmt(r['lam'])} ({fmt(r['se_lam'])}){stars(r['p_lam'])}",
            "N obs": safe_int(r["n_obs"]),
        })
    s += md_table(pd.DataFrame(rows))
    return s


def sec_bank_intrastate():
    df = load(OUT / "bank_intrastate_credit_results.csv")
    if df is None:
        return ""
    s = section("10. Placebo Check -- Intrastate Bank Links Only")

    s += """
**Source:** `analysis/sem/sem_link_restrictions.py`

**References:** Favara & Imbs (2015); Rice & Strahan (2010)

---

**Model:** Same as section 1, equations (1)-(2), using `W_bank_intrastate`.

**Matrix construction** (complement of eq. 28):

```
W_bank_intrastate[i,j] = W_bank[i,j]  if state(i) = state(j)
                        = 0            if state(i) != state(j)     ...(29)
```

followed by row-standardisation.

**Placebo logic:** If the bank-network channel captures genuine cross-state deregulation spillovers, then `W_bank_intrastate` should yield lambda_intrastate < lambda_bank. A large lambda_intrastate would suggest the bank channel proxies intra-state economic correlation rather than the IBBEA deregulation shock.

**Results:**

"""
    rows = []
    for _, r in df.dropna(subset=["lam"]).iterrows():
        rows.append({
            "Model": r["model"],
            "beta": f"{fmt(r.get('beta', r.get('beta_D','')))} ({fmt(r.get('se_beta', r.get('se_beta_D','')))}){stars(r.get('p_beta', 0.001))}",
            "lambda": f"{fmt(r['lam'])} ({fmt(r['se_lam'])}){stars(r['p_lam'])}",
            "N obs": safe_int(r["n_obs"]),
        })
    s += md_table(pd.DataFrame(rows))
    return s


def sec_bank_nongeo():
    df = load(OUT / "bank_nongeo_credit_results.csv")
    if df is None:
        return ""
    s = section("11. Non-Geographic Bank Network")

    s += """
**Source:** `analysis/sem/sem_link_restrictions.py`

**References:** Anselin (1988); Favara & Imbs (2015)

---

**Model:** Same as section 1, equations (1)-(2), using `W_bank_nonGeo`.

**Matrix construction:**

```
W_bank_nonGeo[i,j] = W_bank[i,j]  if W_geo[i,j] = 0
                   = 0             if W_geo[i,j] > 0          ...(30)
```

followed by row-standardisation. Removes all bank-network links that coincide with geographic adjacency.

**Purpose:** Tests whether the bank-network spatial channel is genuinely distinct from geographic proximity.

**Results:**

"""
    rows = []
    for _, r in df.dropna(subset=["lam"]).iterrows():
        rows.append({
            "Model": r["model"],
            "beta": f"{fmt(r.get('beta', r.get('beta_D','')))} ({fmt(r.get('se_beta', r.get('se_beta_D','')))}){stars(r.get('p_beta', 0.001))}",
            "lambda": f"{fmt(r['lam'])} ({fmt(r['se_lam'])}){stars(r['p_lam'])}",
            "N obs": safe_int(r["n_obs"]),
        })
    s += md_table(pd.DataFrame(rows))
    return s


def sec_knn():
    df = load(OUT / "knn_sweep_credit_results.csv")
    if df is None:
        return ""

    # Detect column naming convention -- prefer new names, fall back to old
    def _col(new, *fallbacks):
        if new in df.columns:
            return new
        for fb in fallbacks:
            if fb in df.columns:
                return fb
        return new  # will show missing gracefully

    knn_co_col  = _col("lam_bank_knn_contig",  "lam_bank_knn_border", "lam_bank_knn_nb")
    gap_co_col  = _col("gap_contig",            "gap_border",          "gap_nb")

    s = section("12. KNN Crossover Sweep")

    s += """
**Source:** `analysis/sem/sem_knn_sweep.py`

**References:** Anselin (1988); LeSage & Pace (2009)

---

**Model:** Same as section 1, equations (1)-(2), replacing W_bank with `W_bank_knn_k`.

**KNN matrix construction:**

```
W_bank_knn_k[i, :] retains only the top-k entries of W_bank[i, :]
                   (by weight magnitude), sets all others to zero     ...(31)
```

followed by row-standardisation. k sweeps from 1 to 20.

**Purpose:** Finds the crossover k where lambda_knn(k) > lambda_geo, establishing the bank-network density at which spatial error dependence matches geographic contiguity. `lambda_geo` is estimated in the KNN sweep script with the same credit controls and fixed effects.

**Density** = non-zero off-diagonal links / (N * (N-1))

**Results (selected k values):**

"""
    show_ks = [1, 2, 4, 8, 10, 15, 20]
    sub = df[df["k"].isin(show_ks)].copy()
    rows = []
    for _, r in sub.iterrows():
        rows.append({
            "k": safe_int(r["k"]),
            "Density (%)": fmt(r["density"] * 100, 3),
            "lambda_knn (full)": fmt(r["lam_bank_knn"]),
            "gap (full)": fmt(r["gap"]),
            "lambda_knn (contig)": fmt(r.get(knn_co_col, np.nan)),
            "gap (contig)": fmt(r.get(gap_co_col, np.nan)),
        })
    s += md_table(pd.DataFrame(rows))

    cf = df[df["gap"] > 0]["k"].min() if (df["gap"] > 0).any() else ">20"
    cb_vals = df[df.get(gap_co_col, pd.Series(dtype=float)) > 0]["k"] if gap_co_col in df.columns else pd.Series(dtype=float)
    cb = cb_vals.min() if len(cb_vals) > 0 else ">20"
    s += f"\n**Crossover k:** Full sample = **k={cf}**; Contig sample = **k={cb}**\n"
    return s


def sec_spatial_multiplier():
    df = load(OUT / "spatial_multiplier_decomposition.csv")
    if df is None:
        return ""
    s = section("13. SEM Spatial Multiplier Decomposition")

    s += """
**Source:** `analysis/sar/multiplier_decomposition.py`

**References:** LeSage & Pace (2009) *Introduction to Spatial Econometrics*, Ch. 2

---

**Framework:** Under the SEM (eq. 2), the reduced-form error covariance satisfies:

```
u = (I - lambda W)^{-1} eps  =  S * eps
```

The **impact matrix** `S = (I - lambda W)^{-1}` captures total statistical co-movement in the disturbance process between county pairs.

**Power-series expansion** (converges when |lambda| < 1/rho(W)):

```
S = I + lambda W + lambda^2 W^2 + lambda^3 W^3 + ...     ...(32)
```

**Scalar summaries:**
```
Total multiplier  = sum_k lambda^k tr(W^k)/N  = 1/(1-lambda) for symmetric W
Avg indirect effect = (Total multiplier) - (Avg direct)
Indirect share (%) = 100 * Avg indirect / Total multiplier     ...(33)
```

**Average reach (km):** Weighted average distance between counties, with weights proportional to the k-th order impact share. Uses sample-specific county subsets with W subsetted to sample counties before dropping islands.

Series terminates when ||lambda^k W^k||_inf < 10^{-4} (convergence threshold) or k > 600.

**Results:**

"""
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Outcome": r["outcome"],
            "W": r["w_matrix"],
            "lambda": fmt(r["lambda"]),
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
    s = section("14. SAR Multiplier Counterfactual")

    s += """
**Source:** `analysis/extensions/multiplier_counterfactual.py`

**References:** LeSage & Pace (2009), Ch. 2; Kelejian & Prucha (1998)

---

**Framework:** For the SAR model (eq. 24) with row-stochastic W (W*1 = 1), the scalar total multiplier is:

```
M = 1 / (1 - rho)     ...(34)
```

**Effect decomposition** for a unit deregulation shock (delta_D = 1):

```
Direct effect   = beta_D                       (own-county, no network)
Total effect    = beta_D / (1 - rho) = beta_D * M   (own-county + network)
Indirect effect = Total - Direct = beta_D * rho / (1 - rho)    ...(35)
```

**Parametric bootstrap CI** (1000 draws, seed = 0):

```
(rho*, beta*) ~ N([rho_hat, beta_hat], diag([SE_rho^2, SE_beta^2]))    ...(36)
```

Draws with |rho*| >= 1 are discarded (SAR stability condition). CIs = [5th, 95th] percentile of accepted draws.

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
            "rho": fmt(r["rho"]),
            "M = 1/(1-rho)": fmt(r["multiplier"], 3),
            "Direct (beta)": fmt(r["point_direct"]),
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
    s = section("15. SLX Exposure -- Bank-Network Deregulation Spillover")
    s += """
**Source:** `analysis/extensions/slx_exposure.py`

**References:** Halleck Vega & Elhorst (2015) *The SLX Model*; Conley (1999); Colella et al. (2019)

---

**SLX exposure variable:**

```
E_{c,t} = sum_{c'} w^{interstate}_{cc'} * D_{s(c'),t}     ...(37)
```

where `D = Linter_bra` (state-level deregulation index) and `w^{interstate}` is the row-standardised interstate bank-network weight matrix (cross-state links only, eq. 28).

**Regression specifications** (two-way county + year FE, within estimator):

```
(1) Base:    delta_ln(loans)_it = beta_D * D_it + gamma' X_ct + eps_it
(2) SLX:     delta_ln(loans)_it = beta_D * D_it + gamma' X_ct + beta_E * E_it + eps_it     ...(38)
(3) Placebo: delta_ln(loans_pl)_it = beta_D * D_it + gamma' X_pl,ct + beta_E * E_it + eps_it
```

where `loans_pl` = non-bank mortgage lending, `X_ct` = `{CREDIT_X_SPEC}`,
and `X_pl,ct` = `{PLACEBO_X_SPEC}`.

**Permutation test** for beta_E (999 permutations, seed=42):

```
p_perm = #{|beta^pi_E| >= |beta_E_obs|} / 999     ...(39)
```

**Results:**

"""
    rows = []
    for _, r in df.iterrows():
        beta_e_str = (f"{fmt(r['beta_E'])} ({fmt(r['se_E'])}){stars(r.get('p_E', 1.0))}"
                      if pd.notna(r.get("beta_E", np.nan)) else "--")
        perm_str = fmt(r["p_perm"]) if pd.notna(r.get("p_perm", np.nan)) else "--"
        rows.append({
            "Sample": r["sample"],
            "Spec": r["spec"],
            "DV": r["dv"],
            "beta_D (SE)": f"{fmt(r['beta_D'])} ({fmt(r['se_D'])}){stars(r['p_D'])}",
            "beta_E (SE)": beta_e_str,
            "p_perm": perm_str,
            "N obs": safe_int(r["n_obs"]),
        })
    s += md_table(pd.DataFrame(rows))
    s += "\n*Standard errors in parentheses. p_perm: permutation-based p-value for beta_E.*\n"
    return s


def sec_moran_diagnostics():
    df = load(DIAG / "moran_i_results.csv")
    if df is None:
        return ""
    s = section("16. Moran's I -- Pre-Estimation Diagnostics")
    s += """
**Source:** `analysis/diagnostics/moran_baseline.py`

**References:** Moran (1950) *Notes on Continuous Stochastic Phenomena*; Cliff & Ord (1981) *Spatial Processes*

---

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
             "Significant years / total": f"{safe_int(r['sig_years'])}/{safe_int(r['total_years'])}"}
            for _, r in summary.iterrows()]
    s += md_table(pd.DataFrame(rows))

    s += "\n**Year-by-year (county level):**\n\n"
    sub = df[df["level"] == "county"][
        ["year", "outcome", "moran_I", "z_score", "p_value", "significant"]].copy()
    rows2 = [{"Year": safe_int(r["year"]), "Outcome": r["outcome"],
               "Moran's I": fmt(r["moran_I"]), "z": fmt(r["z_score"], 2),
               "p": pval_str(r["p_value"]), "Sig": "Y" if r["significant"] else ""}
             for _, r in sub.iterrows()]
    s += md_table(pd.DataFrame(rows2))
    return s


def sec_moran_wbank():
    df = load(DIAG / "moran_i_wbank_summary.csv")
    if df is None:
        return ""
    s = section("18. Moran's I -- Bank Network W Variants")
    s += """
**Source:** `analysis/diagnostics/moran_bank_variants.py`

**References:** Moran (1950); Anselin (1988)

---

Moran's I (eq. 41) on SEM residuals for all W_bank specifications and KNN truncations k=1...20.
Lower Moran's I = better residual whitening.
Credit outcome only.

**Summary (non-KNN matrices):**

"""
    non_knn = df[~df["w_matrix"].str.contains("knn", na=False)].copy()
    rows = [{"Outcome": r["outcome"], "Level": r.get("level", "county"),
              "W matrix": r["w_matrix"],
              "Mean Moran's I": fmt(r["mean_moran_I"]),
              "Median Moran's I": fmt(r["median_moran_I"]),
              "Sig. years": f"{safe_int(r['significant_years'])}/{safe_int(r['n_years'])}",
              "Density": fmt(r["density"] * 100, 3) + "%"}
             for _, r in non_knn.iterrows()]
    s += md_table(pd.DataFrame(rows))

    # KNN rows for k = 3 and 4.
    KNN_SHOW = [3, 4]
    knn_df = df[df["w_matrix"].str.contains("knn", na=False)].copy()
    # Extract k value
    if not knn_df.empty:
        knn_df = knn_df.copy()
        knn_df["_k"] = knn_df["w_matrix"].str.extract(r"knn(\d+)").astype(float)
        if "level" not in knn_df.columns:
            knn_df["level"] = "county"
        knn_df["_level_order"] = knn_df["level"].map({"county": 0, "state": 1})
        knn_df = (
            knn_df[knn_df["_k"].isin(KNN_SHOW)]
            .sort_values(["_level_order", "_k"])
        )
        if not knn_df.empty:
            s += "\n**KNN matrices (k = 3, 4):**\n\n"
            knn_rows = []
            for _, r in knn_df.iterrows():
                knn_rows.append({
                    "Level": r.get("level", "county"),
                    "W matrix": r["w_matrix"],
                    "k": safe_int(r["_k"]),
                    "Mean Moran's I": fmt(r["mean_moran_I"]),
                    "Median Moran's I": fmt(r["median_moran_I"]),
                    "Sig. years": f"{safe_int(r['significant_years'])}/{safe_int(r['n_years'])}",
                    "Density": fmt(r["density"] * 100, 3) + "%",
                })
            s += md_table(pd.DataFrame(knn_rows))
    return s


def sec_bank_overlap():
    stats = load(OUT / "geo_bank_overlap_stats.csv")
    dist  = load(OUT / "geo_bank_overlap_distribution.csv")
    if stats is None:
        return ""
    s = section("19. Bank-Geography Network Overlap")
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


def sec_w_density():
    df = load(OUT / "w_density_summary.csv")
    if df is None:
        return ""
    s = section("20. W Matrix Density Summary")
    s += """
**Source:** `analysis/diagnostics/w_density_summary.py`

**References:** Anselin (1988); LeSage & Pace (2009)

---

Density statistics for every W matrix used in the analysis pipeline.

- **N**: number of counties (rows)
- **nnz**: non-zero off-diagonal entries
- **density %**: 100 * nnz / (N*(N-1))
- **mean nbrs / median nbrs**: average / median off-diagonal connections per county
- **% isolated**: % of rows with zero off-diagonal weight
- **mean w / median w**: mean and median value of non-zero off-diagonal entries.
  For row-standardised matrices these reflect the typical weight per active
  neighbour.  The two special rows at the bottom use the **UN-row-standardised**
  time-averaged cosine W_bank split by pair type, so their mean w / median w are
  raw cosine similarity scores directly comparable as bank-portfolio overlap.

**Results:**

"""
    STATE_LABELS = {"W_bank same-state entries", "W_bank cross-state entries"}
    std_rows   = df[~df["matrix"].isin(STATE_LABELS)]
    split_rows = df[df["matrix"].isin(STATE_LABELS)]

    def _density_row(r):
        return {
            "Matrix":      r["matrix"],
            "N":           safe_int(r["N"]),
            "nnz":         safe_int(r.get("nnz", np.nan)),
            "density %":   fmt(r["density_pct"], 4),
            "mean nbrs":   fmt(r["mean_nbrs"], 2),
            "median nbrs": fmt(r["median_nbrs"], 2),
            "% isolated":  fmt(r["pct_isolated"], 1),
            "mean w":      fmt(r.get("mean_nonzero_weight",   np.nan)),
            "median w":    fmt(r.get("median_nonzero_weight", np.nan)),
        }

    s += md_table(pd.DataFrame([_density_row(r) for _, r in std_rows.iterrows()]))

    if not split_rows.empty:
        s += """
**Raw cosine W_bank: weight distribution by pair type**
*(density % = nnz / total eligible pairs of that type; mean w / median w are*
*raw un-standardised cosine similarity values)*

"""
        s += md_table(pd.DataFrame([_density_row(r) for _, r in split_rows.iterrows()]))
        s += ("\n*Same-state entries tend to have higher raw cosine similarity "
              "because intra-state banks share overlapping branch footprints. "
              "Cross-state entries isolate the post-IBBEA deregulation channel.*\n")

    return s


def sec_bank_location_branches():
    df = load(OUT / "bank_location_branches.csv")
    sod = load(OUT / "bank_network_descriptives.csv")
    if df is None:
        return ""
    s = section("3. Bank-Type Decomposition -- Importance of Location and Branches")

    s += """
**Source:** `analysis/diagnostics/bank_location_branches.py`

**References:** Favara & Imbs (2015, AER 104(3): 272-323) Tables 3 and A2

---

**Purpose:** Decomposes the credit-growth response to interstate branching
deregulation (`Linter_bra`) by bank type, testing whether the effect
concentrates in out-of-state banks with a *local branch presence*.

**Three panels:**
- **Panel A** (OOS-lb): Out-of-state banks with local branches -- the
  transmission channel identified in Favara & Imbs (2015).
- **Panel B** (OOS-nb): Out-of-state banks without local branches.
  Should show no effect if the channel requires physical presence.
- **Panel C** (IS): In-state banks. Placebo: deregulation allowed
  *in*-state branching which was already mostly complete by 1994.

**Regression** (same as F&I equation 1, two-way FE):

```
delta_ln(outcome)_ct = beta * Linter_bra_ct + gamma' X_ct
                     + L_delta_ln(outcome)_ct
                     + alpha_c + tau_t + u_ct
```

- `X_ct`: `{CREDIT_X_SPEC}` (8 economic controls)
- `alpha_c` = county FE (absorbed by PanelOLS `entity_effects=True`)
- `tau_t` = year FE (year dummies 1995-2005)
- SEs clustered by state

**Cross-panel test:** Wald z-test H0: beta_panel = beta_A (matched by
outcome type), assuming panel-independence (conservative).

---

"""

    samples = df["sample"].unique().tolist() if "sample" in df.columns else ["Full"]
    for samp in samples:
        sub = df[df["sample"] == samp].copy() if "sample" in df.columns else df.copy()
        if sub.empty:
            continue
        s += f"\n**Sample: {samp}**\n\n"

        panels = sub["panel"].unique().tolist() if "panel" in sub.columns else []
        for panel_id in sorted(panels):
            prows = sub[sub["panel"] == panel_id].copy()
            if prows.empty:
                continue
            panel_lbl = prows.iloc[0].get("panel_label", f"Panel {panel_id}")
            s += f"*{panel_id}: {panel_lbl}*\n\n"

            rows = []
            for _, r in prows.iterrows():
                beta_str = (f"{fmt(r['beta'])} ({fmt(r['se'])}){stars(r['pval'])}"
                            if pd.notna(r.get("beta")) else "--")
                wald_str = (pval_str(r["test_vs_A_pval"])
                            if (panel_id != "A"
                                and pd.notna(r.get("test_vs_A_pval")))
                            else "--")
                rows.append({
                    "Outcome": r.get("outcome", r.get("outcome_key", "")),
                    "beta (SE)": beta_str,
                    "p-val": pval_str(r["pval"]) if pd.notna(r.get("pval")) else "--",
                    "Wald vs A": wald_str,
                    "N obs": safe_int(r.get("N", np.nan)),
                    "R2 within": fmt(r.get("R2_within", np.nan)),
                })
            s += md_table(pd.DataFrame(rows))

    if sod is not None:
        s += "\n**Bank-network descriptives (FDIC Summary of Deposits, 1994-2005):**\n\n"
        sod_rows = [{"Statistic": r["stat"], "Value": fmt(r["value"])}
                    for _, r in sod.iterrows()]
        s += md_table(pd.DataFrame(sod_rows))
        s += ("\n*Multi-county BHCs are those with branches in >1 county on average.*\n"
              "*County pairs: connected = non-zero in W\\_bank\\_binary (shared BHC presence).*\n"
              "*Multi-state BHC pairs = pairs where ALL shared BHCs operate in 2+ states.*\n")

    return s


# ── Preserved (not in active builders) ────────────────────────────────────────
# sec_sar_iv        -- deprecated (see output/deprecated/sar_iv_results.csv)
# sec_lambda_time   -- deprecated (year-specific lambda removed from pipeline)


def sec_hub_map():
    """Hub county maps and centrality statistics."""
    hub_csv = OUT / "hub_counties.csv"
    str_csv = OUT / "bank_strength_centrality.csv"
    map_a   = OUT / "hub_map_bank_strength.png"
    map_b   = OUT / "hub_map_geo_transmission.png"

    has_any = hub_csv.exists() or str_csv.exists() or map_a.exists() or map_b.exists()
    if not has_any:
        return ""

    s = section("21. Hub Map -- Bank Network Centrality")
    s += """
**Source:** `analysis/extensions/hub_map.py`

**References:** Favara & Imbs (2015); LeSage & Pace (2009); Bonacich (1987) *Power and Centrality*

---

**Map (a) -- Bank-network in-strength centrality:**

```
strength_i = sum_j w^{raw}_{ji}     (column sum of un-row-standardised W_bank)     ...(45)
```

Captures how many other counties' bank portfolios are most similar to county i in the raw cosine
graph.  High-strength counties are major recipients of bank-network similarity links.

**Map (b) -- SEM total transmission receiver (W_geo, Full sample):**

```
S = (I - lambda * W_geo)^{-1}
receiver_i = sum_{j != i} S_{ij}     (off-diagonal column sum)                     ...(46)
```

Captures the total shock absorbed by county i from the spatial error network.
lambda = lambda_hat_geo (Full sample) from section 1.

"""
    if map_a.exists():
        s += "![Bank-network in-strength centrality](hub_map_bank_strength.png)\n\n"
    else:
        s += "*hub_map_bank_strength.png not yet generated.*\n\n"

    if map_b.exists():
        s += "![SEM total transmission receiver](hub_map_geo_transmission.png)\n\n"
    else:
        s += "*hub_map_geo_transmission.png not yet generated.*\n\n"

    # Top-10 receiver table from hub_counties.csv
    hub = load(hub_csv)
    if hub is not None:
        if "outcome" in hub.columns:
            hub = hub[hub["outcome"] == "credit"].copy()
        if "total_rx" in hub.columns and len(hub) > 0:
            s += "\n**Top-10 receiver counties (total transmission received, W_geo):**\n\n"
            top = hub.nlargest(10, "total_rx")[["fips5", "total_rx"]].copy()
            top.columns = ["FIPS5", "Total received"]
            s += md_table(top)
        if "in_strength" in hub.columns and len(hub) > 0:
            s += "\n**Top-10 in-strength counties (W_bank centrality):**\n\n"
            top = hub.nlargest(10, "in_strength")[["fips5", "in_strength"]].copy()
            top.columns = ["FIPS5", "In-strength"]
            s += md_table(top)

    # Fallback: bank_strength_centrality.csv
    str_df = load(str_csv)
    if hub is None and str_df is not None and "in_strength" in str_df.columns:
        s += "\n**Top-10 in-strength counties (W_bank centrality):**\n\n"
        top = str_df.nlargest(10, "in_strength")[["fips5", "in_strength"]].copy()
        top.columns = ["FIPS5", "In-strength"]
        s += md_table(top)

    return s


def sec_w1994():
    df = load(OUT / "sem_w1994_results.csv")
    if df is None:
        return ""
    s = section("22. Look-Ahead Robustness -- W_bank_1994")

    s += """
**Source:** `analysis/sem/sem_w1994.py`

**References:** Favara & Imbs (2015, Data Appendix); Rice & Strahan (2010); Anselin (1988)

---

**Motivation:** `W_bank_avg` is time-averaged over 1994-2005, the same period as the outcome
and treatment variables.  Post-IBBEA deregulation reshapes bank branching patterns, so
`W_bank_avg` may be partly endogenous to the treatment.  `W_bank_1994` is built from the 1994
FDIC cross-section only -- the year of IBBEA enactment -- predating post-deregulation network
reshaping and exogenous to subsequent shocks.  Stability of lambda under `W_bank_1994` relative
to `W_bank_avg` (section 1) supports the identification.

**Construction:**

```
M[c,h] = 1  if BHC h has any branch in county c  (1994 only)
w_cc'  = (M @ M.T)[c,c'] / sqrt(diag[c] * diag[c'])   (cosine similarity)  ...(47)
```

Zero diagonal, row-standardised after sample subsetting.  Construction identical to
`pipeline/04_build_bank_weights.py` applied to the 1994 slice.

**Lambda gap test** (H1: lambda_1994 > lambda_geo, one-sided):

```
z_gap = (lambda_hat_1994 - lambda_hat_geo) / sqrt(SE_1994^2 + SE_geo^2)    ...(48)
p = Phi(-z_gap)
```

**Network persistence** (W_bank_1994 vs W_bank_avg):

- `corr_rs`: Pearson r between row-standardised W_bank_1994 and W_bank_avg on the union
  of their nonzero off-diagonal entries, after applying the estimation sample subset.
- `corr_raw`: Pearson r between raw (un-standardised) W_bank_1994 cosine entries and
  raw time-averaged W_bank_avg cosine entries on the union of nonzero positions.
- `pct_shared`: % of W_bank_avg nonzero pairs also nonzero in W_bank_1994.

A high corr_rs / large pct_shared indicates the pre-deregulation network already captures
the same county pairs as the time-averaged network -- reducing concern about attenuation bias
from using W_bank_1994.

**Results:**

"""
    # Split into SEM results and persistence stats
    sem_rows = []
    persist_rows = []
    for _, r in df.iterrows():
        sem_rows.append({
            "Sample": r["sample"],
            "W": r["w_matrix"],
            "beta": f"{fmt(r.get('beta',''))} ({fmt(r.get('se_beta',''))}){stars(r.get('p_beta', np.nan))}",
            "lambda": f"{fmt(r.get('lam',''))} ({fmt(r.get('se_lam',''))}){stars(r.get('p_lam', np.nan))}",
            "gap vs W_geo": f"{fmt(r.get('gap_lam', np.nan))} (z={fmt(r.get('z_gap', np.nan), 2)}){stars(r.get('p_gap_onesided', np.nan))}"
                if pd.notna(r.get('gap_lam', np.nan)) else "--",
            "N obs": safe_int(r.get("n_obs", np.nan)),
        })
        if pd.notna(r.get("corr_1994_avg_rs", np.nan)):
            persist_rows.append(r)

    s += md_table(pd.DataFrame(sem_rows))
    s += "\n*Standard errors in parentheses. Gap = lambda_1994 - lambda_geo, one-sided z-test.*\n"

    if persist_rows:
        s += "\n**Network persistence statistics by W_bank_1994 estimation sample:**\n\n"
        prows = []
        for r in persist_rows:
            prows.append({
                "Sample": r.get("sample", ""),
                "BHCs 1994": safe_int(r.get("n_hcs_1994", np.nan)),
                "Pairs W_avg": safe_int(r.get("n_pairs_avg", np.nan)),
                "Pairs W_1994": safe_int(r.get("n_pairs_1994", np.nan)),
                "Shared %": fmt(r.get("pct_pairs_shared", np.nan), 1) + "%",
                "corr rs": fmt(r.get("corr_1994_avg_rs", np.nan)),
                "corr raw": fmt(r.get("corr_1994_avg_raw", np.nan)),
            })
        s += md_table(pd.DataFrame(prows))
        s += ("\n*High corr / large pct_shared indicates the 1994 network spans the same county "
              "pairs as the time-average, supporting use of W_bank_1994 as a pre-deregulation "
              "instrument.*\n")

    return s


# ══════════════════════════════════════════════════════════════════════════════
# MASTER REPORT
# ══════════════════════════════════════════════════════════════════════════════

def build_report():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    parts = [
        "# Spatial Credit Analysis -- Results Report\n\n",
        f"*Auto-generated by `analysis/generate_report.py` on {ts}.*\n",
        "*Re-run `python analysis/generate_report.py` any time to incorporate new results.*\n\n",
        "---\n\n",
        "",
        "---\n",
    ]
    toc_index = 4

    builders = [
        sec_panel_fe_credit,
        sec_four_w,
        sec_bank_location_branches,
        sec_jtest,
        sec_lm_diag,
        sec_conley,
        sec_fgls,
        sec_sar_robustness,
        sec_bank_interstate,
        sec_bank_intrastate,
        sec_bank_nongeo,
        sec_knn,
        sec_spatial_multiplier,
        sec_sar_multiplier,
        sec_slx,
        sec_moran_diagnostics,
        sec_moran_wbank,
        sec_bank_overlap,
        sec_w_density,
        sec_hub_map,
        sec_w1994,
    ]
    N_TOTAL = len(builders)

    found = 0
    toc_entries = []
    for fn in builders:
        try:
            chunk = fn()
        except Exception as e:
            chunk = f"\n> **Error in {fn.__name__}:** {e}\n"
        if chunk:
            found += 1
            heading = first_section_heading(chunk)
            if heading:
                toc_entries.append(heading)
            parts.append(chunk)
            parts.append("\n---\n")

    parts[toc_index] = build_toc(toc_entries)
    parts.append(f"\n*Report covers {found}/{N_TOTAL} analyses.*\n")
    report = "".join(parts)
    return (
        report
        .replace("{CREDIT_X_SPEC}", CREDIT_X_SPEC)
        .replace("{PLACEBO_X_SPEC}", PLACEBO_X_SPEC)
    )


if __name__ == "__main__":
    report = build_report()
    out_path = OUT / "results_report.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"Report written to {out_path}")
    print(f"  {len(report):,} characters, {report.count(chr(10)):,} lines")
