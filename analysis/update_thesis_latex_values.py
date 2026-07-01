"""
Update thesis LaTeX tables from the current CSV outputs.

This restores the full updater behavior from the old builder registry and adds
the later KNN count/binary changes:
  * labels for W_bank_count_knn3/4 and W_bank_binary_knn3/4
  * expanded tab:sem-full
  * tab:knn-type insertion/update
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
DIAG = OUT / "diagnostics"

SAMPLES = ["Full", "Contig", "NonContig"]
SAMPLE_LABELS = {"Full": "Full", "Contig": "Contig", "NonContig": "Non-contig"}

FULL_W_ORDER = [
    "W_geo",
    "W_bank",
    "W_bank_count",
    "W_bank_binary",
    "W_bank_knn3",
    "W_bank_knn4",
    "W_bank_count_knn3",
    "W_bank_count_knn4",
    "W_bank_binary_knn3",
    "W_bank_binary_knn4",
    "W_bank_nonGeo",
    "W_bank_interstate",
    "W_bank_intrastate",
]

KNN_FAMILY_ORDER = [
    "W_bank_knn3",
    "W_bank_knn4",
    "W_bank_count_knn3",
    "W_bank_count_knn4",
    "W_bank_binary_knn3",
    "W_bank_binary_knn4",
]

D_CTL = [
    "Dl_inc", "LDl_inc",
    "Dl_pop", "LDl_pop",
    "Dl_hpi", "LDl_hpi",
    "Dl_her_v", "LDl_her_v",
]

W_LABELS = {
    "W_geo": r"$W_{\text{geo}}$",
    "W_bank": r"$W_{\text{bank}}$",
    "W_bank_count": r"$W_{\text{bank,count}}$",
    "W_bank_binary": r"$W_{\text{bank,bin}}$",
    "W_bank_knn3": r"$W_{\text{bank,knn3}}$",
    "W_bank_knn4": r"$W_{\text{bank,knn4}}$",
    "W_bank_count_knn3": r"$W_{\text{bank,count,knn3}}$",
    "W_bank_count_knn4": r"$W_{\text{bank,count,knn4}}$",
    "W_bank_binary_knn3": r"$W_{\text{bank,bin,knn3}}$",
    "W_bank_binary_knn4": r"$W_{\text{bank,bin,knn4}}$",
    "W_bank_nonGeo": r"$W_{\text{bank,nonGeo}}$",
    "W_bank_interstate": r"$W_{\text{bank,inter}}$",
    "W_bank_intrastate": r"$W_{\text{bank,intra}}$",
    "W_bank_1994": r"$W_{\text{bank,1994}}$",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    raw = path.read_bytes().replace(b"\x00", b"")
    text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    return list(csv.DictReader(text.splitlines()))


def read_latex(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n").strip()


def val(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    x = row.get(key, "")
    if x in ("", "None", "nan", "NaN", None):
        return default
    return float(x)


def fmt(x: float, digits: int = 3) -> str:
    if x is None or math.isnan(x):
        return "--"
    return f"{x:.{digits}f}"


def fmt_int(x: float) -> str:
    if x is None or math.isnan(x):
        return "--"
    return f"{int(round(x)):,}"


def pstars(p: float) -> str:
    if p is None or math.isnan(p):
        return ""
    if p < 0.01:
        return r"^{***}"
    if p < 0.05:
        return r"^{**}"
    if p < 0.10:
        return r"^{*}"
    return ""


def coef_se(beta: float, se: float, p: float, digits: int = 3) -> str:
    return rf"${fmt(beta, digits)}{pstars(p)}\,({fmt(se, digits)})$"


def coef_only(beta: float, p: float, digits: int = 3) -> str:
    return rf"${fmt(beta, digits)}{pstars(p)}$"


def se_only(se: float, digits: int = 3) -> str:
    return rf"$({fmt(se, digits)})$"


def plain_coef(beta: float, p: float, digits: int = 3) -> str:
    return rf"${fmt(beta, digits)}{pstars(p)}$"


def latex_escape(s: str) -> str:
    return (
        s.replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def rows_where(rows: list[dict[str, str]], **criteria: str) -> list[dict[str, str]]:
    return [r for r in rows if all(r.get(k) == v for k, v in criteria.items())]


def row_by(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    hits = rows_where(rows, **criteria)
    if not hits:
        raise KeyError(f"no row matching {criteria}")
    return hits[0]


def table(caption: str, label: str, colspec: str, header: str, body: list[str],
          notes: str = "", size: str = "") -> str:
    size_line = f"  {size}\n" if size else ""
    notes_block = (
        "\n  \\begin{tablenotes}\\footnotesize\n"
        f"  \\item \\emph{{Notes.}} {notes}\n"
        "  \\end{tablenotes}"
        if notes else ""
    )
    return (
        "\\begin{table}[htbp]\n"
        "  \\centering\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{{label}}}\n"
        f"{size_line}"
        f"  \\begin{{tabular}}{{{colspec}}}\n"
        "    \\toprule\n"
        f"{header}\n"
        "    \\midrule\n"
        f"{chr(10).join(body)}\n"
        "    \\bottomrule\n"
        "  \\end{tabular}"
        f"{notes_block}\n"
        "\\end{table}"
    )


def gap_cell(r: dict[str, str], key: str = "gap_vs_geo", zkey: str = "z_gap",
             pkey: str = "p_gap_onesided") -> str:
    gap = val(r, key)
    if math.isnan(gap):
        return "--"
    return rf"${fmt(gap, 3)}\ (z{{=}}{fmt(val(r, zkey), 2)}){pstars(val(r, pkey))}$"


@lru_cache(maxsize=1)
def baseline_placebo_results() -> dict[str, dict[str, dict[str, float]]]:
    import pandas as pd
    import pyreadstat
    import statsmodels.formula.api as smf

    repl = ROOT / "Replication" / "20121416_1data" / "data"
    hmda, _ = pyreadstat.read_dta(repl / "hmda.dta")
    hp, _ = pyreadstat.read_dta(repl / "hp_dereg_controls.dta")
    hp = hp.drop(columns=[c for c in hp.columns if c.startswith("yryear")] + ["state_n"], errors="ignore")

    for df in (hmda, hp):
        df["county"] = pd.to_numeric(df["county"], errors="coerce").astype("Int64")
        df["year"] = df["year"].astype(int)

    df = hmda.merge(hp, on=["county", "year"], how="left", validate="one_to_one")
    df = df.dropna(subset=["county"]).copy()
    df["county_fe"] = df["county"].astype(int).astype(str).str.zfill(5)
    df["state_cluster"] = df["county_fe"].str[:2].astype(int)

    specs = {
        "nloans": ("Dl_nloans_b", "LDl_nloans_b", "Dl_nloans_pl", "LDl_nloans_pl"),
        "vloans": ("Dl_vloans_b", "LDl_vloans_b", "Dl_vloans_pl", "LDl_vloans_pl"),
        "nden": ("Dl_nden_b", "LDl_nden_b", "Dl_nden_pl", "LDl_nden_pl"),
        "lir": ("Dl_lir_b", "LDl_lir_b", "Dl_lir_pl", "LDl_lir_pl"),
        "nsold": ("Dl_nsold_b", "LDl_nsold_b", "Dl_nsold_pl", "LDl_nsold_pl"),
    }

    def fit_one(dv: str, lag: str) -> dict[str, float]:
        required = [dv, "Linter_bra", lag, *D_CTL, "county_fe", "year", "state_cluster"]
        sub = df.dropna(subset=required).copy()
        formula = f"{dv} ~ Linter_bra + {lag} + " + " + ".join(D_CTL) + " + C(county_fe) + C(year)"
        res = smf.ols(formula, data=sub).fit(
            cov_type="cluster",
            cov_kwds={"groups": sub["state_cluster"]},
        )
        return {
            "beta": float(res.params["Linter_bra"]),
            "se": float(res.bse["Linter_bra"]),
            "pval": float(res.pvalues["Linter_bra"]),
            "N": float(len(sub)),
            "n_counties": float(sub["county_fe"].nunique()),
        }

    out: dict[str, dict[str, dict[str, float]]] = {}
    for key, (dv_b, lag_b, dv_p, lag_p) in specs.items():
        a = fit_one(dv_b, lag_b)
        b = fit_one(dv_p, lag_p)
        se_diff = math.sqrt(a["se"] ** 2 + b["se"] ** 2)
        z = (a["beta"] - b["beta"]) / se_diff if se_diff > 0 else float("nan")
        p_equal = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
        out[key] = {"A": a, "B": b, "p_equal": {"pval": p_equal}}
    return out


@lru_cache(maxsize=1)
def build_placebo_tables() -> tuple[str, str]:
    rows = baseline_placebo_results()
    a = rows["nloans"]["A"]
    b = rows["nloans"]["B"]
    main_body = [
        rf"    Deregulation $D_{{s,t-1}}$ & {coef_only(a['beta'], a['pval'])} & {coef_only(b['beta'], b['pval'])} \\",
        rf"                             & {se_only(a['se'])} & {se_only(b['se'])} \\",
    ]
    main = table(
        "Commercial-bank credit growth and the placebo lenders.",
        "tab:placebo",
        "lcc",
        r"     & Panel A: Commercial banks & Panel B: Placebo lenders \\",
        main_body,
        "Dependent variable is log mortgage-originations growth. Standard errors "
        f"clustered by state in parentheses. Panel A: {fmt_int(a['N'])} obs, "
        f"{fmt_int(a['n_counties'])} counties; Panel B: {fmt_int(b['N'])} obs, "
        f"{fmt_int(b['n_counties'])} counties.",
    )

    outcomes = ["nloans", "vloans", "nden", "lir", "nsold"]
    labels = {
        "nloans": r"\# mortgages",
        "vloans": "Mortgage volume",
        "nden": "Denials",
        "lir": "Loan-to-income",
        "nsold": "Loans sold",
    }
    a_cells = []
    a_se = []
    b_cells = []
    b_se = []
    tests = []
    for key in outcomes:
        aa = rows[key]["A"]
        bb = rows[key]["B"]
        a_cells.append(coef_only(aa["beta"], aa["pval"]))
        a_se.append(se_only(aa["se"]))
        b_cells.append(coef_only(bb["beta"], bb["pval"]))
        b_se.append(se_only(bb["se"]))
        tests.append(rf"$[{fmt(rows[key]['p_equal']['pval'], 3)}]$")
    body = [
        r"    \multicolumn{6}{l}{\emph{Panel A. Commercial banks}}\\",
        rf"    Deregulation & {' & '.join(a_cells)} \\",
        rf"                 & {' & '.join(a_se)} \\",
        r"    \addlinespace",
        r"    \multicolumn{6}{l}{\emph{Panel B. Independent mortgage cos., thrifts, credit unions}}\\",
        rf"    Deregulation & {' & '.join(b_cells)} \\",
        rf"                 & {' & '.join(b_se)} \\",
        rf"    Test (Panel A $=$ Panel B) & {' & '.join(tests)} \\",
    ]
    full = table(
        "Deregulation and credit growth: commercial banks versus placebo lenders (full sample, five outcomes).",
        "tab:placebo-full",
        "lccccc",
        r"     & " + " & ".join(labels[k] for k in outcomes) + r" \\",
        body,
        "Dependent variables are log changes in mortgage outcomes. Two-way fixed "
        "effects; standard errors clustered by state. Test row reports p-values "
        "for equality of Panel A and Panel B coefficients.",
        size=r"\small",
    )
    return main, full


def build_banktype_table(sample: str, label: str, caption_suffix: str,
                         main: bool = False) -> str:
    rows = read_csv(OUT / "bank_location_branches.csv")
    outcomes = [
        ("nloans", r"\# mortgages originated"),
        ("vloans", "Volume of mortgages"),
        ("nden", "Denial rate"),
        ("lir", "Loan-to-income ratio"),
        ("nsold", r"\# loans sold"),
        ("nbra", r"\# local branches"),
    ]
    if main:
        outcomes = [outcomes[0], outcomes[-1]]
    body = []
    for row_idx, (outcome, out_label) in enumerate(outcomes):
        if row_idx:
            body.append(r"    \addlinespace")
        coef_cells = []
        se_cells = []
        counts = []
        for panel in ["A", "B", "C"]:
            hits = rows_where(rows, sample=sample, panel=panel, outcome_key=outcome)
            if not hits:
                coef_cells.append("---")
                se_cells.append("---")
                counts.append("---")
            else:
                r = hits[0]
                ptest = val(r, "test_vs_A_pval")
                suffix = "" if math.isnan(ptest) else rf"\,[{fmt(ptest, 3)}]"
                coef_cells.append(coef_only(val(r, "beta"), val(r, "pval")))
                se_cells.append(rf"{se_only(val(r,'se'))}{suffix}")
                counts.append(
                    rf"{{\scriptsize ${fmt_int(val(r,'N'))}$/${fmt_int(val(r,'n_counties'))}$}}"
                )
        body.append(rf"    {out_label} & {coef_cells[0]} & {coef_cells[1]} & {coef_cells[2]} \\")
        body.append(rf"                            & {se_cells[0]} & {se_cells[1]} & {se_cells[2]} \\")
        if not main:
            body.append(rf"                            & {counts[0]} & {counts[1]} & {counts[2]} \\")
    return table(
        f"The importance of bank location and of bank branches {caption_suffix}.",
        label,
        "lccc",
        r"     & OOS, local branch & OOS, no branch & In-state \\",
        body,
        "Bracketed values are p-values for equality with the out-of-state local-branch "
        "coefficient where available.",
        size=r"\small",
    )


def build_moran_summary() -> str:
    rows = read_csv(DIAG / "moran_i_wbank_summary.csv")
    matrices = ["W_geo", "W_bank_knn3", "W_bank_intrastate", "W_bank_interstate", "W_bank"]
    body = []
    for w in matrices:
        r = row_by(rows, outcome="credit", level="county", w_matrix=w)
        body.append(
            rf"    {W_LABELS[w]} & ${fmt(val(r,'mean_moran_I'), 2)}$ & "
            rf"${fmt(100 * val(r,'density'), 1)}$ & ${fmt(val(r,'avg_nbrs'), 1)}$ \\"
        )
    return table(
        "Residual spatial dependence and matrix density (full sample).",
        "tab:moran-summary",
        "lccc",
        r"    Weights matrix & Mean Moran's $I$ & Density (\%) & Mean neighbours \\",
        body,
        "Mean over 1995--2005 of year-by-year Moran's $I$ on within-OLS residuals.",
        size=r"\small",
    )


def build_moran_yearly() -> str:
    rows = read_csv(DIAG / "moran_i_wbank_results.csv")
    summary = read_csv(DIAG / "moran_i_wbank_summary.csv")
    matrices = FULL_W_ORDER
    body = []
    years = sorted({r["year"] for r in rows if r.get("outcome") == "credit" and r.get("level") == "county"})
    for yr in years:
        cells = []
        for w in matrices:
            r = row_by(rows, outcome="credit", level="county", year=yr, w_matrix=w)
            cells.append(plain_coef(val(r, "moran_I"), val(r, "p_value"), 3))
        body.append(rf"    {yr} & {' & '.join(cells)} \\")
    body.append(r"    \midrule")
    mean_cells = []
    sig_cells = []
    density_cells = []
    for w in matrices:
        yr_rows = rows_where(rows, outcome="credit", level="county", w_matrix=w)
        mean_cells.append(rf"${fmt(sum(val(r, 'moran_I') for r in yr_rows) / len(yr_rows), 3)}$")
        sig_cells.append(rf"${sum(val(r, 'p_value') < 0.05 for r in yr_rows)}/{len(yr_rows)}$")
        sr = row_by(summary, outcome="credit", level="county", w_matrix=w)
        density_cells.append(rf"${fmt(100 * val(sr, 'density'), 2)}$")
    body.append(rf"    Mean & {' & '.join(mean_cells)} \\")
    body.append(rf"    Sig.\ years & {' & '.join(sig_cells)} \\")
    body.append(rf"    Density (\%) & {' & '.join(density_cells)} \\")
    return table(
        "Year-by-year residual Moran's $I$ by weights matrix.",
        "tab:moran-yearly",
        "l" + "c" * len(matrices),
        "    Year & " + " & ".join(W_LABELS[w] for w in matrices) + r" \\",
        body,
        "Permutation p-values use 999 permutations.",
        size=r"\scriptsize",
    )


def build_sem_main() -> str:
    rows = read_csv(OUT / "four_w_comparison_credit.csv")
    body = []
    for sample in SAMPLES:
        for w in ["W_geo", "W_bank_knn3", "W_bank"]:
            r = row_by(rows, sample=sample, w_matrix=w)
            body.append(
                rf"    {SAMPLE_LABELS[sample]} & {W_LABELS[w]} & "
                rf"{coef_se(val(r,'beta'), val(r,'se_beta'), val(r,'p_beta'))} & "
                rf"{coef_se(val(r,'lam'), val(r,'se_lam'), val(r,'p_lam'))} & {gap_cell(r)} \\"
            )
        if sample != SAMPLES[-1]:
            body.append(r"    \addlinespace")
    return table(
        "Spatial error model: deregulation effect and spatial parameter, all samples.",
        "tab:sem-main",
        "llccc",
        r"    Sample & Weights matrix & $\hat\beta_1$ & $\hat\lambda$ & Gap vs.\ $W_{\text{geo}}$ \\",
        body,
        "Gap statistics compare each bank matrix with a same-sample geographic baseline.",
    )


def build_sem_full() -> str:
    rows = read_csv(OUT / "four_w_comparison_credit.csv")
    matrices = [
        "W_geo", "W_bank", "W_bank_count", "W_bank_binary",
        "W_bank_knn3", "W_bank_knn4",
        "W_bank_count_knn3", "W_bank_count_knn4",
        "W_bank_binary_knn3", "W_bank_binary_knn4",
        "W_bank_nonGeo", "W_bank_interstate", "W_bank_intrastate",
    ]
    body = []
    for sample in SAMPLES:
        for idx, w in enumerate(matrices):
            r = row_by(rows, sample=sample, w_matrix=w)
            s = SAMPLE_LABELS[sample] if idx == 0 else ""
            body.append(
                rf"    {s} & {W_LABELS[w]} & "
                rf"{coef_se(val(r,'beta'), val(r,'se_beta'), val(r,'p_beta'))} & "
                rf"{coef_se(val(r,'lam'), val(r,'se_lam'), val(r,'p_lam'))} & {gap_cell(r)} \\"
            )
        if sample != SAMPLES[-1]:
            body.append(r"    \addlinespace")
    return table(
        "Full SEM robustness table across geographic and bank-network weight matrices.",
        "tab:sem-full",
        "llccc",
        r"    Sample & Weight matrix & $\hat\beta_1$ & $\hat\lambda$ & Gap vs.\ $W_{\text{geo}}$ \\",
        body,
        "Entries report coefficient estimates with standard errors in parentheses. "
        "The four count/binary KNN rows hold density fixed while changing the link ranking "
        "or weighting scheme.",
        size=r"\footnotesize",
    )


def build_knn_type_comparison() -> str:
    rows = read_csv(OUT / "four_w_comparison_credit.csv")
    order = [
        ("W_bank_knn3", "Continuous", "3"),
        ("W_bank_count_knn3", "Count", "3"),
        ("W_bank_binary_knn3", "Binary", "3"),
        ("W_bank_knn4", "Continuous", "4"),
        ("W_bank_count_knn4", "Count", "4"),
        ("W_bank_binary_knn4", "Binary", "4"),
    ]
    body = []
    for w, typ, k in order:
        if k == "4" and body and not any(r"\addlinespace" in line for line in body):
            body.append(r"    \addlinespace")
        r = row_by(rows, sample="Full", w_matrix=w)
        body.append(
            rf"    {typ} & $k={k}$ & {coef_se(val(r,'lam'), val(r,'se_lam'), val(r,'p_lam'))} & {gap_cell(r)} \\"
        )
    return table(
        "Spatial error parameter by connection type and density: KNN-3 vs KNN-4, "
        "continuous vs count vs binary weights (full sample).",
        "tab:knn-type",
        "llcc",
        r"    Weight type & Density & $\hat\lambda$ & Gap vs.\ $W_{\text{geo}}$ \\",
        body,
        "Continuous KNN matrices rank links by branch-weighted strength. Count and "
        "binary KNN matrices rank by shared-bank count; count weights retain counts, "
        "while binary weights are unweighted 0/1 links. All matrices are row-standardised.",
    )


def build_linkrestrict() -> str:
    rows = read_csv(OUT / "four_w_comparison_credit.csv")
    matrices = ["W_bank_knn3", "W_bank_knn4", "W_bank_nonGeo", "W_bank_interstate", "W_bank_intrastate", "W_geo"]
    body = []
    for w in matrices:
        cells = []
        for s in SAMPLES:
            r = row_by(rows, sample=s, w_matrix=w)
            cells.append(plain_coef(val(r, "lam"), val(r, "p_lam"), 3))
        body.append(rf"    {W_LABELS[w]} & {' & '.join(cells)} \\")
    return table(
        "Cross-state versus within-state bank links and the non-geographic network.",
        "tab:linkrestrict",
        "lccc",
        r"    Matrix & Full & Contig & Non-contig \\",
        body,
        "Entries are SEM spatial-error parameters.",
        size=r"\small",
    )


def build_lm_table() -> str:
    rows = read_csv(OUT / "lm_diagnostics_credit.csv")
    matrices = FULL_W_ORDER
    body = []
    for s in SAMPLES:
        for w in matrices:
            r = row_by(rows, sample=s, w_matrix=w)
            body.append(
                rf"    {SAMPLE_LABELS[s]} & {W_LABELS[w]} & "
                rf"{plain_coef(val(r,'LM_error'), val(r,'p_LM_error'), 1)} & "
                rf"{plain_coef(val(r,'LM_lag'), val(r,'p_LM_lag'), 1)} & "
                rf"{plain_coef(val(r,'rLM_error'), val(r,'p_rLM_error'), 1)} & "
                rf"{plain_coef(val(r,'rLM_lag'), val(r,'p_rLM_lag'), 1)} & "
                rf"{latex_escape(r['decision'])} \\"
            )
        if s != SAMPLES[-1]:
            body.append(r"    \addlinespace")
    return table(
        "LM diagnostics for spatial error and spatial lag dependence.",
        "tab:lm",
        "llccccc",
        r"    Sample & $W$ & LM err. & LM lag & Robust err. & Robust lag & Decision \\",
        body,
        "LM test statistics with p-value stars.",
        size=r"\footnotesize",
    )


def build_robust_snapshot() -> str:
    sar = read_csv(OUT / "sar_robustness_credit.csv")
    fgls = read_csv(OUT / "fgls_comparison.csv")
    body = []
    for w in ["W_geo", "W_bank_knn3", "W_bank"]:
        r = row_by(sar, sample="Full", w_matrix=w)
        body.append(rf"    SAR $\hat\rho$, {W_LABELS[w]} & {coef_se(val(r,'rho'), val(r,'rho_se'), 0.0)} \\")
        body.append(rf"    SAR $\hat\beta_1$, {W_LABELS[w]} & {coef_se(val(r,'beta_D'), val(r,'beta_D_se'), 0.0)} \\")
    body.append(r"    \addlinespace")
    for est in ["OLS (Favara-Imbs)", "FGLS W_geo", "FGLS W_bank_knn3", "FGLS W_bank"]:
        r = row_by(fgls, sample="Full", estimator=est)
        body.append(rf"    {latex_escape(est)} & {coef_se(val(r,'beta'), val(r,'se'), val(r,'p_value'))} \\")
    return table(
        "Robustness of the ranking across estimators (full sample).",
        "tab:robust-snap",
        "lc",
        r"    Estimator & Estimate \\",
        body,
        "SAR entries use ML spatial-lag estimates; FGLS entries use state-clustered standard errors.",
        size=r"\small",
    )


def build_reach_table() -> str:
    rows = read_csv(OUT / "spatial_multiplier_decomposition.csv")
    body = []
    for w in ["W_geo", "W_bank_knn3", "W_bank_knn4", "W_bank"]:
        r = row_by(rows, outcome="credit", sample="full", w_matrix=w)
        body.append(
            rf"    {W_LABELS[w]} & ${fmt(val(r,'lambda'), 3)}$ & ${fmt(val(r,'total_multiplier'), 2)}$ & "
            rf"${fmt(val(r,'indirect_share_pct'), 1)}$ & ${fmt(val(r,'avg_reach_km'), 0)}$ \\"
        )
    return table(
        "Spatial error multiplier: indirect share and geographic reach, full sample.",
        "tab:reach",
        "lcccc",
        r"    Weights matrix & $\hat\lambda$ & Total mult. & Indirect share (\%) & Avg. reach (km) \\",
        body,
        "Reach is the transmission-weighted average great-circle distance.",
    )


def build_slx_tables() -> tuple[str, str]:
    rows = read_csv(OUT / "slx_exposure_results.csv")
    body = []
    for s in SAMPLES:
        r = row_by(rows, sample=s, spec="SLX-OLS", dv="Dl_nloans_b")
        p = row_by(rows, sample=s, spec="Placebo-OLS", dv="Dl_nloans_pl")
        body.append(
            rf"    {SAMPLE_LABELS[s]} & {plain_coef(val(r,'beta_E'), val(r,'p_E'))} & "
            rf"${fmt(val(r,'p_perm'), 3)}$ & {plain_coef(val(p,'beta_E'), val(p,'p_E'))} \\"
        )
    main = table(
        "Bank-network exposure to out-of-state deregulation.",
        "tab:slx",
        "lccc",
        r"    Sample & $\hat\beta_E$, bank credit & Perm. $p$ & $\hat\beta_E$, placebo \\",
        body,
        "Exposure uses the row-standardised interstate bank-network matrix.",
    )

    body2 = []
    for s in SAMPLES:
        for spec in ["Base", "SLX-OLS", "SLX-Conley", "SLX-Cluster", "Placebo-OLS"]:
            r = row_by(rows, sample=s, spec=spec)
            body2.append(
                rf"    {SAMPLE_LABELS[s]} & {latex_escape(spec)} & "
                rf"{plain_coef(val(r,'beta_D'), val(r,'p_D'))} & "
                rf"{plain_coef(val(r,'beta_E'), val(r,'p_E'))} & ${fmt(val(r,'p_perm'), 3)}$ \\"
            )
        if s != SAMPLES[-1]:
            body2.append(r"    \addlinespace")
    full = table(
        "Full SLX exposure specifications.",
        "tab:slx-full",
        "llccc",
        r"    Sample & Spec. & $\hat\beta_D$ & $\hat\beta_E$ & Perm. $p$ \\",
        body2,
        "Base rows omit the exposure term by construction.",
        size=r"\footnotesize",
    )
    return main, full


def build_knn_table() -> str:
    rows = read_csv(OUT / "knn_sweep_credit_results.csv")
    body = []
    for r in rows:
        specs = [
            ("Continuous", "lam_bank_knn", "gap", "lam_bank_knn_contig", "gap_contig", "lam_bank_knn_noncontig", "gap_noncontig"),
            ("Count", "lam_bank_count_knn", "gap_count", "lam_bank_count_knn_contig", "gap_count_contig", "lam_bank_count_knn_noncontig", "gap_count_noncontig"),
            ("Binary", "lam_bank_binary_knn", "gap_binary", "lam_bank_binary_knn_contig", "gap_binary_contig", "lam_bank_binary_knn_noncontig", "gap_binary_noncontig"),
        ]
        for typ, lf, gf, lc, gc, ln, gn in specs:
            body.append(
                rf"    {int(val(r,'k'))} & {typ} & ${fmt(val(r,'avg_nbrs'), 1)}$ & "
                rf"${fmt(val(r,lf), 3)}$ & ${fmt(val(r,gf), 3)}$ & "
                rf"${fmt(val(r,lc), 3)}$ & ${fmt(val(r,gc), 3)}$ & "
                rf"${fmt(val(r,ln), 3)}$ & ${fmt(val(r,gn), 3)}$ \\"
            )
        body.append(r"    \addlinespace")
    if body and body[-1] == r"    \addlinespace":
        body.pop()
    return table(
        "KNN density sweep for continuous, count, and binary bank-network SEM specifications.",
        "tab:knn",
        "rlrrrrrrr",
        r"    $k$ & Type & Avg. nbrs & $\hat\lambda_F$ & Gap$_F$ & $\hat\lambda_C$ & Gap$_C$ & $\hat\lambda_N$ & Gap$_N$ \\",
        body,
        r"Gap is $\hat\lambda_{\text{knn}}-\hat\lambda_{\text{geo}}$.",
        size=r"\scriptsize",
    )


def build_sar_full() -> str:
    rows = read_csv(OUT / "sar_robustness_credit.csv")
    matrices = FULL_W_ORDER
    body = []
    for s in SAMPLES:
        for w in matrices:
            r = row_by(rows, sample=s, w_matrix=w)
            gap = val(r, "delta_rho")
            gap_txt = "--" if math.isnan(gap) else rf"${fmt(gap, 3)}\ (z{{=}}{fmt(val(r,'z_stat'), 2)})$"
            body.append(
                rf"    {SAMPLE_LABELS[s]} & {W_LABELS[w]} & "
                rf"{coef_se(val(r,'rho'), val(r,'rho_se'), 0.0)} & "
                rf"{coef_se(val(r,'beta_D'), val(r,'beta_D_se'), 0.0)} & {gap_txt} \\"
            )
        if s != SAMPLES[-1]:
            body.append(r"    \addlinespace")
    return table(
        "SAR robustness across weight matrices.",
        "tab:sar-full",
        "llccc",
        r"    Sample & Weight matrix & $\hat\rho$ & $\hat\beta_1$ & Gap vs.\ $W_{\text{geo}}$ \\",
        body,
        "SAR is estimated by maximum likelihood with two-way fixed effects.",
        size=r"\footnotesize",
    )


def build_hubs_table() -> str:
    rows = read_csv(OUT / "hub_counties.csv")
    body = []
    for outcome in ["credit"]:
        for w in [x for x in FULL_W_ORDER if x != "W_geo"]:
            for direction in ["sender", "receiver"]:
                hits = rows_where(rows, outcome=outcome, w_matrix=w, direction=direction)[:3]
                counties = ", ".join(f"{h['fips5']} ({h['state']})" for h in hits)
                body.append(rf"    {W_LABELS[w]} & {latex_escape(direction.title())} & {latex_escape(counties)} \\")
    return table(
        "Top transmission hubs in the bank-network multiplier.",
        "tab:hubs",
        "lll",
        r"    Matrix & Direction & Top counties \\",
        body,
        "Top three counties by fitted transmission score.",
        size=r"\footnotesize",
    )


def build_fgls_table() -> str:
    rows = read_csv(OUT / "fgls_comparison.csv")
    body = []
    for s in SAMPLES:
        for r in [x for x in rows if x["sample"] == s]:
            body.append(
                rf"    {SAMPLE_LABELS[s]} & {latex_escape(r['estimator'])} & "
                rf"${fmt(val(r,'beta'), 3)}$ & ${fmt(val(r,'se'), 3)}$ & ${fmt(val(r,'p_value'), 3)}$ \\"
            )
        if s != SAMPLES[-1]:
            body.append(r"    \addlinespace")
    return table(
        "Deregulation coefficient after spatial filtering (FGLS) versus within-OLS.",
        "tab:fgls",
        "llccc",
        r"    Sample & Estimator & $\hat\beta_1$ & SE & $p$ \\",
        body,
        "State-clustered standard errors.",
        size=r"\small",
    )


def build_jtest_table() -> str:
    rows = read_csv(OUT / "jtest_credit_results.csv")
    body = []
    for s in SAMPLES:
        for r in [x for x in rows if x["sample"] == s]:
            w = r["w_alt"]
            body.append(
                rf"    {SAMPLE_LABELS[s]} & {W_LABELS.get(w, latex_escape(w))} & "
                rf"{plain_coef(val(r,'d1_j_z'), val(r,'d1_j_p'), 2)} & "
                rf"{plain_coef(val(r,'d2_j_z'), val(r,'d2_j_p'), 2)} & "
                rf"{latex_escape(r['conclusion'])} \\"
            )
        if s != SAMPLES[-1]:
            body.append(r"    \addlinespace")
    return table(
        r"Davidson--MacKinnon $J$-tests, $W_{\text{geo}}$ versus bank-network matrices.",
        "tab:jtest",
        "llccl",
        r"    Sample & Rival $W$ & Dir. 1 $z$ & Dir. 2 $z$ & Conclusion \\",
        body,
        "Direction 1 adds the rival matrix prediction to the geographic model; "
        "Direction 2 reverses the roles.",
        size=r"\footnotesize",
    )


def build_conley_table() -> str:
    rows = [r for r in read_csv(OUT / "conley_se_comparison.csv") if r["sample"] == "Full"]
    body = []
    for r in rows:
        label = r["estimator"]
        if r.get("kernel"):
            label += f" ({r['kernel']})"
        body.append(
            rf"    {latex_escape(label)} & ${fmt(val(r,'se'), 4)}$ & "
            rf"{plain_coef(val(r,'t_stat'), val(r,'p_value'), 2)} \\"
        )
    return table(
        r"Standard error of $\hat\beta_1$ under alternative covariance estimators (full sample).",
        "tab:conley",
        "lcc",
        r"    Estimator & SE & $t$ \\",
        body,
        "Point estimate fixed at the baseline deregulation coefficient.",
        size=r"\small",
    )


def build_w1994_table() -> str:
    rows = read_csv(OUT / "sem_w1994_results.csv")
    body = []
    for s in SAMPLES:
        for w in ["W_geo", "W_bank_1994", "W_bank_knn3", "W_bank_knn4"]:
            r = row_by(rows, sample=s, w_matrix=w)
            gap = val(r, "gap_lam")
            gap_txt = "--" if math.isnan(gap) else rf"${fmt(gap, 3)}$"
            body.append(
                rf"    {SAMPLE_LABELS[s]} & {W_LABELS[w]} & "
                rf"{coef_se(val(r,'lam'), val(r,'se_lam'), val(r,'p_lam'))} & {gap_txt} & "
                rf"${fmt(val(r,'corr_1994_avg_raw'), 2)}$ \\"
            )
        if s != SAMPLES[-1]:
            body.append(r"    \addlinespace")
    return table(
        "Look-ahead robustness using the 1994 bank network.",
        "tab:w1994",
        "llccc",
        r"    Sample & Matrix & $\hat\lambda$ & Gap & Raw corr. \\",
        body,
        "Correlation statistics apply to the 1994 bank matrix rows.",
        size=r"\footnotesize",
    )


def build_ivsar_table() -> str:
    rows = read_csv(OUT / "sar_iv_results.csv")
    body = []
    for s in ["Full", "Contig"]:
        for r in [x for x in rows if x["sample"] == s]:
            body.append(
                rf"    {SAMPLE_LABELS[s]} & {W_LABELS.get(r['W'], r['W'])} & {latex_escape(r['spec'])} & "
                rf"${fmt(val(r,'rho'), 3)}$ & ${fmt(val(r,'beta'), 3)}$ & "
                rf"${fmt(val(r,'first_stage_F'), 1)}$ & ${fmt(val(r,'first_stage_F_cluster'), 2)}$ \\"
            )
        if s != "Contig":
            body.append(r"    \addlinespace")
    return table(
        "IV-SAR and SDM-IV estimates.",
        "tab:ivsar",
        "lllcccc",
        r"    Sample & $W$ & Spec. & $\hat\rho$ & $\hat\beta_1$ & First-stage $F$ & Cluster $F$ \\",
        body,
        "Cluster $F$ is the squared state-clustered t-statistic for the first excluded instrument.",
        size=r"\footnotesize",
    )


def table_bounds(text: str, label: str) -> tuple[int, int] | None:
    pos = text.find(rf"\label{{{label}}}")
    if pos < 0:
        return None
    start = text.rfind(r"\begin{table", 0, pos)
    end_marker = r"\end{table}"
    end = text.find(end_marker, pos)
    if start < 0 or end < 0:
        raise ValueError(f"Found {label}, but not its table environment")
    return start, end + len(end_marker)


def replace_table(text: str, label: str, content: str) -> tuple[str, bool]:
    bounds = table_bounds(text, label)
    if bounds is None:
        return text, False
    start, end = bounds
    return text[:start] + content + text[end:], True


def insert_after_table(text: str, after_label: str, content: str) -> tuple[str, bool]:
    bounds = table_bounds(text, after_label)
    if bounds is None:
        return text, False
    _, end = bounds
    return text[:end] + "\n\n% --- Generated table ---\n" + content + text[end:], True


def append_before_end(text: str, content: str) -> str:
    marker = r"\end{document}"
    pos = text.rfind(marker)
    block = "\n\n% --- Generated table appended by updater ---\n" + content + "\n"
    if pos < 0:
        return text.rstrip() + block
    return text[:pos].rstrip() + block + "\n" + text[pos:]


def replace_section(text: str, start_marker: str, end_marker: str, replacement: str) -> str:
    start = text.find(start_marker)
    if start < 0:
        raise ValueError(f"Could not find section start marker: {start_marker}")
    end = text.find(end_marker, start + len(start_marker))
    if end < 0:
        raise ValueError(f"Could not find section end marker after {start_marker}: {end_marker}")
    return text[:start] + replacement.rstrip() + "\n\n" + text[end:]


def update_narrative(text: str) -> str:
    sentence = (
        "Table~\\ref{tab:knn-type} isolates the same comparison at KNN density, "
        "holding the number of neighbours fixed while varying the ranking and "
        "weighting of bank-network links."
    )
    if "Table~\\ref{tab:knn-type} isolates the same comparison" in text:
        return text
    anchor = "The analysis above uses continuous branch-weighted links."
    pos = text.find(anchor)
    if pos >= 0:
        end = text.find("\n\n", pos)
        if end >= 0:
            return text[:end] + "\n" + sentence + text[end:]
    return text


def update_latex(
    source: Path,
    dest: Path,
    results_template: Path | None = None,
    appendix_template: Path | None = None,
) -> None:
    text = source.read_text(encoding="utf-8")
    if results_template:
        text = replace_section(
            text,
            r"\section{Results}",
            r"\section{Discussion}",
            read_latex(results_template),
        )
    if appendix_template:
        text = replace_section(
            text,
            r"\section{Full results}",
            r"\section{Estimation and inference details}",
            read_latex(appendix_template),
        )
    builders = [
        ("tab:placebo", lambda: build_placebo_tables()[0]),
        ("tab:placebo-full", lambda: build_placebo_tables()[1]),
        ("tab:banktype", lambda: build_banktype_table("Contig", "tab:banktype", "(contiguous-county sample)", main=True)),
        ("tab:banktype-full", lambda: build_banktype_table("Full", "tab:banktype-full", "(full sample)")),
        ("tab:banktype-contig", lambda: build_banktype_table("Contig", "tab:banktype-contig", "(contiguous-county sample)")),
        ("tab:banktype-noncontig", lambda: build_banktype_table("NonContig", "tab:banktype-noncontig", "(non-contiguous sample)")),
        ("tab:moran-summary", build_moran_summary),
        ("tab:moran-yearly", build_moran_yearly),
        ("tab:sem-main", build_sem_main),
        ("tab:sem-full", build_sem_full),
        ("tab:linkrestrict", build_linkrestrict),
        ("tab:lm", build_lm_table),
        ("tab:robust-snap", build_robust_snapshot),
        ("tab:reach", build_reach_table),
        ("tab:slx", lambda: build_slx_tables()[0]),
        ("tab:slx-full", lambda: build_slx_tables()[1]),
        ("tab:knn", build_knn_table),
        ("tab:knn-type", build_knn_type_comparison),
        ("tab:sar-full", build_sar_full),
        ("tab:hubs", build_hubs_table),
        ("tab:fgls", build_fgls_table),
        ("tab:jtest", build_jtest_table),
        ("tab:conley", build_conley_table),
        ("tab:w1994", build_w1994_table),
        ("tab:ivsar", build_ivsar_table),
    ]

    updated: list[str] = []
    inserted: list[str] = []
    skipped: list[tuple[str, str]] = []

    for label, builder in builders:
        try:
            content = builder()
            text, did_replace = replace_table(text, label, content)
            if did_replace:
                updated.append(label)
                continue
            if label == "tab:knn-type":
                text, did_insert = insert_after_table(text, "tab:knn", content)
                if did_insert:
                    inserted.append(label)
                    continue
            if label == "tab:ivsar":
                text = append_before_end(text, content)
                inserted.append(label)
                continue
            skipped.append((label, "label not found"))
        except Exception as exc:
            skipped.append((label, str(exc)[:120]))

    text = update_narrative(text)
    dest.write_text(text, encoding="utf-8", newline="\n")
    print(f"Updated {len(updated)}: {updated}")
    if inserted:
        print(f"Inserted {len(inserted)}: {inserted}")
    if skipped:
        for label, reason in skipped:
            print(f"  SKIP {label}: {reason}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=ROOT / "thesis_updated.tex")
    parser.add_argument("--results-template", type=Path)
    parser.add_argument("--appendix-template", type=Path)
    args = parser.parse_args()
    update_latex(args.source, args.output, args.results_template, args.appendix_template)


if __name__ == "__main__":
    main()
