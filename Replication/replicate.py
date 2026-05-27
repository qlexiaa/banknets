"""
Python replication of:
    "Credit Supply and the Price of Housing"
    Favara, G. & Imbs, J. (2014), American Economic Review 104(3): 272-323

Replicates Tables 1-6, Figures 1-4, and Appendix Tables A1-A4.
"""

import warnings
warnings.filterwarnings("ignore")

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import pyreadstat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from linearmodels.panel import PanelOLS
from linearmodels.iv import IV2SLS

# ============================================================
# PATHS
# ============================================================
DATA = Path("Replication/20121416_1data/data")
OUT  = Path("Replication/output")
OUT.mkdir(exist_ok=True)

# ============================================================
# VARIABLE LISTS (mirrors Stata globals)
# ============================================================
# Year dummies: drop 1994 as base year
YR = [f"yryear_{y}" for y in range(1995, 2006)]

D_CTL    = ["Dl_inc", "LDl_inc", "Dl_pop", "LDl_pop",
            "Dl_hpi", "LDl_hpi", "Dl_her_v", "LDl_her_v"]
D_CTL_HP = ["Dl_inc", "LDl_inc", "Dl_pop", "LDl_pop",
            "Dl_her_v", "LDl_her_v"]
B_CTL    = ["etoa_b", "liatoa_b", "totltoa_b"]

# ============================================================
# DATA LOADING
# ============================================================
def load_merged():
    """
    Load hmda.dta, hp_dereg_controls.dta, call.dta and merge them
    (mirrors the Stata master.do merge block).
    """
    hmda, _  = pyreadstat.read_dta(DATA / "hmda.dta")
    hp,   _  = pyreadstat.read_dta(DATA / "hp_dereg_controls.dta")
    call, _  = pyreadstat.read_dta(DATA / "call.dta")

    # Drop duplicate columns before merging
    yr_cols  = [c for c in hp.columns if c.startswith("yryear")]
    hp_clean = hp.drop(columns=yr_cols + ["state_n"], errors="ignore")

    df = (hmda
          .merge(hp_clean, on=["county", "year"], how="left")
          .merge(call.drop(columns=["state_n"], errors="ignore"),
                 on=["county", "year"], how="left"))

    df["county"]  = df["county"].astype(int)
    df["year"]    = df["year"].astype(int)
    df["state_n"] = df["state_n"].astype(int)
    return df

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def _stars(p):
    if p < .01: return "***"
    if p < .05: return "**"
    if p < .10: return "*"
    return ""


def fe_reg(df, dep, indep, cluster="state_n", weight=None, mask=None):
    """
    Panel FE (entity effects) with state-clustered SE.
    Stata equivalent: xtreg dep indep, fe cl(state_n)
    Weighted version:  xtreg dep indep [aw=weight], fe cl(state_n)
    """
    sub  = df if mask is None else df[mask]
    cols = [dep] + indep + [cluster, "county", "year"]
    if weight:
        cols.append(weight)
    sub = sub[cols].dropna().copy()

    # Ensure uniqueness of (county, year) panel index
    sub = sub.drop_duplicates(subset=["county", "year"])
    pan = sub.set_index(["county", "year"])

    # Drop zero-variance columns (can happen when NAs reduce the sample
    # and some year dummies have no variation left)
    valid_indep = [c for c in indep
                   if pan[c].std() > 1e-12 and pan[c].nunique() > 1]

    kwds = dict(entity_effects=True)
    if weight:
        kwds["weights"] = pan[weight]

    mod = PanelOLS(pan[dep], pan[valid_indep], **kwds)
    return mod.fit(cov_type="clustered", clusters=pan[cluster])


def panel_iv(df, dep, endog, instr, exog, cluster="state_n",
             weight=None, bw=3, mask=None):
    """
    Panel IV-2SLS with entity FE and HAC (kernel) SE.
    Stata equivalent:
        xtivreg2 dep (endog=instr) exog yr*, fe r bw(bw) partial(yr*) [aw=weight]

    Implementation:
      1. Two-way demeaning (entity then time) – this is equivalent to Stata's
         'fe' + 'partial(yr*)'.  After demeaning, year dummies are redundant
         and dropped from exog so there is no rank deficiency.
      2. Non-year exogenous controls are included explicitly.
      3. IV2SLS with kernel (HAC) SE, bandwidth=bw.
    """
    sub      = df if mask is None else df[mask]

    # Separate year-dummy cols from other exog controls
    yr_cols      = [c for c in exog if c.startswith("yryear_")]
    other_exog   = [c for c in exog if not c.startswith("yryear_")]

    all_cols = [dep, endog, instr] + other_exog + ["county", "year"]
    if weight:
        all_cols.append(weight)
    sub = (sub[all_cols].dropna()
               .drop_duplicates(subset=["county", "year"])
               .copy()
               .reset_index(drop=True))

    demean_vars = [dep, endog, instr] + other_exog

    if weight:
        w = sub[weight].values
        def wdemean_entity(col):
            num   = sub.groupby("county").apply(
                        lambda g: (g[col] * g[weight]).sum())
            denom = sub.groupby("county")[weight].sum()
            wm    = (num / denom).rename("wm")
            return sub[col].values - wm.reindex(sub["county"]).values
        dm = {v: wdemean_entity(v) for v in demean_vars}
        # Also partial out time (year) effects from entity-demeaned data
        for v in demean_vars:
            s = pd.Series(dm[v], index=sub.index)
            dm[v] = (s - s.groupby(sub["year"].values).transform("mean")).values
        sqrt_w = np.sqrt(w)
        dm = {v: dm[v] * sqrt_w for v in dm}
    else:
        def demean_entity(col):
            return (sub[col]
                    - sub.groupby("county")[col].transform("mean")).values
        dm = {v: demean_entity(v) for v in demean_vars}
        # Also partial out time (year) effects
        for v in demean_vars:
            s = pd.Series(dm[v], index=sub.index)
            dm[v] = (s - s.groupby(sub["year"].values).transform("mean")).values

    Y    = pd.Series(dm[dep],   name=dep,   index=sub.index)
    E    = pd.DataFrame({endog: dm[endog]}, index=sub.index)
    Z    = pd.DataFrame({instr: dm[instr]}, index=sub.index)
    Xex  = (pd.DataFrame({v: dm[v] for v in other_exog}, index=sub.index)
            if other_exog else None)

    mod = IV2SLS(Y, Xex, E, Z)
    return mod.fit(cov_type="kernel", bandwidth=bw)


def print_table(title, results, main_vars, col_labels=None, fmt=".3f"):
    """Print a formatted regression table to stdout."""
    n   = len(results)
    lbl = col_labels or [f"({i+1})" for i in range(n)]
    W   = 22 + 13 * n
    sep = "─" * W

    print(f"\n{sep}")
    print(title)
    print(sep)
    print(f"{'':22}" + "".join(f"{l:>13}" for l in lbl))
    print(sep)

    for var in main_vars:
        coef_row = f"{var:<22}"
        se_row   = f"{'':22}"
        for res in results:
            if var in res.params.index:
                b  = res.params[var]
                se = res.std_errors[var]
                p  = res.pvalues[var]
                s  = _stars(p)
                coef_row += f"  {b:{fmt}}{s:<3}"
                se_row   += f"  ({se:{fmt}})   "
            else:
                coef_row += f"{'':>13}"
                se_row   += f"{'':>13}"
        print(coef_row)
        print(se_row)

    print(sep)
    for lbl_s, fn in [("N obs",     lambda r: f"{int(r.nobs):>10,}"),
                      ("R² within", lambda r: f"{r.rsquared:>13.3f}")]:
        row = f"{lbl_s:<22}"
        for res in results:
            try:    row += fn(res)
            except: row += f"{'':>13}"
        print(row)
    print()


def save_table_csv(rows, filename):
    pd.DataFrame(rows).to_csv(OUT / filename, index=False)
    print(f"  → saved {filename}")

# ============================================================
# TABLE 1: Summary Statistics
# ============================================================
def table_1():
    print("\n" + "="*70)
    print("TABLE 1: Descriptive Statistics on Mortgage Activity")
    print("="*70)

    tab1, _ = pyreadstat.read_dta(DATA / "data_tab1.dta")
    tab1["year"] = tab1["year"].astype(int)

    for s in ("b", "imc", "tfcu"):
        tab1[f"nar_{s}"]  = tab1[f"nrequested_{s}"]
        tab1[f"nlo_{s}"]  = tab1[f"noriginated_{s}"]
        n = tab1[f"noriginated_{s}"].replace(0, np.nan)
        tab1[f"amto_{s}"] = tab1[f"amtoriginated_{s}"]   / n
        tab1[f"inco_{s}"] = tab1[f"incomeoriginated_{s}"] / n

    vars_ = (["nar_b",  "nar_imc",  "nar_tfcu"]
           + ["nlo_b",  "nlo_imc",  "nlo_tfcu"]
           + ["amto_b", "amto_imc", "amto_tfcu"]
           + ["inco_b", "inco_imc", "inco_tfcu"])

    rows = []
    for v in vars_:
        rows.append({
            "variable": v,
            "All years": tab1[v].mean(),
            "1995":      tab1.loc[tab1.year==1995, v].mean(),
            "2000":      tab1.loc[tab1.year==2000, v].mean(),
            "2005":      tab1.loc[tab1.year==2005, v].mean(),
        })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format="{:.1f}".format))
    out.to_csv(OUT / "table_1.csv", index=False)
    print("  → saved table_1.csv")


# ============================================================
# TABLE 2: Credit Supply – Commercial Banks and Placebo Lenders
# ============================================================
def table_2(df):
    print("\n" + "="*70)
    print("TABLE 2: Mortgage Credit – Commercial Banks vs. Placebo Lenders")
    print("="*70)

    INSTR = "Linter_bra"
    rows  = []

    for panel_lbl, vars_ in [
        ("Panel A: Commercial Banks",
         ["Dl_nloans_b","Dl_vloans_b","Dl_nden_b","Dl_lir_b","Dl_nsold_b"]),
        ("Panel B: Placebo Lenders (TFCU & IMC)",
         ["Dl_nloans_pl","Dl_vloans_pl","Dl_nden_pl","Dl_lir_pl","Dl_nsold_pl"]),
    ]:
        results = []
        for var in vars_:
            lag = f"L{var}"
            indep = [INSTR] + D_CTL + ([lag] if lag in df.columns else []) + YR
            indep = [c for c in indep if c in df.columns]
            res   = fe_reg(df, var, indep)
            results.append(res)
            rows.append({"panel": panel_lbl, "dep": var,
                         "coef": res.params.get(INSTR),
                         "se":   res.std_errors.get(INSTR),
                         "pval": res.pvalues.get(INSTR),
                         "N":    int(res.nobs), "R2": res.rsquared})

        print_table(panel_lbl, results, [INSTR],
                    col_labels=["Nloans","Vloans","Nden","LIR","Nsold"])

    save_table_csv(rows, "table_2.csv")


# ============================================================
# TABLE 3: Credit Supply – Decomposition by Bank Type
# ============================================================
def table_3(df):
    print("\n" + "="*70)
    print("TABLE 3: Mortgage Credit – Decomposition by Bank Type")
    print("="*70)

    INSTR  = "Linter_bra"
    panels = {
        "Panel A: OOS banks – local branches": [
            "Dl_nloans_b_oos_lb","Dl_vloans_b_oos_lb","Dl_nden_b_oos_lb",
            "Dl_lir_b_oos_lb","Dl_nsold_b_oos_lb","Dl_nbra_oos_lb"],
        "Panel B: OOS banks – no local branches": [
            "Dl_nloans_b_oos","Dl_vloans_b_oos","Dl_nden_b_oos",
            "Dl_lir_b_oos","Dl_nsold_b_oos"],
        "Panel C: In-state banks": [
            "Dl_nloans_b_is","Dl_vloans_b_is","Dl_nden_b_is",
            "Dl_lir_b_is","Dl_nsold_b_is","Dl_nbra_is"],
    }

    rows = []
    for panel_lbl, vars_ in panels.items():
        results = []
        for var in vars_:
            lag   = f"L{var}"
            indep = [INSTR] + D_CTL + ([lag] if lag in df.columns else []) + YR
            indep = [c for c in indep if c in df.columns]
            res   = fe_reg(df, var, indep)
            results.append(res)
            rows.append({"panel": panel_lbl, "dep": var,
                         "coef": res.params.get(INSTR),
                         "se":   res.std_errors.get(INSTR),
                         "pval": res.pvalues.get(INSTR),
                         "N":    int(res.nobs), "R2": res.rsquared})

        print_table(panel_lbl, results, [INSTR])

    save_table_csv(rows, "table_3.csv")


# ============================================================
# TABLE 4: House Prices – Reduced Form
# ============================================================
def table_4(df):
    print("\n" + "="*70)
    print("TABLE 4: House Prices and Branching Deregulation (Reduced Form)")
    print("="*70)

    INSTR = "Linter_bra"

    # Five specifications from the paper
    specs = [
        ("Dl_hpi",   [INSTR]),
        ("Dl_hpi",   [INSTR, "Linter_ela"]),
        ("Dl_hpi",   [INSTR, "Linter_ela", "LDl_hpi"] + D_CTL_HP),
        ("Dl_hsosf", [INSTR, "Linter_inela"]),
        ("Dl_hsosf", [INSTR, "Linter_inela", "LDl_hsosf"] + D_CTL_HP),
    ]

    results = []
    rows    = []
    for i, (dep, extra) in enumerate(specs):
        indep = [c for c in extra + YR if c in df.columns]
        res   = fe_reg(df, dep, indep, weight="w1")
        results.append(res)
        for var in [INSTR, "Linter_ela", "Linter_inela"]:
            if var in res.params.index:
                rows.append({"col": i+1, "dep": dep, "var": var,
                             "coef": res.params[var],
                             "se":   res.std_errors[var],
                             "pval": res.pvalues[var],
                             "N":    int(res.nobs), "R2": res.rsquared})

    print_table("Table 4", results,
                [INSTR, "Linter_ela", "Linter_inela"],
                col_labels=["HPI(1)","HPI(2)","HPI(3)","HS(1)","HS(2)"],
                fmt=".5f")
    save_table_csv(rows, "table_4.csv")


# ============================================================
# TABLE 5: House Prices – IV/2SLS
# ============================================================
def table_5(df):
    print("\n" + "="*70)
    print("TABLE 5: House Prices and Mortgage Credit (IV/2SLS)")
    print("="*70)

    INSTR      = "Linter_bra"
    endog_vars = ["Dl_nloans_b", "Dl_vloans_b", "Dl_lir_b"]
    exog       = [c for c in ["LDl_hpi"] + D_CTL_HP + YR if c in df.columns]

    results = []
    rows    = []
    for endog in endog_vars:
        print(f"  Running IV: Dl_hpi ~ ({endog} = {INSTR}) ...", end=" ", flush=True)
        res = panel_iv(df, "Dl_hpi", endog, INSTR, exog, weight="w1", bw=3)
        results.append(res)
        b, se, p = (res.params[endog], res.std_errors[endog],
                    res.pvalues[endog])
        rows.append({"endog": endog, "coef": b, "se": se, "pval": p,
                     "N": int(res.nobs)})
        print(f"coef={b:.5f}{_stars(p)} ({se:.5f})  N={int(res.nobs)}")

    # Print compact summary
    print(f"\n{'':22}" + "".join(f"{e:>22}" for e in endog_vars))
    for lbl, fn in [("Coef (endog var)",
                     lambda r, e: f"{r.params[e]:.5f}{_stars(r.pvalues[e]):<3}"),
                    ("SE",
                     lambda r, e: f"({r.std_errors[e]:.5f})")]:
        row = f"{lbl:<22}"
        for res, e in zip(results, endog_vars):
            row += f"{fn(res, e):>22}"
        print(row)
    print(f"{'N obs':22}" + "".join(f"{int(r.nobs):>22,}" for r in results))

    save_table_csv(rows, "table_5.csv")
    return results


# ============================================================
# TABLE 6: Bank Balance Sheet and Deregulation
# ============================================================
def table_6(df):
    print("\n" + "="*70)
    print("TABLE 6: Bank Balance Sheet and Branching Deregulation")
    print("="*70)

    INSTR    = "Linter_bra"
    bss_vars = ["ifireltomtg_b","roa_b","iodtototd_b","nplrel_b","Dl_totd_b"]
    labels   = ["IFire/MTG","ROA","IOD/OTD","NPL","ΔTotDebt"]

    def run_bss(mask=None):
        res_list = []
        for var in bss_vars:
            extra = ["LDl_totd_b"] if var == "Dl_totd_b" else []
            indep = [INSTR] + B_CTL + D_CTL + extra + YR
            indep = [c for c in indep if c in df.columns]
            res_list.append(fe_reg(df, var, indep, mask=mask))
        return res_list

    rows = []
    for panel_lbl, mask in [
        ("Panel A: Full Sample",    None),
        ("Panel B: Border Counties", df["border"] == 1),
    ]:
        res_list = run_bss(mask)
        print_table(panel_lbl, res_list, [INSTR],
                    col_labels=labels, fmt=".5f")
        for var, res in zip(bss_vars, res_list):
            rows.append({"panel": panel_lbl, "dep": var,
                         "coef": res.params.get(INSTR),
                         "se":   res.std_errors.get(INSTR),
                         "pval": res.pvalues.get(INSTR),
                         "N":    int(res.nobs), "R2": res.rsquared})

    save_table_csv(rows, "table_6.csv")


# ============================================================
# FIGURE 1: Credit IRF via Projection Method
# ============================================================
def figure_1(df):
    """
    Impulse responses of mortgage credit to deregulation shock.
    Stata: xtreg F.h.var L.var D_control yr* Linter_bra, fe cl(state_n)
    h = 0 … 8
    """
    print("\nFigure 1: Credit IRF (projection method) ...")

    INSTR     = "Linter_bra"
    irf_vars  = ["Dl_nloans_b", "Dl_vloans_b", "Dl_lir_b"]
    H         = 9   # horizons 0 … 8

    coefs = {v: [] for v in irf_vars}
    ses   = {v: [] for v in irf_vars}

    df_s = df.sort_values(["county", "year"]).copy()

    for var in irf_vars:
        lag   = f"L{var}"
        indep = [lag] + D_CTL + YR + [INSTR]
        indep = [c for c in indep if c in df_s.columns]

        for h in range(H):
            col_name = var if h == 0 else f"F{h}_{var}"
            if h > 0:
                df_s[col_name] = (df_s.groupby("county")[var]
                                      .shift(-h))
            mask = df_s[col_name].notna() if h > 0 else None
            res  = fe_reg(df_s, col_name, indep, mask=mask)
            coefs[var].append(float(res.params[INSTR]))
            ses[var].append(float(res.std_errors[INSTR]))
            print(f"  {var} h={h}: β={coefs[var][-1]:.4f}")

    _plot_irf(
        coefs, ses, irf_vars,
        panel_titles=[
            "Panel A. Number of Originations",
            "Panel B. Volume of Loans",
            "Panel C. Loan-to-Income Ratio",
        ],
        suptitle=("Figure 1. Mortgage Credit by Commercial Banks:\n"
                  "Impulse Responses to Branching Deregulation Shock"),
        filename="figure_1.png",
    )
    return coefs, ses


# ============================================================
# FIGURE 3: House Price IRF via IV Projection Method
# ============================================================
def figure_3(df):
    """
    Impulse responses of house prices to instrumented credit shock.
    Stata: xtivreg2 F.h.Dl_hpi L.Dl_hpi D_control_hp yr* (endog=Linter_bra),
               fe r [aw=w1]
    """
    print("\nFigure 3: House Price IRF (IV projection) ...")

    INSTR     = "Linter_bra"
    endog_map = {
        "credit_n":   "Dl_nloans_b",
        "credit_v":   "Dl_vloans_b",
        "credit_lir": "Dl_lir_b",
    }
    H    = 9
    exog = [c for c in ["LDl_hpi"] + D_CTL_HP + YR if c in df.columns]

    coefs = {k: [] for k in endog_map}
    ses   = {k: [] for k in endog_map}

    df_s = df.sort_values(["county", "year"]).copy()

    for key, endog in endog_map.items():
        for h in range(H):
            dep_h = "Dl_hpi" if h == 0 else f"F{h}_Dl_hpi"
            if h > 0:
                df_s[dep_h] = df_s.groupby("county")["Dl_hpi"].shift(-h)

            sub = df_s[df_s[dep_h].notna()].copy() if h > 0 else df_s.copy()
            res = panel_iv(sub, dep_h, endog, INSTR, exog,
                           weight="w1", bw=3)
            coefs[key].append(float(res.params[endog]))
            ses[key].append(float(res.std_errors[endog]))
            print(f"  {key} h={h}: β={coefs[key][-1]:.4f}")

    _plot_irf(
        coefs, ses, list(endog_map.keys()),
        panel_titles=[
            "Panel A. Response to Number of Originations",
            "Panel B. Response to Volume of Loans",
            "Panel C. Response to Loan-to-Income Ratio",
        ],
        suptitle=("Figure 3. House Prices:\n"
                  "Impulse Responses to Instrumented Credit Shock"),
        filename="figure_3.png",
    )
    return coefs, ses


def _plot_irf(coefs, ses, keys, panel_titles, suptitle, filename):
    fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True)
    x = np.arange(len(list(coefs.values())[0]))

    for ax, key, title in zip(axes, keys, panel_titles):
        b  = np.array(coefs[key])
        se = np.array(ses[key])
        ax.plot(x, b,            "k-",  lw=1.8, label="Point estimate")
        ax.plot(x, b + 1.64*se, "k--", lw=1.0, label="90% CI")
        ax.plot(x, b - 1.64*se, "k--", lw=1.0)
        ax.axhline(0, color="gray", lw=0.8, ls=":")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Horizon (years)", fontsize=10)
    fig.suptitle(suptitle, fontsize=10, y=1.01)
    plt.tight_layout()
    fig.savefig(OUT / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → saved {filename}")


# ============================================================
# FIGURES 2 & 4: Aggregate Credit and House Prices
# ============================================================
def figure_2_4():
    """
    Actual vs. predicted aggregate mortgage credit (Fig 2)
    and house price index (Fig 4).

    Follows the aggregation_fig_2-4.do logic:
    county-year → state-year → national year.
    Predicted series use paper's estimated coefficients:
      credit growth: +2.8% per deregulation unit (Table 2, col 1)
      HP elasticity: 0.12 (Table 5)
    """
    print("\nFigures 2 & 4: Aggregate credit and house prices ...")

    agg, _ = pyreadstat.read_dta(DATA / "data_aggregation.dta")
    agg["year"]    = agg["year"].astype(int)
    agg["state_n"] = agg["state_n"].astype(int)

    # Real values
    agg["hpi_real"] = (agg["hpi"]            / agg["cpi2000"]) * 100
    agg["crs_obs"]  = (agg["amtoriginated_b"] / agg["cpi2000"]) * 100

    # Collapse to state-year
    sy = (agg.groupby(["state_n", "year"])
             .agg(hpis         = ("hpi_real", "mean"),
                  crs          = ("crs_obs",  "sum"),
                  yeard        = ("yeard",    "first"),
                  DLinter_bra  = ("DLinter_bra",  "first"),
                  DLinter_bra1 = ("DLinter_bra1", "first"),
                  year2        = ("year2",    "first"),
                  state        = ("state",    "first"))
             .reset_index())
    sy["crs"] = sy["crs"] / 1e6          # billions of 2000 USD
    sy = sy[sy["yeard"].notna()].copy()   # drop states never deregulated
    sy = sy.sort_values(["state_n", "year"]).reset_index(drop=True)

    # ── Build base prediction series (pcrs) ──────────────────────────
    # pcrs = crs in pre-deregulation years
    #       = crs * (1.028)^DLinter_bra in the first deregulation year
    #       = forward-fill (held constant) after that
    sy["pcrs"] = np.nan
    for idx, row in sy.iterrows():
        dlb  = row["DLinter_bra"]  if pd.notna(row["DLinter_bra"])  else 0.0
        dlb1 = row["DLinter_bra1"] if pd.notna(row["DLinter_bra1"]) else 0.0
        if dlb1 == 0:
            sy.loc[idx, "pcrs"] = row["crs"]
        elif dlb != 0 and dlb == dlb1:        # first deregulation year
            sy.loc[idx, "pcrs"] = row["crs"] * (1.028 ** dlb)
    sy["pcrs"] = sy.groupby("state_n")["pcrs"].ffill()

    # ── Permanent method (ppcrs): grow at +2.8% per unit each year ──
    sy["lpcrs"]  = np.log(sy["pcrs"].clip(lower=1e-10))
    sy["Dlpcrs"] = sy.groupby("state_n")["lpcrs"].diff().fillna(0.0)

    # Determine period (years since first deregulation)
    sy["period"] = 0
    post = (sy["DLinter_bra"] != sy["DLinter_bra1"]) & (sy["DLinter_bra1"] > 0)
    yrs_since = sy["year"] - sy["yeard"]
    sy.loc[post & (yrs_since <= 2),                    "period"] = 1
    sy.loc[post & (yrs_since > 2) & (yrs_since <= 4),  "period"] = 2
    sy.loc[post & (yrs_since > 4),                     "period"] = 3

    # Build ppcrs iteratively (permanent method)
    sy["ppcrs"] = sy["pcrs"].copy()
    prev = {}
    for idx, row in sy.iterrows():
        sn   = row["state_n"]
        dlb1 = row["DLinter_bra1"] if pd.notna(row["DLinter_bra1"]) else 0.0
        p    = row["period"]
        if sn in prev and p > 0 and dlb1 > 0:
            sy.loc[idx, "ppcrs"] = prev[sn] * (1.028 ** dlb1)
        prev[sn] = sy.loc[idx, "ppcrs"]

    # Aggregate to national year
    ny_cr = (sy.groupby("year")
               .agg(cr=("crs", "sum"), pcr_p=("ppcrs", "sum"))
               .reset_index())

    # ── Build base HP prediction series (phpis) ──────────────────────
    sy["Dlppcrs"] = sy.groupby("state_n")["ppcrs"].transform(
                        lambda x: np.log(x).diff().fillna(0.0))

    sy["phpis"] = np.nan
    for idx, row in sy.iterrows():
        dlb1 = row["DLinter_bra1"] if pd.notna(row["DLinter_bra1"]) else 0.0
        if dlb1 == 0:
            sy.loc[idx, "phpis"] = row["hpis"]
    sy["phpis"] = sy.groupby("state_n")["phpis"].ffill()

    # Permanent HP prediction: elasticity 0.12 (Table 5 estimate)
    sy["pphpis"] = sy["phpis"].copy()
    prev_hp = {}
    for idx, row in sy.iterrows():
        sn   = row["state_n"]
        dlb1 = row["DLinter_bra1"] if pd.notna(row["DLinter_bra1"]) else 0.0
        p    = row["period"]
        Dlcr = row["Dlppcrs"] if pd.notna(row["Dlppcrs"]) else 0.0
        if sn in prev_hp and p > 0 and dlb1 > 0:
            sy.loc[idx, "pphpis"] = prev_hp[sn] * (1.0 + 0.12 * Dlcr)
        prev_hp[sn] = sy.loc[idx, "pphpis"]

    ny_hp = (sy.groupby("year")
               .agg(hpia=("hpis", "mean"), phpi_p=("pphpis", "mean"))
               .reset_index())

    # ── Plot Figure 2 ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ny_cr["year"], ny_cr["cr"],    "k-",  lw=1.8,
            label="Actual mortgage volume")
    ax.plot(ny_cr["year"], ny_cr["pcr_p"], "k--", lw=1.2,
            label="Predicted: Permanent method")
    ax.set_xlabel("Year")
    ax.set_ylabel("Billions of 2000 U.S. Dollars")
    ax.set_title("Figure 2. Actual and Predicted Aggregate\n"
                 "Mortgage Volume by Commercial Banks", fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figure_2.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → saved figure_2.png")

    # ── Plot Figure 4 ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ny_hp["year"], ny_hp["hpia"],   "k-",  lw=1.8,
            label="Actual house price index")
    ax.plot(ny_hp["year"], ny_hp["phpi_p"], "k--", lw=1.2,
            label="Predicted: Permanent method")
    ax.set_xlabel("Year")
    ax.set_ylabel("House Price Index (2000 = 100)")
    ax.set_title("Figure 4. Actual and Predicted Real\n"
                 "House Price Index", fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figure_4.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → saved figure_4.png")


# ============================================================
# APPENDIX TABLES A1-A4: Border Counties
# ============================================================
def appendix_a1_a4(df):
    print("\n" + "="*70)
    print("APPENDIX TABLES A1-A4: Border Counties Robustness")
    print("="*70)

    INSTR  = "Linter_bra"
    border = df["border"] == 1

    # ── Table A1: Credit Supply – Border Counties ─────────────────────
    print("\nTable A1, Panel A: Commercial Banks (Border Counties)")
    rows_a1 = []
    for var in ["Dl_nloans_b","Dl_vloans_b","Dl_nden_b","Dl_lir_b","Dl_nsold_b"]:
        lag   = f"L{var}"
        indep = [INSTR] + D_CTL + ([lag] if lag in df.columns else []) + YR
        indep = [c for c in indep if c in df.columns]
        res   = fe_reg(df, var, indep, mask=border)
        b, se, p = (res.params.get(INSTR), res.std_errors.get(INSTR),
                    res.pvalues.get(INSTR))
        print(f"  {var:<24} {b:.3f}{_stars(p):<3} ({se:.3f})  N={int(res.nobs):,}")
        rows_a1.append({"dep": var, "panel": "A1_Banks_Border",
                        "coef": b, "se": se, "pval": p, "N": int(res.nobs)})

    print("\nTable A1, Panel B: Placebo Lenders (Border Counties)")
    for var in ["Dl_nloans_pl","Dl_vloans_pl","Dl_nden_pl","Dl_lir_pl","Dl_nsold_pl"]:
        lag   = f"L{var}"
        indep = [INSTR] + D_CTL + ([lag] if lag in df.columns else []) + YR
        indep = [c for c in indep if c in df.columns]
        res   = fe_reg(df, var, indep, mask=border)
        b, se, p = (res.params.get(INSTR), res.std_errors.get(INSTR),
                    res.pvalues.get(INSTR))
        print(f"  {var:<24} {b:.3f}{_stars(p):<3} ({se:.3f})  N={int(res.nobs):,}")
        rows_a1.append({"dep": var, "panel": "A1_Placebo_Border",
                        "coef": b, "se": se, "pval": p, "N": int(res.nobs)})
    save_table_csv(rows_a1, "table_A1.csv")

    # ── Table A3: House Prices Reduced Form – Border Counties ─────────
    print("\nTable A3: House Prices Reduced Form (Border Counties)")
    specs_a3 = [
        ("Dl_hpi",   [INSTR]),
        ("Dl_hpi",   [INSTR, "Linter_ela"]),
        ("Dl_hpi",   [INSTR, "Linter_ela", "LDl_hpi"] + D_CTL_HP),
        ("Dl_hsosf", [INSTR, "Linter_inela"]),
        ("Dl_hsosf", [INSTR, "Linter_inela", "LDl_hsosf"] + D_CTL_HP),
    ]
    rows_a3 = []
    for i, (dep, extra) in enumerate(specs_a3):
        indep = [c for c in extra + YR if c in df.columns]
        res   = fe_reg(df, dep, indep, weight="w1", mask=border)
        b, se, p = (res.params.get(INSTR), res.std_errors.get(INSTR),
                    res.pvalues.get(INSTR))
        print(f"  Col {i+1}  {dep:<12} {INSTR}: {b:.5f}{_stars(p):<3} "
              f"({se:.5f})  N={int(res.nobs):,}")
        rows_a3.append({"col": i+1, "dep": dep, "coef": b, "se": se,
                        "pval": p, "N": int(res.nobs), "R2": res.rsquared})
    save_table_csv(rows_a3, "table_A3.csv")

    # ── Table A4: House Prices IV – Border Counties ───────────────────
    print("\nTable A4: House Prices IV/2SLS (Border Counties)")
    exog  = [c for c in ["LDl_hpi"] + D_CTL_HP + YR if c in df.columns]
    rows_a4 = []
    for endog in ["Dl_nloans_b", "Dl_vloans_b", "Dl_lir_b"]:
        print(f"  IV ({endog}) ...", end=" ", flush=True)
        res = panel_iv(df[border].copy(), "Dl_hpi", endog, INSTR, exog,
                       weight="w1", bw=3)
        b, se, p = (res.params[endog], res.std_errors[endog],
                    res.pvalues[endog])
        print(f"coef={b:.5f}{_stars(p):<3} ({se:.5f})  N={int(res.nobs):,}")
        rows_a4.append({"endog": endog, "coef": b, "se": se, "pval": p,
                        "N": int(res.nobs)})
    save_table_csv(rows_a4, "table_A4.csv")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import time

    print("="*70)
    print("Replication: Credit Supply and the Price of Housing")
    print("Favara & Imbs (2014), American Economic Review")
    print("="*70)

    print("\nLoading and merging data ...")
    df = load_merged()
    print(f"  Observations : {len(df):,}")
    print(f"  Counties     : {df['county'].nunique():,}")
    print(f"  States       : {df['state_n'].nunique():,}")
    print(f"  Years        : {df['year'].min()} – {df['year'].max()}")
    print(f"  Border obs   : {(df['border']==1).sum():,}")

    t0 = time.time()

    table_1()
    table_2(df)
    table_3(df)
    table_4(df)
    table_5(df)
    table_6(df)
    figure_1(df)
    figure_3(df)
    figure_2_4()
    appendix_a1_a4(df)

    print("\n" + "="*70)
    print(f"Done in {time.time()-t0:.0f}s  |  Output saved to Replication\output")
    print("="*70)
