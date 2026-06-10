"""
run_all.py -- Current credit-growth thesis pipeline.

Main dependent variable: Dl_nloans_b (credit growth).
"""
from pathlib import Path
import sys


ANALYSIS_DIR = Path(__file__).parent
ROOT = ANALYSIS_DIR.parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Let sibling analysis modules import utils.py when run from the project root.
sys.path.insert(0, str(ANALYSIS_DIR))

import bank_geo_overlap
import bank_moran_diagnostics
import bank_nongeo_credit
import diagnostics
import knn_crossover_credit
import lm_diagnostics_credit
import panel_fe_credit
import panel_fe_hpi
import rq1_composite_w
import rq1_composite_w_hpi
import rq1_four_w_comparison_credit
import rq1_jtest
import spatial_multiplier_decomposition
import sar_robustness_credit
import conley_se_comparison
import sar_iv_credit
import fgls_comparison


def run_step(label, func):
    print(f"\n=== {label} ===")
    func()


def main():
    run_step("1. Baseline diagnostics", lambda: diagnostics.run(OUTPUT_DIR))
    run_step("2. Moran's I by W_bank specification", lambda: bank_moran_diagnostics.run(OUTPUT_DIR))
    run_step("3. Bank-geography overlap", lambda: bank_geo_overlap.run(OUTPUT_DIR))
    run_step("4. LM diagnostics, credit growth", lambda: lm_diagnostics_credit.run(OUTPUT_DIR))
    run_step("5. Panel FE SEM, credit growth", lambda: panel_fe_credit.run(OUTPUT_DIR))
    run_step("6. Panel FE SEM, HPI companion", lambda: panel_fe_hpi.run(OUTPUT_DIR))
    run_step("7. Composite W sweep, HPI companion", lambda: rq1_composite_w_hpi.run(OUTPUT_DIR))
    run_step("8. Four-W comparison, credit growth", lambda: rq1_four_w_comparison_credit.run(OUTPUT_DIR))
    run_step("9. Non-geographic W_bank, credit growth", lambda: bank_nongeo_credit.run(OUTPUT_DIR))
    run_step("10. KNN sweep, credit growth", lambda: knn_crossover_credit.run(OUTPUT_DIR))
    run_step("11. J-test, credit growth", lambda: rq1_jtest.run(OUTPUT_DIR))
    run_step("12. Composite W sweep, credit growth", lambda: rq1_composite_w.run(OUTPUT_DIR))
    run_step("13. Spatial multiplier decomposition", lambda: spatial_multiplier_decomposition.run(OUTPUT_DIR))
    run_step("14. SAR robustness, credit growth",    lambda: sar_robustness_credit.run(OUTPUT_DIR))
    run_step("15. Conley HAC SE comparison",         lambda: conley_se_comparison.run(OUTPUT_DIR))
    run_step("16. IV-SAR Kelejian-Prucha",            lambda: sar_iv_credit.run(OUTPUT_DIR))
    run_step("17. FGLS comparison, credit growth",    lambda: fgls_comparison.run(OUTPUT_DIR))


if __name__ == "__main__":
    main()
