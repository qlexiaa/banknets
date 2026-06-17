"""
run_all.py -- Credit-growth spatial-econometric pipeline.

Main dependent variable: Dl_nloans_b (credit growth).
Samples: Full, Contig (border==1), NonContig (border==0).
"""
from pathlib import Path
import sys


ANALYSIS_DIR = Path(__file__).parent
ROOT = ANALYSIS_DIR.parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ANALYSIS_DIR))

from diagnostics import moran_baseline, moran_bank_variants, network_overlap
from diagnostics import w_density_summary, bank_location_branches
from extensions import multiplier_counterfactual, slx_exposure
from inference import conley_se, fgls
from model_selection import jtest, lm_tests
from sar import multiplier_decomposition, sar_credit, sar_iv_credit
from sem import (
    sem_credit,
    sem_knn_sweep,
    sem_link_restrictions,
    sem_w1994,
    sem_w_variants,
)


def run_step(label, func):
    print(f"\n=== {label} ===")
    func()


def main():
    run_step("1. Baseline Moran's I (W_geo, credit)",
             lambda: moran_baseline.run(OUTPUT_DIR))
    run_step("2. Moran's I under W_bank variants",
             lambda: moran_bank_variants.run(OUTPUT_DIR))
    run_step("3. Bank-geography link overlap",
             lambda: network_overlap.run(OUTPUT_DIR))
    run_step("4. W matrix density summary",
             lambda: w_density_summary.run(OUTPUT_DIR))
    run_step("5. LM error/lag diagnostics, credit DV",
             lambda: lm_tests.run(OUTPUT_DIR))
    run_step("6. Main Panel_FE_Error: W_geo vs W_bank",
             lambda: sem_credit.run(OUTPUT_DIR))
    run_step("7. W_geo / W_bank_bin / W_bank_count / W_bank_nonGeo",
             lambda: sem_w_variants.run(OUTPUT_DIR))
    run_step("8. nonGeo / interstate / intrastate restrictions",
             lambda: sem_link_restrictions.run(OUTPUT_DIR))
    run_step("9. KNN sweep k = 1..20",
             lambda: sem_knn_sweep.run(OUTPUT_DIR))
    run_step("10. Davidson-MacKinnon J-tests",
             lambda: jtest.run(OUTPUT_DIR))
    run_step("11. Panel_FE_Lag robustness",
             lambda: sar_credit.run(OUTPUT_DIR))
    run_step("12. IV-SAR and SDM-IV robustness",
             lambda: sar_iv_credit.run(OUTPUT_DIR))
    run_step("13. Direct/indirect effects, distance decay",
             lambda: multiplier_decomposition.run(OUTPUT_DIR))
    run_step("14. SE estimator comparison (Conley 1999; Colella et al. 2019)",
             lambda: conley_se.run(OUTPUT_DIR))
    run_step("15. FGLS vs OLS point-estimator comparison",
             lambda: fgls.run(OUTPUT_DIR))
    run_step("16. Bank-network SLX exposure augmentation",
             lambda: slx_exposure.run(OUTPUT_DIR))
    run_step("17. SAR multiplier decomposition and counterfactual",
             lambda: multiplier_counterfactual.run(OUTPUT_DIR))
    run_step("18. Look-ahead robustness -- W_bank_1994",
             lambda: sem_w1994.run(OUTPUT_DIR))
    run_step("19. Bank-type decomposition (Favara-Imbs Table 3 / A2)",
             lambda: bank_location_branches.run(OUTPUT_DIR))


if __name__ == "__main__":
    main()
