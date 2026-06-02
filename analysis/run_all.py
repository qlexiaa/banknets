"""
run_all.py -- Master analysis pipeline
Runs every analysis module in dependency order and saves all results to output/.
No console output.
"""
from pathlib import Path
import sys

# Ensure analysis/ is on path
sys.path.insert(0, str(Path(__file__).parent))

ROOT       = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Import modules (with __name__ guard, importing does NOT run computation)
import panel_fe_error
import spatial_hpi_models
import panel_fe_knn
import knn_crossover
import bank_geo_overlap
import bank_nongeo
import lambda_time
import lambda_time_yearspecific
import rq1_inference
import rq1_lm_tests
import rq1_four_w_comparison
import rq1_jtest
import rq1_composite_w

# Run all in order
panel_fe_error.run(OUTPUT_DIR)
spatial_hpi_models.run(OUTPUT_DIR)
panel_fe_knn.run(OUTPUT_DIR)
knn_crossover.run(OUTPUT_DIR)
bank_geo_overlap.run(OUTPUT_DIR)
bank_nongeo.run(OUTPUT_DIR)
lambda_time.run(OUTPUT_DIR)
lambda_time_yearspecific.run(OUTPUT_DIR)
rq1_inference.run(OUTPUT_DIR)
rq1_lm_tests.run(OUTPUT_DIR)
rq1_four_w_comparison.run(OUTPUT_DIR)
rq1_jtest.run(OUTPUT_DIR)
rq1_composite_w.run(OUTPUT_DIR)
