#!/usr/bin/env python3
# ============================================================
# EXECUTE DRY-RUN  ➜  BUILD FEATURES & LABELS  (Step 0-A)
# ============================================================
# 1) Run the dry-run validator script you just generated.
# 2) Parse the "CLEAN-layer validation successful for <DATE>" line.
# 3) Re-use that DATE to build features, labels and a joined
#    train-dataset parquet.
# 4) Run lag/leakage validators.
#
# NOTE:  • If the dry-run fails, the script aborts immediately.
#        • No new clean files are created unless missing (script logic).
# ------------------------------------------------------------

import os
import pathlib
import re
import subprocess
import sys

VALIDATOR = "python scripts/dry_run_schema_validation.py"


def run(cmd, capture=False):
    print("\n▶", cmd)
    if capture:
        return subprocess.check_output(cmd, shell=True, text=True)
    else:
        subprocess.check_call(cmd, shell=True)


# 1. Run dry-run validator and capture output
output = run(VALIDATOR, capture=True)

# 2. Extract DATE from the success banner
m = re.search(r"CLEAN-layer validation completed for (\d{4}-\d{2}-\d{2})", output)
if not m:
    print(output)  # echo for debugging
    sys.exit("❌ Dry-run validator failed – see output above.")

DATE = m.group(1)
print(f"\n✅ Using DATE = {DATE} for downstream steps\n")

# 3. Build features & labels
run(f"python src/calculate_features.py --date {DATE} --use-gpu")
run(f"python src/generate_labels.py --date {DATE}")

# 4. Join into train_dataset parquet
run(
    f"./scripts/generate-training-dataset.sh "
    f"data/Parquet_data/features_{DATE}.parquet "
    f"data/Parquet_data/labels_{DATE}.parquet "
    f"data/Parquet_data/train_dataset_{DATE}.parquet"
)

# 5. Lag / leakage validation
run(
    f"python validate_option_features.py --input-path "
    f"data/Parquet_data/train_dataset_{DATE}.parquet"
)
run(
    f"python validate_feature_lagging.py --input-path "
    f"data/Parquet_data/train_dataset_{DATE}.parquet"
)

print(
    f"\n🎉  FULL single-day pipeline completed for {DATE}\n"
    "   Next: launch the historical back-fill with run_historical_backfill.py"
)
