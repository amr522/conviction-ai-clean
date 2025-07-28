#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def check_data_drift(
    reference_path: str,
    current_path: str,
    drift_enabled: bool = False,
    drift_threshold: float = 0.1,
    json_output: str = None,
) -> dict:
    """Check for data drift between reference and current datasets using statistical tests"""

    print(f"Loading reference data from {reference_path}")
    reference_data = pd.read_parquet(reference_path)

    print(f"Loading current data from {current_path}")
    current_data = pd.read_parquet(current_path)

    # Align columns between datasets
    common_cols = list(set(reference_data.columns) & set(current_data.columns))
    numeric_cols = (
        reference_data[common_cols].select_dtypes(include=["number"]).columns.tolist()
    )

    print(f"Analyzing drift for {len(numeric_cols)} numeric features")

    # Calculate drift scores using Kolmogorov-Smirnov test
    drift_scores = {}

    for col in numeric_cols:
        ref_values = reference_data[col].dropna()
        cur_values = current_data[col].dropna()

        if len(ref_values) > 0 and len(cur_values) > 0:
            # Use KS test statistic as drift score
            ks_stat, p_value = stats.ks_2samp(ref_values, cur_values)
            drift_scores[col] = ks_stat
        else:
            drift_scores[col] = 0.0

    # Get overall drift score
    overall_drift = False
    max_drift_score = 0.0

    if drift_scores:
        max_drift_score = max(drift_scores.values())
        overall_drift = max_drift_score > drift_threshold

    result = {
        "drift_detected": bool(overall_drift),
        "max_drift_score": float(max_drift_score),
        "threshold": float(drift_threshold),
        "feature_drift_scores": {k: float(v) for k, v in drift_scores.items()},
        "total_features": int(len(numeric_cols)),
    }

    # Write JSON output if requested
    if json_output:
        Path(json_output).parent.mkdir(parents=True, exist_ok=True)
        with open(json_output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Drift report written to {json_output}")

    # Print summary
    print(f"Drift Analysis Summary:")
    print(f"  Max drift score: {max_drift_score:.4f}")
    print(f"  Threshold: {drift_threshold}")
    print(f"  Drift detected: {overall_drift}")
    print(f"  Features analyzed: {len(numeric_cols)}")

    # Exit with error if drift detected and enabled
    if drift_enabled and overall_drift:
        print(
            f"❌ Data drift detected (max score: {max_drift_score:.4f} > {drift_threshold})"
        )
        sys.exit(1)
    elif drift_enabled:
        print(f"✅ No significant data drift detected")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check for data drift between datasets"
    )
    parser.add_argument("--reference", required=True, help="Path to reference dataset")
    parser.add_argument("--current", required=True, help="Path to current dataset")
    parser.add_argument(
        "--drift-enabled",
        action="store_true",
        help="Enable drift detection (fail on drift)",
    )
    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=0.1,
        help="Drift threshold (default: 0.1)",
    )
    parser.add_argument("--drift-report-json", help="Output path for drift report JSON")

    args = parser.parse_args()

    check_data_drift(
        reference_path=args.reference,
        current_path=args.current,
        drift_enabled=args.drift_enabled,
        drift_threshold=args.drift_threshold,
        json_output=args.drift_report_json,
    )
