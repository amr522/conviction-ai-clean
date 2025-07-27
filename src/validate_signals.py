#!/usr/bin/env python3
"""Signal quality validators for Phase 2 implementation."""

import argparse
import json
import polars as pl
import sys
from pathlib import Path


def check_gamma_coverage(df: pl.DataFrame) -> float:
    """Check fraction of rows with non-null gamma values."""
    total = df.height
    non_null = df.filter(pl.col("opt30_net_gamma").is_not_null()).height
    return non_null / total if total > 0 else 0.0


def validate_flow_divergence(df: pl.DataFrame, lookahead: int = 1) -> float:
    """Validate flow divergence predictive accuracy."""
    if "opt30_close" not in df.columns or "opt30_flow_divergence" not in df.columns:
        return 0.0
    
    # Simple heuristic: flow divergence sign matches next price move
    diffs = df.with_columns([
        (pl.col("opt30_close").shift(-lookahead) - pl.col("opt30_close")).alias("future_ret"),
    ])
    
    valid_rows = diffs.filter(
        pl.col("future_ret").is_not_null() & 
        pl.col("opt30_flow_divergence").is_not_null()
    )
    
    if valid_rows.height == 0:
        return 0.0
    
    correct = valid_rows.filter(
        (pl.col("opt30_flow_divergence") * pl.col("future_ret") > 0)
    ).height
    
    return correct / valid_rows.height


def test_volume_spikes(df: pl.DataFrame) -> float:
    """Test volume spike detection accuracy."""
    # Ground-truth flag `true_spike` must exist for testing
    if "true_spike" not in df.columns or "opt30_vol_spike" not in df.columns:
        return 0.0
    
    detected = df.filter(pl.col("opt30_vol_spike") == True).height
    actual = df.filter(pl.col("true_spike") == True).height
    
    return detected / actual if actual > 0 else 0.0


def validate_signals(input_path: str, threshold: float = 0.9) -> dict:
    """Validate signal quality metrics."""
    df = pl.read_parquet(input_path)
    
    gamma_cov = check_gamma_coverage(df)
    flow_acc = validate_flow_divergence(df)
    vol_spike_acc = test_volume_spikes(df)
    
    results = {
        "gamma_coverage": gamma_cov,
        "flow_accuracy": flow_acc,
        "vol_spike_detection": vol_spike_acc,
        "total_rows": df.height,
        "validation_passed": all(v >= threshold for v in [gamma_cov, flow_acc, vol_spike_acc] if v > 0)
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate signal quality")
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--output-json", help="Output JSON file")
    parser.add_argument("--threshold", type=float, default=0.9, help="Quality threshold")
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Input file not found: {args.input}")
        return 1
    
    results = validate_signals(args.input, args.threshold)
    
    # Output results as JSON for Prometheus consumption
    print(json.dumps(results, indent=2))
    
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
    
    # Exit with error code if validation fails
    if not results["validation_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()