#!/usr/bin/env python3
import argparse
import json
import sys
import polars as pl
from typing import Dict


def check_gamma_coverage(df: pl.DataFrame) -> float:
    """Fraction of non-null net_gamma rows."""
    total = df.height
    non_null = df.filter(pl.col("opt30_net_gamma").is_not_null()).height
    return non_null / total if total else 0.0


def validate_flow_divergence(df: pl.DataFrame) -> float:
    """Predictive accuracy of flow divergence signals."""
    if "ret_30m" not in df.columns:
        # Create synthetic return column if not present
        df = df.with_columns(
            (pl.col("opt30_close").shift(-1) - pl.col("opt30_close")).alias("ret_30m")
        )
    
    df = df.with_columns((pl.col("ret_30m") > 0).alias("up"))
    
    correct = df.filter(
        ((pl.col("opt30_flow_divergence") > 0) & pl.col("up")) |
        ((pl.col("opt30_flow_divergence") < 0) & ~pl.col("up"))
    ).height
    
    total = df.filter(pl.col("ret_30m").is_not_null()).height
    return correct / total if total else 0.0


def test_volume_spikes(df: pl.DataFrame) -> float:
    """Volume spike detection accuracy using true_spike ground truth."""
    if "true_spike" not in df.columns:
        return 0.0
    
    # True positives: detected and actual
    tp = df.filter(pl.col("opt30_vol_spike") & pl.col("true_spike")).height
    # False negatives: not detected but actual
    fn = df.filter(~pl.col("opt30_vol_spike") & pl.col("true_spike")).height
    
    # Recall: tp / (tp + fn)
    return tp / (tp + fn) if (tp + fn) else 0.0


def run_validations(input_path: str, threshold: float) -> Dict[str, float]:
    """Run advanced signal validations and exit on failure."""
    df = pl.read_parquet(input_path)
    
    gamma_cov = check_gamma_coverage(df)
    flow_acc = validate_flow_divergence(df)
    vol_spk = test_volume_spikes(df)
    
    results = {
        "gamma_coverage": gamma_cov,
        "flow_accuracy": flow_acc,
        "vol_spike_detection": vol_spk
    }
    
    print(results)
    
    failed = [k for k, v in results.items() if v < threshold]
    if failed:
        print(f"❌ Signal validation failed for: {failed}, threshold={threshold}")
        sys.exit(1)
    
    print(f"✅ All signals meet threshold {threshold}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate advanced signal quality")
    parser.add_argument("--input", required=True, help="Path to options_30min parquet")
    parser.add_argument("--threshold", type=float, default=0.9, help="Quality threshold")
    parser.add_argument("--output-json", help="Write results JSON")
    
    args = parser.parse_args()
    
    results = run_validations(args.input, args.threshold)
    
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    main()