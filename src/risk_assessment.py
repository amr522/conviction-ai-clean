#!/usr/bin/env python3
import argparse
import json
import polars as pl
from typing import Dict

# Define risk thresholds
RISK_THRESHOLDS = {
    "data_gaps": 0.95,
    "signal_noise": 0.90,
    "performance": 0.50
}

def assess_data_gaps(df: pl.DataFrame) -> float:
    """Fraction of complete rows across all core features."""
    cols = [c for c in df.columns if c.startswith(("optd_", "opt30_", "stock_"))]
    if not cols:
        return 0.0
    
    complete = df.drop_nulls(cols).height
    total = df.height
    return complete / total if total else 0.0

def assess_signal_noise(metrics: Dict[str, float]) -> float:
    """Average of gamma, flow, vol spike metrics."""
    if not metrics:
        return 0.0
    return sum(metrics.values()) / len(metrics)

def assess_performance(latency_ms: float) -> float:
    """Convert latency to performance score [0..1]."""
    return max(0.0, 1 - (latency_ms / 1000))

def run_risk_assessment(
    parquet_path: str,
    signal_metrics_json: str,
    latency_ms: float,
    output_path: str
):
    """Run comprehensive risk assessment."""
    df = pl.read_parquet(parquet_path)
    data_gap_score = assess_data_gaps(df)

    with open(signal_metrics_json) as f:
        metrics = json.load(f)
    noise_score = assess_signal_noise(metrics)

    perf_score = assess_performance(latency_ms)

    results = {
        "data_gaps": data_gap_score,
        "signal_noise": noise_score,
        "performance": perf_score
    }
    
    # Determine which risks exceed threshold
    risks = {k: v for k, v in results.items() if v < RISK_THRESHOLDS[k]}
    report = {"scores": results, "risks": risks, "thresholds": RISK_THRESHOLDS}
    
    print(json.dumps(report, indent=2))

    if risks:
        print(f"❌ Risks above threshold: {list(risks.keys())}")
        exit(1)
    
    print("✅ All risk scores are within acceptable limits")

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Run risk assessment")
    parser.add_argument("--parquet", required=True, help="Path to parquet file")
    parser.add_argument("--signal-metrics", required=True, help="Path to signal metrics JSON")
    parser.add_argument("--latency-ms", type=float, default=200, help="Latency in milliseconds")
    parser.add_argument("--output-path", required=True, help="Output path for risk report")
    
    args = parser.parse_args()
    
    run_risk_assessment(args.parquet, args.signal_metrics, args.latency_ms, args.output_path)

if __name__ == "__main__":
    main()