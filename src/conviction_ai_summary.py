#!/usr/bin/env python3
"""
Conviction-AI summary command for daily pipeline status.
"""

import argparse
import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl


def get_pipeline_status(target_date: str) -> Dict:
    """Get pipeline status for a specific date."""
    status = {
        "date": target_date,
        "features": {"status": "unknown", "count": 0, "path": None},
        "labels": {"status": "unknown", "count": 0, "path": None},
        "training_dataset": {"status": "unknown", "count": 0, "path": None},
        "models": {"status": "unknown", "count": 0, "latest": None},
        "validation": {"status": "unknown", "results": {}},
        "drift": {"status": "unknown", "score": None},
        "signals": {"status": "unknown", "metrics": {}},
    }

    date_suffix = datetime.strptime(target_date, "%Y-%m-%d").strftime("%Y%m%d")
    data_dir = Path("data/Parquet_data")

    # Check features
    feature_path = data_dir / f"features_{date_suffix}.parquet"
    if feature_path.exists():
        try:
            df = pl.read_parquet(feature_path)
            status["features"] = {
                "status": "available",
                "count": len(df),
                "path": str(feature_path),
                "columns": len(df.columns),
            }
        except Exception as e:
            status["features"]["status"] = f"error: {e}"
    else:
        status["features"]["status"] = "missing"

    # Check labels
    label_path = data_dir / f"labels_{target_date}.parquet"
    if label_path.exists():
        try:
            df = pl.read_parquet(label_path)
            status["labels"] = {
                "status": "available",
                "count": len(df),
                "path": str(label_path),
            }
        except Exception as e:
            status["labels"]["status"] = f"error: {e}"
    else:
        status["labels"]["status"] = "missing"

    # Check training dataset
    train_path = data_dir / f"train_dataset_{target_date}.parquet"
    if train_path.exists():
        try:
            df = pl.read_parquet(train_path)
            status["training_dataset"] = {
                "status": "available",
                "count": len(df),
                "path": str(train_path),
            }
        except Exception as e:
            status["training_dataset"]["status"] = f"error: {e}"
    else:
        status["training_dataset"]["status"] = "missing"

    # Check models
    models_dir = Path("models")
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pkl"))
        if model_files:
            latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
            status["models"] = {
                "status": "available",
                "count": len(model_files),
                "latest": str(latest_model),
                "last_modified": datetime.fromtimestamp(
                    latest_model.stat().st_mtime
                ).isoformat(),
            }
        else:
            status["models"]["status"] = "missing"

    # Check validation results
    metrics_dir = Path("metrics")
    if metrics_dir.exists():
        validation_files = list(metrics_dir.glob(f"*{target_date}*.json"))
        if validation_files:
            status["validation"]["status"] = "available"
            for vf in validation_files:
                try:
                    with open(vf) as f:
                        data = json.load(f)
                    status["validation"]["results"][vf.name] = data
                except Exception:
                    pass
        else:
            status["validation"]["status"] = "missing"

    # Check drift report
    drift_file = Path(f"drift_report_{target_date}.json")
    if drift_file.exists():
        try:
            with open(drift_file) as f:
                drift_data = json.load(f)
            status["drift"] = {
                "status": "available",
                "score": drift_data.get("max_drift_score"),
                "threshold": drift_data.get("threshold"),
                "features_analyzed": drift_data.get("features_analyzed"),
            }
        except Exception as e:
            status["drift"]["status"] = f"error: {e}"
    else:
        status["drift"]["status"] = "missing"

    return status


def format_status_output(status: Dict, format_type: str = "text") -> str:
    """Format status output for display."""
    if format_type == "json":
        return json.dumps(status, indent=2)

    # Text format
    output = []
    output.append(f"📊 Conviction-AI Pipeline Summary for {status['date']}")
    output.append("=" * 50)

    # Features
    feat = status["features"]
    if feat["status"] == "available":
        output.append(
            f"✅ Features: {feat['count']:,} records, {feat.get('columns', 0)} columns"
        )
    else:
        output.append(f"❌ Features: {feat['status']}")

    # Labels
    labels = status["labels"]
    if labels["status"] == "available":
        output.append(f"✅ Labels: {labels['count']:,} records")
    else:
        output.append(f"❌ Labels: {labels['status']}")

    # Training dataset
    train = status["training_dataset"]
    if train["status"] == "available":
        output.append(f"✅ Training Dataset: {train['count']:,} records")
    else:
        output.append(f"❌ Training Dataset: {train['status']}")

    # Models
    models = status["models"]
    if models["status"] == "available":
        output.append(f"✅ Models: {models['count']} available")
        output.append(f"   Latest: {Path(models['latest']).name}")
    else:
        output.append(f"❌ Models: {models['status']}")

    # Validation
    val = status["validation"]
    if val["status"] == "available":
        output.append(f"✅ Validation: {len(val['results'])} reports")
    else:
        output.append(f"❌ Validation: {val['status']}")

    # Drift
    drift = status["drift"]
    if drift["status"] == "available":
        score = drift.get("score", 0)
        threshold = drift.get("threshold", 0.1)
        drift_status = "🟢" if score < threshold else "🔴"
        output.append(f"{drift_status} Drift: {score:.3f} (threshold: {threshold:.3f})")
    else:
        output.append(f"❌ Drift: {drift['status']}")

    return "\n".join(output)


def get_recent_runs(days: int = 7) -> List[Dict]:
    """Get summary of recent pipeline runs."""
    runs = []
    data_dir = Path("data/Parquet_data")

    # Find all feature files
    feature_files = list(data_dir.glob("features_*.parquet"))

    for feature_file in sorted(feature_files, reverse=True)[:days]:
        # Extract date from filename
        date_part = feature_file.stem.split("_")[-1]
        try:
            run_date = datetime.strptime(date_part, "%Y%m%d").strftime("%Y-%m-%d")
            status = get_pipeline_status(run_date)
            runs.append(
                {
                    "date": run_date,
                    "features_count": status["features"].get("count", 0),
                    "has_labels": status["labels"]["status"] == "available",
                    "has_training_data": status["training_dataset"]["status"]
                    == "available",
                    "validation_status": status["validation"]["status"],
                }
            )
        except ValueError:
            continue

    return runs


def main():
    """CLI for pipeline summary."""
    parser = argparse.ArgumentParser(description="Conviction-AI pipeline summary")
    parser.add_argument("--date", required=True, help="Date (YYYY-MM-DD)")
    parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )
    parser.add_argument("--recent", type=int, help="Show recent runs (number of days)")

    args = parser.parse_args()

    if args.recent:
        runs = get_recent_runs(args.recent)
        if args.format == "json":
            print(json.dumps(runs, indent=2))
        else:
            print(f"📈 Recent Pipeline Runs ({args.recent} days)")
            print("=" * 40)
            for run in runs:
                status_icons = []
                status_icons.append("✅" if run["features_count"] > 0 else "❌")
                status_icons.append("🏷️" if run["has_labels"] else "⚪")
                status_icons.append("🎯" if run["has_training_data"] else "⚪")
                status_icons.append(
                    "✅" if run["validation_status"] == "available" else "❌"
                )

                print(
                    f"{run['date']}: {' '.join(status_icons)} ({run['features_count']:,} features)"
                )
    else:
        status = get_pipeline_status(args.date)
        print(format_status_output(status, args.format))


if __name__ == "__main__":
    main()
