#!/usr/bin/env python3
import argparse
import json
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

def export_drift_metrics(json_path: str, pushgateway_url: str, job: str = "data_drift"):
    """Export drift metrics to Prometheus Pushgateway"""
    
    print(f"Loading drift metrics from {json_path}")
    with open(json_path, 'r') as f:
        drift_data = json.load(f)
    
    # Create registry and metrics
    registry = CollectorRegistry()
    
    # Overall drift metrics
    max_drift_gauge = Gauge('data_drift_max_score', 'Maximum feature drift score', registry=registry)
    drift_detected_gauge = Gauge('data_drift_detected', 'Whether drift was detected (1=yes, 0=no)', registry=registry)
    threshold_gauge = Gauge('data_drift_threshold', 'Configured drift threshold', registry=registry)
    features_analyzed_gauge = Gauge('data_drift_features_analyzed', 'Number of features analyzed', registry=registry)
    
    # Set metric values
    max_drift_gauge.set(drift_data.get('max_drift_score', 0.0))
    drift_detected_gauge.set(1 if drift_data.get('drift_detected', False) else 0)
    threshold_gauge.set(drift_data.get('threshold', 0.1))
    features_analyzed_gauge.set(drift_data.get('total_features', 0))
    
    # Individual feature drift scores
    feature_drift_scores = drift_data.get('feature_drift_scores', {})
    if feature_drift_scores:
        feature_drift_gauge = Gauge('data_drift_feature_score', 'Drift score per feature', 
                                   ['feature'], registry=registry)
        
        for feature, score in feature_drift_scores.items():
            feature_drift_gauge.labels(feature=feature).set(score)
    
    # Push to gateway
    print(f"Pushing metrics to {pushgateway_url}")
    try:
        push_to_gateway(pushgateway_url, job=job, registry=registry)
        print("✅ Metrics pushed successfully")
    except Exception as e:
        print(f"❌ Failed to push metrics: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export drift metrics to Prometheus")
    parser.add_argument("--json", required=True, help="Path to drift report JSON")
    parser.add_argument("--pushgateway-url", required=True, help="Pushgateway URL")
    parser.add_argument("--job", default="data_drift", help="Prometheus job name")
    
    args = parser.parse_args()
    
    export_drift_metrics(args.json, args.pushgateway_url, args.job)