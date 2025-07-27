#!/usr/bin/env python3
import os
import pickle
import numpy as np
import polars as pl
from pathlib import Path

def load_model(model_path: str):
    """Load trained model from pickle file"""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def explain_predictions(model, X: pl.DataFrame, pushgateway_url=None):
    """Compute SHAP explanations and optionally push to Prometheus"""
    try:
        import shap
    except ImportError:
        print("⚠️ SHAP not installed, skipping explanations")
        return {}
    
    # Convert to numpy for SHAP
    X_np = X.select([col for col in X.columns if X[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]).to_numpy()
    feature_names = [col for col in X.columns if X[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
    
    if X_np.shape[1] == 0:
        print("⚠️ No numeric features found for SHAP analysis")
        return {}
    
    print(f"Computing SHAP explanations for {X_np.shape[1]} features on {X_np.shape[0]} samples")
    
    # Create explainer based on model type
    try:
        if hasattr(model, 'predict') and hasattr(model, 'feature_importances_'):
            # Tree-based model (RandomForest, XGBoost, LightGBM)
            explainer = shap.TreeExplainer(model)
        else:
            # Fallback to KernelExplainer for other models
            explainer = shap.KernelExplainer(model.predict, X_np[:100])  # Use sample for background
    except Exception as e:
        print(f"⚠️ Could not create SHAP explainer: {e}")
        return {}
    
    # Compute SHAP values
    try:
        shap_values = explainer.shap_values(X_np)
        
        # Handle multi-output case
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # Use first output for multi-class
        
        # Compute mean absolute SHAP per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_summary = dict(zip(feature_names, mean_abs_shap))
        
        print(f"✅ SHAP analysis completed. Top features:")
        sorted_features = sorted(shap_summary.items(), key=lambda x: x[1], reverse=True)
        for feat, importance in sorted_features[:5]:
            print(f"  {feat}: {importance:.4f}")
        
        # Push metrics if requested
        if pushgateway_url:
            push_shap_metrics(shap_summary, pushgateway_url)
        
        return shap_summary
        
    except Exception as e:
        print(f"⚠️ SHAP computation failed: {e}")
        return {}

def push_shap_metrics(shap_summary: dict, url: str, job: str = "model_explain"):
    """Push SHAP metrics to Prometheus Pushgateway"""
    try:
        from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
        
        registry = CollectorRegistry()
        
        # Mean absolute SHAP per feature
        shap_gauge = Gauge('shap_mean_abs', 'Mean absolute SHAP value', ['feature'], registry=registry)
        
        for feature, value in shap_summary.items():
            shap_gauge.labels(feature=feature).set(float(value))
        
        # Overall metrics
        total_importance = sum(shap_summary.values())
        max_importance = max(shap_summary.values()) if shap_summary else 0.0
        
        total_gauge = Gauge('shap_total_importance', 'Total SHAP importance', registry=registry)
        max_gauge = Gauge('shap_max_importance', 'Maximum feature SHAP importance', registry=registry)
        features_gauge = Gauge('shap_features_analyzed', 'Number of features analyzed', registry=registry)
        
        total_gauge.set(float(total_importance))
        max_gauge.set(float(max_importance))
        features_gauge.set(len(shap_summary))
        
        print(f"Pushing SHAP metrics to {url}")
        push_to_gateway(url, job=job, registry=registry)
        print("✅ SHAP metrics pushed successfully")
        
    except Exception as e:
        print(f"❌ Failed to push SHAP metrics: {e}")

def run_inference_with_explanations(model_path: str, feature_path: str, 
                                  pushgateway_url: str = None, output_path: str = None):
    """Run inference with SHAP explanations"""
    
    # Load model and features
    print(f"Loading model from {model_path}")
    model = load_model(model_path)
    
    print(f"Loading features from {feature_path}")
    features = pl.read_parquet(feature_path)
    
    # Make predictions
    print("Making predictions...")
    X_numeric = features.select([col for col in features.columns if features[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]])
    predictions = model.predict(X_numeric.to_numpy())
    
    # Add predictions to dataframe
    result_df = features.with_columns(pl.Series("prediction", predictions))
    
    # Compute SHAP explanations
    shap_summary = explain_predictions(model, features, pushgateway_url)
    
    # Save results if requested
    if output_path:
        result_df.write_parquet(output_path)
        print(f"✅ Results saved to {output_path}")
    
    return result_df, shap_summary

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference with SHAP explanations")
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    parser.add_argument("--feature-path", required=True, help="Path to features parquet")
    parser.add_argument("--pushgateway-url", help="Prometheus Pushgateway URL")
    parser.add_argument("--output-path", help="Output path for predictions")
    
    args = parser.parse_args()
    
    run_inference_with_explanations(
        model_path=args.model_path,
        feature_path=args.feature_path,
        pushgateway_url=args.pushgateway_url,
        output_path=args.output_path
    )