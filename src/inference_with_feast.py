#!/usr/bin/env python3
"""
Model inference using Feast feature store
"""
import os
import logging
import pickle
from typing import List, Dict, Optional
import pandas as pd
from feast_materialize import get_online_features, get_feature_store

logger = logging.getLogger(__name__)

def load_model(model_path: str):
    """Load trained model from pickle file"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def get_inference_features(
    tickers: List[str],
    feature_views: List[str] = None,
    repo_path: str = "feature_repo"
) -> Optional[pd.DataFrame]:
    """
    Get features for inference from Feast online store
    
    Args:
        tickers: List of ticker symbols
        feature_views: List of feature views to use
        repo_path: Path to feature repository
        
    Returns:
        DataFrame with features or None if failed
    """
    try:
        # Default feature views if not specified
        if feature_views is None:
            feature_views = ["stocks_30min", "options_30min", "stocks_daily", "options_daily"]
        
        # Build feature names
        feature_names = []
        for fv in feature_views:
            if fv == "stocks_30min":
                feature_names.extend([
                    "stocks_30min:open", "stocks_30min:high", "stocks_30min:low",
                    "stocks_30min:close", "stocks_30min:volume", "stocks_30min:returns"
                ])
            elif fv == "options_30min":
                feature_names.extend([
                    "options_30min:opt30_close", "options_30min:opt30_volume",
                    "options_30min:opt30_call_flow", "options_30min:opt30_put_flow",
                    "options_30min:opt30_gamma_squeeze", "options_30min:opt30_implied_volatility"
                ])
            elif fv == "stocks_daily":
                feature_names.extend([
                    "stocks_daily:close", "stocks_daily:volume", "stocks_daily:returns",
                    "stocks_daily:volatility_30d", "stocks_daily:rsi_14"
                ])
            elif fv == "options_daily":
                feature_names.extend([
                    "options_daily:optd_close", "options_daily:optd_volume",
                    "options_daily:optd_iv30", "options_daily:optd_vrp_30d",
                    "options_daily:optd_vol_spike"
                ])
        
        # Create entity rows
        entity_rows = [{"ticker": ticker} for ticker in tickers]
        
        # Get features from online store
        features_dict = get_online_features(
            entity_rows=entity_rows,
            feature_names=feature_names,
            repo_path=repo_path
        )
        
        if features_dict is None:
            return None
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_dict)
        
        logger.info(f"Retrieved features for {len(tickers)} tickers: {features_df.shape}")
        return features_df
        
    except Exception as e:
        logger.error(f"Failed to get inference features: {str(e)}")
        return None

def run_inference(
    tickers: List[str],
    model_path: str,
    feature_views: List[str] = None,
    repo_path: str = "feature_repo"
) -> Optional[Dict]:
    """
    Run model inference using Feast features
    
    Args:
        tickers: List of ticker symbols
        model_path: Path to trained model
        feature_views: List of feature views to use
        repo_path: Path to feature repository
        
    Returns:
        Dictionary with predictions or None if failed
    """
    try:
        # Load model
        model = load_model(model_path)
        
        # Get features
        features_df = get_inference_features(tickers, feature_views, repo_path)
        if features_df is None:
            logger.error("Failed to retrieve features for inference")
            return None
        
        # Prepare features for model (remove ticker column if present)
        feature_columns = [col for col in features_df.columns if col != "ticker"]
        X = features_df[feature_columns].fillna(0)  # Handle missing values
        
        # Run inference
        predictions = model.predict(X)
        
        # Create results dictionary
        results = {
            "tickers": tickers,
            "predictions": predictions.tolist(),
            "features_used": feature_columns,
            "model_path": model_path
        }
        
        logger.info(f"Inference completed for {len(tickers)} tickers")
        return results
        
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        return None

def batch_inference(
    input_file: str,
    model_path: str,
    output_file: str,
    ticker_column: str = "ticker",
    feature_views: List[str] = None,
    repo_path: str = "feature_repo"
) -> bool:
    """
    Run batch inference on a file of tickers
    
    Args:
        input_file: Path to input CSV/Parquet with tickers
        model_path: Path to trained model
        output_file: Path to save predictions
        ticker_column: Name of ticker column
        feature_views: List of feature views to use
        repo_path: Path to feature repository
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load input data
        if input_file.endswith('.parquet'):
            input_df = pd.read_parquet(input_file)
        else:
            input_df = pd.read_csv(input_file)
        
        tickers = input_df[ticker_column].unique().tolist()
        logger.info(f"Running batch inference for {len(tickers)} unique tickers")
        
        # Run inference
        results = run_inference(tickers, model_path, feature_views, repo_path)
        if results is None:
            return False
        
        # Create output DataFrame
        output_df = pd.DataFrame({
            "ticker": results["tickers"],
            "prediction": results["predictions"]
        })
        
        # Merge with original data if needed
        if len(input_df) > len(tickers):
            output_df = input_df.merge(output_df, on=ticker_column, how="left")
        
        # Save results
        if output_file.endswith('.parquet'):
            output_df.to_parquet(output_file, index=False)
        else:
            output_df.to_csv(output_file, index=False)
        
        logger.info(f"Batch inference results saved to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Batch inference failed: {str(e)}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Model inference with Feast features")
    parser.add_argument("--action", choices=["single", "batch"], required=True)
    parser.add_argument("--tickers", nargs="+", help="Ticker symbols for single inference")
    parser.add_argument("--input-file", help="Input file for batch inference")
    parser.add_argument("--output-file", help="Output file for batch inference")
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    parser.add_argument("--feature-views", nargs="+", help="Feature views to use")
    parser.add_argument("--repo-path", default="feature_repo", help="Feature repository path")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    if args.action == "single":
        if not args.tickers:
            print("❌ --tickers required for single inference")
            exit(1)
        
        results = run_inference(
            tickers=args.tickers,
            model_path=args.model_path,
            feature_views=args.feature_views,
            repo_path=args.repo_path
        )
        
        if results:
            print("✅ Inference completed:")
            for ticker, pred in zip(results["tickers"], results["predictions"]):
                print(f"  {ticker}: {pred:.4f}")
        else:
            print("❌ Inference failed")
            exit(1)
    
    elif args.action == "batch":
        if not args.input_file or not args.output_file:
            print("❌ --input-file and --output-file required for batch inference")
            exit(1)
        
        success = batch_inference(
            input_file=args.input_file,
            model_path=args.model_path,
            output_file=args.output_file,
            feature_views=args.feature_views,
            repo_path=args.repo_path
        )
        
        if success:
            print(f"✅ Batch inference completed: {args.output_file}")
        else:
            print("❌ Batch inference failed")
            exit(1)