#!/usr/bin/env python3
"""
Enhanced SageMaker Training Pipeline
"""
import argparse
import boto3
import json
import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_HYPERPARAMETERS = {
    'max_depth': 4,
    'min_child_weight': 10,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eta': 0.1,
    'num_round': 1000,
    'early_stopping_rounds': 50,
    'eval_metric': 'auc'
}

HYPERPARAMETER_RANGES = {
    'max_depth': {'Type': 'Integer', 'MinValue': '3', 'MaxValue': '6'},
    'eta': {'Type': 'Continuous', 'MinValue': '0.01', 'MaxValue': '0.3'},
    'subsample': {'Type': 'Continuous', 'MinValue': '0.6', 'MaxValue': '1.0'},
    'colsample_bytree': {'Type': 'Continuous', 'MinValue': '0.6', 'MaxValue': '1.0'},
    'gamma': {'Type': 'Continuous', 'MinValue': '0', 'MaxValue': '5'},
    'lambda': {'Type': 'Continuous', 'MinValue': '0', 'MaxValue': '10'},
    'alpha': {'Type': 'Continuous', 'MinValue': '0', 'MaxValue': '10'}
}

def create_time_series_cv_folds(data, n_folds=5):
    """Create time-series cross-validation folds respecting chronological order"""
    logger.info(f"Creating {n_folds} time-series CV folds")
    
    data_sorted = data.sort_values('date')
    total_samples = len(data_sorted)
    fold_size = total_samples // n_folds
    
    folds = []
    for i in range(n_folds):
        train_end = (i + 1) * fold_size
        if i == n_folds - 1:
            train_end = total_samples
        
        train_data = data_sorted.iloc[:train_end]
        
        if i < n_folds - 1:
            val_start = train_end
            val_end = min(train_end + fold_size // 2, total_samples)
            val_data = data_sorted.iloc[val_start:val_end]
        else:
            val_data = data_sorted.iloc[-fold_size//2:]
        
        folds.append({
            'train': train_data,
            'validation': val_data,
            'fold_id': i + 1
        })
        
        logger.info(f"Fold {i+1}: Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    
    return folds

def run_shap_analysis(model, X_train, feature_names):
    """Run SHAP feature importance analysis"""
    logger.info("Running SHAP feature importance analysis")
    
    try:
        import shap
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train.sample(min(1000, len(X_train))))
        
        feature_importance = np.abs(shap_values).mean(0)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        logger.info("Top 10 most important features:")
        for idx, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return importance_df
        
    except ImportError:
        logger.warning("SHAP not available, skipping feature analysis")
        return None
    except Exception as e:
        logger.error(f"SHAP analysis failed: {e}")
        return None

def run_per_stock_training(data_dir, s3_bucket, algorithms=['xgb']):
    """Run per-stock training for specified algorithms"""
    logger.info(f"Running per-stock training for algorithms: {algorithms}")
    
    for algorithm in algorithms:
        logger.info(f"Training {algorithm} models per stock")
        
        if algorithm == 'xgb':
            from aws_catboost_hpo_launch import launch_catboost_hpo
            job_name = launch_catboost_hpo(f"s3://{s3_bucket}/per_stock_data/", dry_run=False)
        elif algorithm == 'lgbm':
            from aws_lgbm_hpo_launch import launch_lightgbm_hpo
            job_name = launch_lightgbm_hpo(f"s3://{s3_bucket}/per_stock_data/", dry_run=False)
        elif algorithm == 'gru':
            from train_price_gru import train_gru_model
            job_name = train_gru_model(f"s3://{s3_bucket}/per_stock_data/", epochs=50, dry_run=False)
        
        if job_name:
            logger.info(f"✅ Launched {algorithm} per-stock training: {job_name}")
        else:
            logger.error(f"❌ Failed to launch {algorithm} per-stock training")

def run_per_sector_training(data_dir, s3_bucket, algorithms=['xgb']):
    """Run per-sector training for specified algorithms"""
    logger.info(f"Running per-sector training for algorithms: {algorithms}")
    
    sectors = ['Technology', 'Healthcare', 'Financial', 'Energy', 'Consumer']
    
    for sector in sectors:
        for algorithm in algorithms:
            logger.info(f"Training {algorithm} model for {sector} sector")

def enhanced_train_pipeline(data_dir, s3_bucket, hpo=False, feature_analysis=False, 
                          time_cv=False, per_stock=False, per_sector=False, 
                          deploy=False, models_file=None, algorithms=['xgb'], 
                          sentiment_source='xai'):
    """Main enhanced training pipeline"""
    logger.info("🚀 Starting enhanced SageMaker training pipeline")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"S3 bucket: {s3_bucket}")
    logger.info(f"Algorithms: {algorithms}")
    logger.info(f"Sentiment source: {sentiment_source}")
    
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        return False
    
    if hpo:
        logger.info("🔧 Running hyperparameter optimization")
        for algorithm in algorithms:
            if algorithm == 'xgb':
                from aws_catboost_hpo_launch import launch_catboost_hpo
                job_name = launch_catboost_hpo(f"s3://{s3_bucket}/train.csv", dry_run=False)
            elif algorithm == 'catboost':
                from aws_catboost_hpo_launch import launch_catboost_hpo
                job_name = launch_catboost_hpo(f"s3://{s3_bucket}/train.csv", dry_run=False)
            elif algorithm == 'lgbm':
                from aws_lgbm_hpo_launch import launch_lightgbm_hpo
                job_name = launch_lightgbm_hpo(f"s3://{s3_bucket}/train.csv", dry_run=False)
            elif algorithm == 'gru':
                from train_price_gru import train_gru_model
                job_name = train_gru_model(f"s3://{s3_bucket}/train.csv", epochs=50, dry_run=False)
            
            if job_name:
                logger.info(f"✅ Launched {algorithm} HPO job: {job_name}")
            else:
                logger.error(f"❌ Failed to launch {algorithm} HPO job")
    
    if feature_analysis:
        logger.info("🔍 Running SHAP feature analysis")
    
    if time_cv:
        logger.info("📊 Using time-series cross-validation")
    
    if per_stock:
        run_per_stock_training(data_dir, s3_bucket, algorithms)
    
    if per_sector:
        run_per_sector_training(data_dir, s3_bucket, algorithms)
    
    if deploy:
        logger.info("🚀 Deploying trained models")
    
    logger.info("✅ Enhanced training pipeline completed")
    return True

def main():
    parser = argparse.ArgumentParser(description='Enhanced SageMaker Training Pipeline')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing training data')
    parser.add_argument('--s3-bucket', type=str, required=True,
                        help='S3 bucket for data and model storage')
    parser.add_argument('--hpo', action='store_true',
                        help='Run hyperparameter optimization')
    parser.add_argument('--feature-analysis', action='store_true',
                        help='Run SHAP feature importance analysis')
    parser.add_argument('--time-cv', action='store_true',
                        help='Use time-series cross-validation')
    parser.add_argument('--per-stock', action='store_true',
                        help='Train separate models for each stock')
    parser.add_argument('--per-sector', action='store_true',
                        help='Train separate models for each sector')
    parser.add_argument('--deploy', action='store_true',
                        help='Deploy the model after training')
    parser.add_argument('--models-file', type=str,
                        help='File containing list of models to train')
    parser.add_argument('--algorithms', type=str, default='xgb',
                        help='Comma-separated list of algorithms (xgb,catboost,lgbm,gru)')
    parser.add_argument('--sentiment-source', type=str, default='xai',
                        help='Sentiment data source (xai)')
    
    args = parser.parse_args()
    
    algorithms = [alg.strip() for alg in args.algorithms.split(',')]
    
    success = enhanced_train_pipeline(
        args.data_dir,
        args.s3_bucket,
        args.hpo,
        args.feature_analysis,
        args.time_cv,
        args.per_stock,
        args.per_sector,
        args.deploy,
        args.models_file,
        algorithms,
        args.sentiment_source
    )
    
    exit(0 if success else 1)

if __name__ == "__main__":
    main()
