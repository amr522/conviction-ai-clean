# Enhanced SageMaker Training Pipeline

This document explains how to use the enhanced SageMaker training pipeline for stock prediction models.

## Overview

The enhanced training pipeline adds several advanced features to the standard SageMaker training:

1. **Early Stopping & Regularization**: Uses optimized XGBoost parameters with early stopping to prevent overfitting.
2. **Hyperparameter Optimization**: Employs Bayesian optimization to find the best hyperparameters.
3. **SHAP Feature Analysis**: Identifies important features and detects potential data leakage.
4. **Time-Series Cross-Validation**: Implements walk-forward validation for more robust performance evaluation.
5. **Per-Stock & Per-Sector Models**: Trains specialized models for individual stocks or sectors.
6. **Twitter Sentiment Integration**: Real-time sentiment analysis from Twitter/X with multi-timeframe aggregation.
7. **Detailed Logging**: Provides comprehensive logs and metrics at each step.

## Usage

Run the pipeline using the `train_models_and_prepare_56.sh` script with various flags:

```bash
# Basic usage with enhanced pipeline
./train_models_and_prepare_56.sh --enhanced --use-aws

# Full usage with all options
./train_models_and_prepare_56.sh --enhanced --use-aws --hpo --feature-analysis --time-cv --per-stock --per-sector --deploy

# Usage with Twitter sentiment features
python scripts/orchestrate_hpo_pipeline.py --algorithm xgboost --twitter-sentiment --include-sentiment --input-data-s3 s3://conviction-ai-data/sagemaker/train.csv
```

### Command Line Options

- `--enhanced`: Use the enhanced training pipeline (required for the features below)
- `--use-aws`: Push training to AWS SageMaker
- `--hpo`: Run hyperparameter optimization
- `--feature-analysis`: Run SHAP feature importance analysis
- `--time-cv`: Use time-series cross-validation
- `--per-stock`: Train separate models for each stock
- `--per-sector`: Train separate models for each sector
- `--twitter-sentiment`: Include Twitter sentiment features in training
- `--include-sentiment`: Include sentiment features in model training
- `--deploy`: Deploy the model after training

### Twitter Sentiment Features

The pipeline now supports real-time Twitter sentiment analysis with the following new features:

- `sent_5m`: 5-minute sentiment aggregation (volume-weighted)
- `sent_10m`: 10-minute sentiment aggregation (volume-weighted)  
- `sent_60m`: 60-minute sentiment aggregation (volume-weighted)
- `sent_daily`: Daily sentiment aggregation (volume-weighted)

These features are automatically generated from Twitter/X streaming data using FinBERT sentiment analysis and integrated into the feature matrix for enhanced prediction accuracy.

### Environment Variables

You can also use environment variables instead of command-line flags:

```bash
ENHANCED=1 USE_AWS=1 HPO=1 FEATURE_ANALYSIS=1 ./train_models_and_prepare_56.sh
```

## Directory Structure

- `data/base_model_outputs/11_models/`: Contains the 11 base models
- `data/sagemaker/56_stocks/`: SageMaker input data
- `data/analysis/`: Feature importance analysis and metrics
- `data/per_stock/`: Per-stock model training data and results
- `data/per_sector/`: Per-sector model training data and results
- `data/time_series_cv/`: Time-series cross-validation data and results
- `logs/`: Log files
- `pipeline_logs/`: Additional pipeline logs

## Advanced Configuration

The `enhanced_train_sagemaker.py` script can be run directly with additional options:

```bash
python enhanced_train_sagemaker.py --data-dir data/sagemaker/56_stocks \
                                  --s3-bucket your-bucket-name \
                                  --hpo \
                                  --feature-analysis \
                                  --time-cv \
                                  --max-jobs 20 \
                                  --max-parallel-jobs 5
```

## Default Hyperparameters

The enhanced pipeline uses the following improved default hyperparameters:

- `max_depth`: 4
- `min_child_weight`: 10
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `eta`: 0.1
- `num_round`: 1000
- `early_stopping_rounds`: 50
- `eval_metric`: auc

During hyperparameter optimization, the following ranges are used:

- `max_depth`: 3-6
- `eta`: 0.01-0.3
- `subsample`: 0.6-1.0
- `colsample_bytree`: 0.6-1.0
- `gamma`: 0-5
- `lambda`: 0-10
- `alpha`: 0-10
