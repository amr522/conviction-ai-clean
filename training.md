# Training Documentation

This document provides comprehensive information about the training pipeline, model configurations, and feature engineering for the conviction-ai trading system.

## Overview

The training pipeline supports multiple algorithms including XGBoost, CatBoost, LightGBM, and GRU models with comprehensive hyperparameter optimization and feature engineering capabilities.

## Enhanced Training Pipeline

The enhanced SageMaker training pipeline provides:

- **Early Stopping**: Prevents overfitting with validation-based stopping
- **Hyperparameter Optimization**: Automated HPO with Optuna integration
- **SHAP Feature Analysis**: Feature importance and stability analysis
- **Time-Series Cross-Validation**: Leak-proof temporal validation
- **Twitter Sentiment Integration**: Real-time social media sentiment analysis

## Command Line Usage

### Basic Training
```bash
python xgboost_train.py --input-data s3://bucket/train.csv --model-dir /opt/ml/model
```

### Enhanced Training with HPO
```bash
./train_models_and_prepare_56.sh --enhanced --use-aws --hpo --feature-analysis --time-cv --per-stock --per-sector --deploy
```

### Training with Twitter Sentiment Features
```bash
python scripts/orchestrate_hpo_pipeline.py --algorithm xgboost --twitter-sentiment --include-sentiment --input-data-s3 s3://hpo-bucket-773934887314/sagemaker/train.csv
```

## Twitter Sentiment Integration

### Overview
The Twitter sentiment integration extends the existing news sentiment capabilities by adding real-time social media sentiment analysis. This provides additional alpha signals through social sentiment aggregation at multiple timeframes.

### Sentiment Features
The sentiment integration adds the following feature columns to the training data:

- **`sent_5m`**: 5-minute sentiment aggregation
- **`sent_10m`**: 10-minute sentiment aggregation  
- **`sent_60m`**: 60-minute sentiment aggregation
- **`sent_daily`**: Daily sentiment aggregation

These features complement the existing `news_sentiment` and `news_volume` columns in the feature schema.

### Enabling Sentiment Features
Sentiment ingestion is live and can be enabled via the `--twitter-sentiment` flag:

```bash
# Enable sentiment features in HPO pipeline
python scripts/orchestrate_hpo_pipeline.py --algorithm xgboost --twitter-sentiment

# Include sentiment features in training
python scripts/orchestrate_hpo_pipeline.py --algorithm xgboost --twitter-sentiment --include-sentiment
```

### Sentiment Data Pipeline
1. **Twitter Stream Ingestion**: Real-time tweet collection using Tweepy v2
2. **FinBERT Sentiment Scoring**: Financial domain-specific sentiment analysis
3. **Feature Engineering**: Multi-timeframe sentiment aggregation
4. **Pipeline Integration**: Seamless integration with existing HPO workflow

### Performance Impact
- **Target Uplift**: ≥ +0.02 AUC improvement vs baseline
- **Baseline AUC**: 0.9989 (AAPL reference)
- **Quality Threshold**: Minimum AUC ≥ 0.60 for all models

## Command Line Options

### Core Training Parameters
- `--algorithm`: Algorithm to use (xgboost, catboost, lightgbm, gru)
- `--input-data-s3`: S3 URI for training data
- `--model-dir`: Directory to save trained models
- `--hyperparameters`: JSON string of hyperparameters

### Enhanced Features
- `--early-stopping`: Enable early stopping (default: True)
- `--hpo`: Enable hyperparameter optimization
- `--feature-analysis`: Enable SHAP feature analysis
- `--time-cv`: Enable time-series cross-validation
- `--twitter-sentiment`: Enable Twitter sentiment features
- `--include-sentiment`: Include sentiment features in training

### HPO Configuration
- `--n-trials`: Number of HPO trials (default: 50)
- `--min-auc`: Minimum AUC threshold (default: 0.60)
- `--max-depth-range`: XGBoost max depth range
- `--learning-rate-range`: Learning rate optimization range

## Feature Engineering

### Base Features (67 columns)
- OHLCV price and volume data
- Technical indicators (RSI, MACD, Bollinger Bands)
- Market microstructure features
- News sentiment and volume

### Sentiment Features (4 additional columns)
- Multi-timeframe Twitter sentiment aggregation
- Volume-weighted sentiment scores
- Sentiment momentum indicators
- Cross-timeframe sentiment divergence

### Feature Validation
- Temporal integrity checks
- SHAP stability validation
- Feature importance thresholding
- Cross-validation consistency

## Model Training Process

### 1. Data Preparation
- Time-series train/validation/test splits
- Feature engineering and validation
- Sentiment data integration (if enabled)

### 2. Hyperparameter Optimization
- Optuna-based optimization
- Early stopping with patience
- AUC threshold enforcement (≥ 0.60)

### 3. Model Training
- Algorithm-specific training
- SHAP feature analysis
- Performance validation

### 4. Model Validation
- Out-of-time testing
- Leak detection tests
- Performance metrics logging

## Output Structure

```
/opt/ml/model/
├── model.pkl                    # Trained model artifact
├── hyperparameters.json         # Optimal hyperparameters
├── feature_importance.json      # SHAP feature importance
├── training_metrics.json        # Training performance metrics
├── validation_results.json      # Validation performance
└── sentiment_metadata.json      # Sentiment feature metadata (if enabled)
```

## Performance Monitoring

### Training Metrics
- Training/validation AUC curves
- Feature importance stability
- Early stopping decisions
- Hyperparameter convergence

### Sentiment Metrics (if enabled)
- Sentiment feature contribution
- Tweet volume correlation
- Sentiment-price relationship
- Multi-timeframe consistency

## Troubleshooting

### Common Issues
1. **Low AUC Performance**: Check feature engineering and hyperparameters
2. **Overfitting**: Increase regularization, enable early stopping
3. **Sentiment Integration**: Verify Twitter API credentials and data pipeline
4. **Memory Issues**: Reduce batch size or use distributed training

### Debugging Commands
```bash
# Check training logs
tail -f /opt/ml/output/failure

# Validate sentiment data
python -c "import pandas as pd; df = pd.read_csv('train.csv'); print(df.columns[df.columns.str.contains('sent_')])"

# Test sentiment pipeline
python scripts/orchestrate_hpo_pipeline.py --algorithm xgboost --twitter-sentiment --dry-run
```

## Recent Updates

### 2025-07-05: Twitter Sentiment Integration
- ✅ Added Twitter sentiment feature support
- ✅ Implemented multi-timeframe sentiment aggregation
- ✅ Integrated sentiment pipeline with HPO workflow
- ✅ Added sentiment-specific validation and monitoring

### 2025-07-03: Enhanced Training Pipeline
- ✅ Implemented leak-proof time-series cross-validation
- ✅ Added SHAP feature stability analysis
- ✅ Enhanced early stopping with AUC thresholds
- ✅ Improved hyperparameter optimization ranges
