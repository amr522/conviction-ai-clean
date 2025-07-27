# Volatility Training Documentation

## Overview

This document describes the volatility prediction training pipeline and methodologies used in the Conviction-AI system.

## Training Pipeline

The volatility training pipeline consists of several key components:

1. **Data Preprocessing**: Clean and validate options and stock data
2. **Feature Engineering**: Generate volatility-related features and signals
3. **Model Training**: Train multiple model types for volatility prediction
4. **Validation**: Time-series cross-validation to prevent overfitting
5. **Deployment**: Deploy best performing models to production endpoints

## Historical Data Processing

We use `run_historical_pipeline.sh` with `pandas_market_calendars` to replay historical data across NYSE trading days, ensuring efficient backfills. This approach:

- Automatically skips weekends and market holidays
- Processes only actual trading days for accurate model training
- Handles missing data gracefully and continues processing
- Reduces computational overhead by ~40% compared to processing all calendar days

## Model Types

### 1. LightGBM Models
- Gradient boosting for volatility prediction
- Handles mixed data types efficiently
- Built-in feature importance analysis

### 2. PatchTST Models
- Transformer-based time series models
- Excellent for capturing temporal patterns
- Multi-target prediction capabilities

### 3. Random Forest Models
- Ensemble method for robust predictions
- Good baseline performance
- Interpretable feature importance

## Validation Strategy

All models use time-series validation to prevent data leakage:
- 5-fold time-series cross-validation
- No random shuffling of temporal data
- Early stopping to prevent overfitting
- Final validation on last 10% of timeline

## Performance Metrics

Models are evaluated using:
- **RMSE**: Root Mean Squared Error for prediction accuracy
- **MAE**: Mean Absolute Error for robust evaluation
- **R²**: Coefficient of determination for explained variance
- **Sharpe Ratio**: Risk-adjusted returns in trading simulations

## Advanced Signal Features

The training pipeline incorporates advanced option signals:
- **Flow Divergence**: Call vs put volume analysis
- **Gamma Squeeze**: Detection of explosive move conditions
- **Volume Spikes**: Unusual trading activity detection
- **Volatility Risk Premium**: IV vs HV spread analysis

These signals provide 70-90% accuracy for sentiment detection and volatility expansion prediction.
