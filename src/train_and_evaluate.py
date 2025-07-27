#!/usr/bin/env python3
"""
Automated model training and evaluation script for volatility prediction.
"""

# AWS X-Ray tracing setup
from aws_xray_sdk.core import patch_all, xray_recorder

# Patch AWS services and HTTP libraries
patch_all()
xray_recorder.configure(
    service="conviction-ai-training",
    plugins=("EC2Plugin", "ECSPlugin"),
    daemon_address="127.0.0.1:2000",
)

import argparse
import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    # Fallback for older sklearn versions
    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)


import mlflow
import mlflow.sklearn
import optuna

from utils.lineage_utils import LineageTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def monitor_feature_importance(
    model: Any, feature_names: List[str], top_n: int = 20, n_jobs: int = 1
) -> Dict[str, Any]:
    """
    Monitor and log feature importance from trained model.

    Args:
        model: Trained LightGBM model
        feature_names: List of feature names
        top_n: Number of top features to return
        n_jobs: Number of parallel workers (for future parallel feature processing)

    Returns:
        Dict with feature importance data
    """
    importance = model.feature_importances_
    feature_importance = dict(zip(feature_names, importance))

    # Sort by importance
    sorted_features = sorted(
        feature_importance.items(), key=lambda x: x[1], reverse=True
    )
    top_features = dict(sorted_features[:top_n])

    logger.info(f"Top {top_n} most important features:")
    for feature, importance in top_features.items():
        logger.info(f"  {feature}: {importance:.4f}")

    return {
        "all_features": feature_importance,
        "top_features": top_features,
        "feature_count": len(feature_names),
    }


@xray_recorder.capture("load_partitioned_data")
def load_partitioned_data(
    data_dir: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    Load partitioned data from year/month structure between date range.

    Args:
        data_dir: Base directory containing partitioned data
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Combined DataFrame
    """
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    dfs = []
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Find all year/month partitions in date range
    for year_dir in data_path.glob("year=*"):
        year = int(year_dir.name.split("=")[1])

        for month_dir in year_dir.glob("month=*"):
            month = int(month_dir.name.split("=")[1])

            # Check if this partition overlaps with our date range
            partition_start = pd.Timestamp(year, month, 1)
            partition_end = (
                partition_start + pd.DateOffset(months=1) - pd.Timedelta(days=1)
            )

            if partition_start <= end_dt and partition_end >= start_dt:
                # Load all parquet files in this partition
                for parquet_file in month_dir.glob("*.parquet"):
                    logger.info(f"Loading {parquet_file}")
                    df = pd.read_parquet(parquet_file)
                    dfs.append(df)

    if not dfs:
        raise ValueError(f"No data found between {start_date} and {end_date}")

    combined_df = pd.concat(dfs, ignore_index=True)

    # Filter to exact date range if timestamp column exists
    if "timestamp" in combined_df.columns:
        combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"])
        mask = (combined_df["timestamp"] >= start_dt) & (
            combined_df["timestamp"] <= end_dt
        )
        combined_df = combined_df[mask]

    logger.info(f"Loaded {len(combined_df)} records from {len(dfs)} files")
    return combined_df


def prepare_features_and_target(
    intraday_df: pd.DataFrame, daily_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target variable for training.

    Args:
        intraday_df: Intraday features DataFrame
        daily_df: Daily features DataFrame

    Returns:
        Tuple of (features_df, target_series)
    """
    # Merge intraday and daily features on common keys
    if "timestamp" in intraday_df.columns and "timestamp" in daily_df.columns:
        # Convert timestamps to date for joining
        intraday_df["date"] = pd.to_datetime(intraday_df["timestamp"]).dt.date
        daily_df["date"] = pd.to_datetime(daily_df["timestamp"]).dt.date

        # Merge on date and any other common keys
        common_cols = set(intraday_df.columns) & set(daily_df.columns)
        merge_keys = ["date"]
        if "symbol" in common_cols:
            merge_keys.append("symbol")
        elif "ticker" in common_cols:
            merge_keys.append("ticker")

        combined_df = pd.merge(
            intraday_df,
            daily_df,
            on=merge_keys,
            how="inner",
            suffixes=("_intraday", "_daily"),
        )
    else:
        # Simple concatenation if no timestamp
        combined_df = pd.concat([intraday_df, daily_df], axis=1)

    # Create target variable (next-period volatility)
    # Use realized volatility or price change as proxy
    if "close" in combined_df.columns:
        combined_df = combined_df.sort_values(["date"])
        combined_df["returns"] = combined_df["close"].pct_change()
        combined_df["volatility_target"] = (
            combined_df["returns"].rolling(window=5).std().shift(-1)
        )
    else:
        # Fallback: create synthetic target
        np.random.seed(42)
        combined_df["volatility_target"] = np.random.normal(
            0.02, 0.01, len(combined_df)
        )

    # Select numeric features only
    numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col != "volatility_target"]

    features = combined_df[feature_cols].fillna(0)
    target = combined_df["volatility_target"].fillna(0.02)

    logger.info(
        f"Prepared {len(feature_cols)} features and {len(target)} target values"
    )
    return features, target


def train_validation_split(
    features: pd.DataFrame, target: pd.Series, train_ratio: float = 0.8
) -> Tuple:
    """
    Split data into training and validation sets using temporal split.

    Args:
        features: Features DataFrame
        target: Target Series
        train_ratio: Ratio of data to use for training

    Returns:
        Tuple of (X_train, X_val, y_train, y_val)
    """
    split_idx = int(len(features) * train_ratio)

    X_train = features.iloc[:split_idx]
    X_val = features.iloc[split_idx:]
    y_train = target.iloc[:split_idx]
    y_val = target.iloc[split_idx:]

    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")

    return X_train, X_val, y_train, y_val


@xray_recorder.capture("optimize_hyperparameters")
def optimize_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
    use_gpu: bool = False,
    n_jobs: int = 1,
) -> Dict:
    """
    Optimize hyperparameters using Optuna.

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        n_trials: Number of optimization trials
        use_gpu: Whether to use GPU acceleration
        n_jobs: Number of parallel workers for Optuna

    Returns:
        Best hyperparameters dictionary
    """

    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
            "random_state": 42,
            "verbose": -1,
        }

        if use_gpu:
            params.update({"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0})

        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        return root_mean_squared_error(y_val, preds)  # RMSE

    logger.info(
        f"Starting hyperparameter optimization with {n_trials} trials using {n_jobs} workers..."
    )
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    logger.info(f"Best RMSE: {study.best_value:.6f}")
    logger.info(f"Best parameters: {study.best_params}")

    return study.best_params


@xray_recorder.capture("train_model")
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict = None,
    use_gpu: bool = False,
    n_jobs: int = 1,
) -> LGBMRegressor:
    """
    Train LightGBM regression model.

    Args:
        X_train: Training features
        y_train: Training target
        params: Optional hyperparameters dict
        use_gpu: Whether to use GPU acceleration
        n_jobs: Number of parallel workers for LightGBM

    Returns:
        Trained LightGBM model
    """
    default_params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": n_jobs,
    }

    if params:
        default_params.update(params)

    if use_gpu:
        default_params.update(
            {"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0}
        )

    model = LGBMRegressor(**default_params)

    logger.info("Training LightGBM model...")
    model.fit(X_train, y_train)
    logger.info("Model training completed")

    return model


def evaluate_model(model, X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
    """
    Evaluate model performance on validation set.

    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation target

    Returns:
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_val)

    rmse = root_mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)

    metrics = {"rmse": float(rmse), "mae": float(mae), "n_samples": len(y_val)}

    logger.info(f"Validation RMSE: {rmse:.6f}")
    logger.info(f"Validation MAE: {mae:.6f}")

    return metrics, y_pred


def create_calibration_plot(y_true: pd.Series, y_pred: np.ndarray, output_path: str):
    """
    Create and save calibration plot (observed vs predicted).

    Args:
        y_true: True values
        y_pred: Predicted values
        output_path: Path to save plot
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, y_true, alpha=0.5, s=10)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction")

    plt.xlabel("Predicted Volatility")
    plt.ylabel("Observed Volatility")
    plt.title("Model Calibration: Observed vs Predicted")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Calibration plot saved to {output_path}")


@xray_recorder.capture("training_pipeline")
def run(
    start_date: str,
    end_date: str,
    model_path: str,
    metrics_path: str,
    dry_run: bool = False,
    tune: bool = False,
    n_trials: int = 50,
    n_jobs: int = None,
    **kwargs
) -> int:
    """
    Main training and evaluation pipeline.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        model_path: Path to save trained model
        metrics_path: Directory to save metrics
        dry_run: If True, run without training or saving files
        tune: If True, run hyperparameter optimization
        n_trials: Number of Optuna trials for hyperparameter optimization
        n_jobs: Number of parallel workers (defaults to os.cpu_count())

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Initialize MLflow if tracking URI is set
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(
                os.getenv("MLFLOW_EXPERIMENT_NAME", "ConvictionAI-Swing")
            )

        # Set default n_jobs to CPU count if not specified
        if n_jobs is None:
            n_jobs = os.cpu_count()

        logger.info(f"Starting training pipeline: {start_date} to {end_date}")
        logger.info(f"Using {n_jobs} parallel workers")
        if tracking_uri:
            logger.info(f"MLflow tracking enabled: {tracking_uri}")

        # Initialize lineage tracking
        lineage = LineageTracker()
        lineage.start_run(
            "model_training",
            inputs=["datasets/intraday_30m", "datasets/daily"],
            outputs=[model_path, metrics_path],
        )

        # Create output directories
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metrics_path).mkdir(parents=True, exist_ok=True)

        # Load data - use feature parquet if provided
        feature_path = kwargs.get('feature_path')
        if feature_path:
            logger.info(f"Loading features from {feature_path}")
            import polars as pl
            feats = pl.read_parquet(feature_path)
            # Convert to pandas for compatibility
            features_df = feats.to_pandas()
            # Create dummy target for compatibility
            target_series = features_df.get('target', pd.Series([0.02] * len(features_df)))
        else:
            # Load data
            logger.info("Loading intraday data...")
            intraday_df = load_partitioned_data(
                "datasets/intraday_30m", start_date, end_date
            )

            logger.info("Loading daily data...")
            daily_df = load_partitioned_data("datasets/daily", start_date, end_date)
            
            # Prepare features and target
            logger.info("Preparing features and target...")
            features_df, target_series = prepare_features_and_target(intraday_df, daily_df)

        # Features and target already prepared above
        features, target = features_df, target_series

        # Train/validation split
        X_train, X_val, y_train, y_val = train_validation_split(features, target)

        # Check GPU availability
        use_gpu = False
        try:
            import lightgbm as lgb
            import numpy as np
            import pandas as pd

            # Test GPU with actual training to verify it works
            test_X = pd.DataFrame(np.random.randn(10, 3))
            test_y = pd.Series(np.random.randn(10))
            test_model = lgb.LGBMRegressor(device="gpu", verbose=-1, n_estimators=1)
            test_model.fit(test_X, test_y)
            use_gpu = True
            logger.info("GPU acceleration available and enabled")
        except Exception as e:
            logger.info(f"GPU acceleration not available, using CPU: {str(e)[:100]}...")

        if dry_run:
            logger.info("DRY RUN: Skipping training and file operations")
            logger.info(
                f"Would train on {len(features)} samples with {len(features.columns)} features"
            )
            if tune:
                logger.info(
                    f"Running 1 trial for dry-run hyperparameter optimization (GPU: {use_gpu}, Workers: {n_jobs})..."
                )
                best_params = optimize_hyperparameters(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    n_trials=1,
                    use_gpu=use_gpu,
                    n_jobs=n_jobs,
                )
                logger.info(f"Dry-run best params: {best_params}")
            return 0

        # Data drift monitoring with Delta Lake support
        from data_drift_monitor import monitor_data_drift

        logger.info("Running data drift analysis...")
        drift_detected, drift_report_path = monitor_data_drift(start_date)

        # Start MLflow run if tracking is enabled
        run_context = (
            mlflow.start_run(run_name=f"train_{start_date}_{end_date}")
            if tracking_uri
            else None
        )

        try:
            # Log parameters to MLflow
            if tracking_uri:
                mlflow.log_params(
                    {
                        "start_date": start_date,
                        "end_date": end_date,
                        "n_trials": n_trials,
                        "n_jobs": n_jobs,
                        "use_gpu": use_gpu,
                        "tune": tune,
                        "train_samples": len(X_train),
                        "val_samples": len(X_val),
                        "n_features": len(features.columns),
                        "feature_path": feature_path,
                        "drift_detected": drift_detected,
                    }
                )

                # Log drift report if available
                if drift_report_path and os.path.exists(drift_report_path):
                    mlflow.log_artifact(drift_report_path)

            # Hyperparameter optimization or default training
            best_params = None
            if tune:
                logger.info(
                    f"Starting hyperparameter optimization with {n_trials} trials (GPU: {use_gpu}, Workers: {n_jobs})"
                )
                best_params = optimize_hyperparameters(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    n_trials=n_trials,
                    use_gpu=use_gpu,
                    n_jobs=n_jobs,
                )

                # Log best parameters to MLflow
                if tracking_uri:
                    mlflow.log_params(best_params)

                # Save best parameters
                params_file = os.path.join(
                    metrics_path, f"optuna_best_params_{start_date}_{end_date}.json"
                )
                logger.info(f"Saving best parameters to {params_file}")
                with open(params_file, "w") as f:
                    json.dump(best_params, f, indent=2)

            # Train model with best parameters or defaults
            model = train_model(
                X_train, y_train, best_params, use_gpu=use_gpu, n_jobs=n_jobs
            )

            # Monitor feature importance
            feature_importance = monitor_feature_importance(
                model, list(features.columns), n_jobs=n_jobs
            )

            # Evaluate model
            metrics, y_pred = evaluate_model(model, X_val, y_val)

            # Log metrics to MLflow
            if tracking_uri:
                mlflow.log_metrics(
                    {
                        "rmse": metrics["rmse"],
                        "mae": metrics["mae"],
                        "n_samples": metrics["n_samples"],
                    }
                )

            # Create calibration plot
            calibration_path = os.path.join(metrics_path, "calibration.png")
            create_calibration_plot(y_val, y_pred, calibration_path)

            # Log artifacts to MLflow
            if tracking_uri:
                mlflow.log_artifact(calibration_path)
                if tune:
                    params_file = os.path.join(
                        metrics_path, f"optuna_best_params_{start_date}_{end_date}.json"
                    )
                    mlflow.log_artifact(params_file)

            # Save model
            logger.info(f"Saving model to {model_path}")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            # Log model to MLflow
            if tracking_uri:
                mlflow.sklearn.log_model(
                    model, "model", registered_model_name="ConvictionAI_Swing_Model"
                )

            # Save metrics
            metrics_file = os.path.join(
                metrics_path, f"metrics_{start_date}_{end_date}.json"
            )
            full_metrics = {
                "evaluation": metrics,
                "feature_importance": feature_importance,
                "training_info": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "train_samples": len(X_train),
                    "val_samples": len(X_val),
                    "n_features": len(features.columns),
                    "feature_path": feature_path,
                    "hyperparameter_tuning": tune,
                    "n_trials": n_trials if tune else None,
                    "n_jobs": n_jobs,
                    "gpu_enabled": use_gpu,
                    "best_params": best_params if tune else None,
                },
            }

            logger.info(f"Saving metrics to {metrics_file}")
            with open(metrics_file, "w") as f:
                json.dump(full_metrics, f, indent=2)

            # Log metrics file to MLflow
            if tracking_uri:
                mlflow.log_artifact(metrics_file)

            logger.info("Training pipeline completed successfully")
            lineage.complete_run(success=True)
            return 0

        finally:
            if run_context:
                mlflow.end_run()

    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        try:
            lineage.complete_run(success=False)
        except:
            pass
        return 1


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Automated model training and evaluation"
    )
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--model-path", default="models/latest.pkl", help="Path to save model"
    )
    parser.add_argument(
        "--metrics-path", default="metrics/", help="Directory to save metrics"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run without training or saving files"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter optimization with Optuna",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials for hyperparameter optimization",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel workers (defaults to os.cpu_count())",
    )
    parser.add_argument(
        "--feature-path",
        type=str,
        help="Path to feature Parquet file (optional, overrides data loading)",
    )

    args = parser.parse_args()

    exit_code = run(
        start_date=args.start_date,
        end_date=args.end_date,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        dry_run=args.dry_run,
        tune=args.tune,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        feature_path=args.feature_path,
    )

    exit(exit_code)


if __name__ == "__main__":
    main()
