#!/usr/bin/env python3
"""
Feature store utilities for FastAPI inference service
"""
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def get_feature_store():
    """Get Feast feature store instance"""
    try:
        from feast import FeatureStore

        repo_path = os.getenv("FEAST_REPO_PATH", "feature_repo")
        fs = FeatureStore(repo_path=repo_path)
        return fs
    except Exception as e:
        logger.error(f"Failed to initialize feature store: {str(e)}")
        return None


def fetch_features(
    entity: str, value: str, timestamp: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch features for a given entity from the feature store

    Args:
        entity: Entity name (e.g., 'ticker')
        value: Entity value (e.g., 'AAPL')
        timestamp: Optional timestamp for point-in-time features

    Returns:
        DataFrame with features
    """
    try:
        fs = get_feature_store()
        if fs is None:
            return pd.DataFrame()

        # Prepare entity rows
        entity_rows = [{entity: value}]
        if timestamp:
            entity_rows[0]["event_timestamp"] = timestamp

        # Get all available features
        feature_views = fs.list_feature_views()
        feature_names = []

        for fv in feature_views:
            for feature in fv.features:
                feature_names.append(f"{fv.name}:{feature.name}")

        # Fetch online features
        feature_vector = fs.get_online_features(
            features=feature_names, entity_rows=entity_rows
        )

        return feature_vector.to_df()

    except Exception as e:
        logger.error(f"Failed to fetch features: {str(e)}")
        return pd.DataFrame()


def get_latest_features(ticker: str) -> Dict:
    """
    Get latest features for a ticker from the feature store

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary of features
    """
    try:
        from feast_materialize import get_online_features

        # Define comprehensive feature set
        feature_names = [
            # Stock 30-minute features
            "stocks_30min:open",
            "stocks_30min:high",
            "stocks_30min:low",
            "stocks_30min:close",
            "stocks_30min:volume",
            "stocks_30min:returns",
            "stocks_30min:volatility",
            # Options 30-minute features
            "options_30min:opt30_open",
            "options_30min:opt30_high",
            "options_30min:opt30_low",
            "options_30min:opt30_close",
            "options_30min:opt30_volume",
            "options_30min:opt30_call_flow",
            "options_30min:opt30_put_flow",
            "options_30min:opt30_flow_divergence",
            "options_30min:opt30_gamma",
            "options_30min:opt30_net_gamma",
            "options_30min:opt30_gamma_squeeze",
            "options_30min:opt30_implied_volatility",
            "options_30min:opt30_delta",
            "options_30min:opt30_moneyness",
            # Stock daily features
            "stocks_daily:open",
            "stocks_daily:high",
            "stocks_daily:low",
            "stocks_daily:close",
            "stocks_daily:volume",
            "stocks_daily:returns",
            "stocks_daily:volatility_30d",
            "stocks_daily:sma_20",
            "stocks_daily:rsi_14",
            # Options daily features
            "options_daily:optd_close",
            "options_daily:optd_volume",
            "options_daily:optd_moneyness",
            "options_daily:optd_iv30",
            "options_daily:optd_hv30",
            "options_daily:optd_vrp_30d",
            "options_daily:optd_iv_percentile",
            "options_daily:optd_vol_spike",
            "options_daily:optd_put_call_ratio",
        ]

        # Fetch features
        features_dict = get_online_features(
            entity_rows=[{"ticker": ticker}], feature_names=feature_names
        )

        if features_dict and len(features_dict.get("ticker", [])) > 0:
            # Convert to flat dictionary, excluding ticker
            features = {}
            for key, values in features_dict.items():
                if key != "ticker" and len(values) > 0:
                    # Handle None values
                    value = values[0]
                    if value is not None:
                        features[key] = (
                            float(value) if isinstance(value, (int, float)) else value
                        )

            return features
        else:
            logger.warning(f"No features found for ticker {ticker}")
            return {}

    except Exception as e:
        logger.error(f"Failed to get latest features for {ticker}: {str(e)}")
        return {}


def validate_features(features: Dict, required_features: List[str]) -> Dict:
    """
    Validate and clean features for model input

    Args:
        features: Raw features dictionary
        required_features: List of required feature names

    Returns:
        Validated features dictionary
    """
    validated = {}

    for feature in required_features:
        if feature in features:
            value = features[feature]
            # Handle different data types
            if isinstance(value, (int, float)):
                validated[feature] = float(value)
            elif isinstance(value, bool):
                validated[feature] = float(value)
            else:
                # Default value for non-numeric features
                validated[feature] = 0.0
        else:
            # Default value for missing features
            validated[feature] = 0.0
            logger.warning(f"Missing feature {feature}, using default value 0.0")

    return validated


def get_feature_metadata() -> Dict:
    """
    Get metadata about available features

    Returns:
        Dictionary with feature metadata
    """
    try:
        fs = get_feature_store()
        if fs is None:
            return {}

        metadata = {"feature_views": [], "total_features": 0}

        feature_views = fs.list_feature_views()
        for fv in feature_views:
            fv_info = {
                "name": fv.name,
                "features": [f.name for f in fv.features],
                "entities": [e for e in fv.entities],
                "ttl_seconds": fv.ttl.total_seconds() if fv.ttl else None,
            }
            metadata["feature_views"].append(fv_info)
            metadata["total_features"] += len(fv.features)

        return metadata

    except Exception as e:
        logger.error(f"Failed to get feature metadata: {str(e)}")
        return {}


# Mock feature generation for testing when feature store is unavailable
def generate_mock_features(ticker: str) -> Dict:
    """
    Generate mock features for testing purposes

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary of mock features
    """
    import random

    import numpy as np

    # Set seed based on ticker for consistent mock data
    random.seed(hash(ticker) % 2**32)
    np.random.seed(hash(ticker) % 2**32)

    mock_features = {
        # Stock features
        "stocks_30min:close": random.uniform(100, 300),
        "stocks_30min:volume": random.randint(10000, 1000000),
        "stocks_30min:returns": random.uniform(-0.05, 0.05),
        "stocks_30min:volatility": random.uniform(0.1, 0.8),
        # Options features
        "options_30min:opt30_close": random.uniform(1, 50),
        "options_30min:opt30_volume": random.randint(100, 10000),
        "options_30min:opt30_gamma_squeeze": random.choice([True, False]),
        "options_30min:opt30_implied_volatility": random.uniform(0.15, 0.6),
        # Daily features
        "stocks_daily:close": random.uniform(100, 300),
        "stocks_daily:rsi_14": random.uniform(20, 80),
        "options_daily:optd_iv30": random.uniform(0.15, 0.6),
        "options_daily:optd_vrp_30d": random.uniform(-0.1, 0.1),
    }

    return mock_features
