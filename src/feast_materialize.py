#!/usr/bin/env python3
"""
Feast feature materialization for online serving
"""
import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional

from feast import FeatureStore

logger = logging.getLogger(__name__)


def get_feature_store(repo_path: str = "feature_repo") -> FeatureStore:
    """
    Get Feast feature store instance

    Args:
        repo_path: Path to feature repository

    Returns:
        FeatureStore instance
    """
    try:
        fs = FeatureStore(repo_path=repo_path)
        logger.info(f"Connected to Feast feature store at {repo_path}")
        return fs
    except Exception as e:
        logger.error(f"Failed to connect to feature store: {str(e)}")
        raise


def materialize_features(
    start_date: str,
    end_date: Optional[str] = None,
    feature_views: Optional[List[str]] = None,
    repo_path: str = "feature_repo",
) -> bool:
    """
    Materialize features to online store

    Args:
        start_date: Start date for materialization (YYYY-MM-DD)
        end_date: End date for materialization (defaults to today)
        feature_views: List of feature views to materialize (defaults to all)
        repo_path: Path to feature repository

    Returns:
        True if successful, False otherwise
    """
    try:
        fs = get_feature_store(repo_path)

        # Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        else:
            end_dt = datetime.now()

        logger.info(f"Materializing features from {start_dt} to {end_dt}")

        # Materialize features
        if feature_views:
            # Materialize specific feature views
            for fv_name in feature_views:
                logger.info(f"Materializing feature view: {fv_name}")
                fs.materialize(
                    start_date=start_dt, end_date=end_dt, feature_views=[fv_name]
                )
        else:
            # Materialize all feature views
            logger.info("Materializing all feature views")
            fs.materialize(start_date=start_dt, end_date=end_dt)

        logger.info("Feature materialization completed successfully")
        return True

    except Exception as e:
        logger.error(f"Feature materialization failed: {str(e)}")
        return False


def get_online_features(
    entity_rows: List[dict], feature_names: List[str], repo_path: str = "feature_repo"
) -> Optional[dict]:
    """
    Retrieve features from online store for inference

    Args:
        entity_rows: List of entity dictionaries (e.g., [{"ticker": "AAPL"}])
        feature_names: List of feature names (e.g., ["stocks_30min:close"])
        repo_path: Path to feature repository

    Returns:
        Feature dictionary or None if failed
    """
    try:
        fs = get_feature_store(repo_path)

        logger.info(f"Fetching online features for {len(entity_rows)} entities")

        # Get online features
        feature_vector = fs.get_online_features(
            entity_rows=entity_rows, features=feature_names
        )

        # Convert to dictionary
        features_dict = feature_vector.to_dict()

        logger.info(f"Retrieved {len(feature_names)} features successfully")
        return features_dict

    except Exception as e:
        logger.error(f"Failed to get online features: {str(e)}")
        return None


def get_historical_features(
    entity_df_path: str, feature_names: List[str], repo_path: str = "feature_repo"
) -> Optional[str]:
    """
    Get historical features for training

    Args:
        entity_df_path: Path to entity DataFrame (Parquet)
        feature_names: List of feature names
        repo_path: Path to feature repository

    Returns:
        Path to output file or None if failed
    """
    try:
        import pandas as pd

        fs = get_feature_store(repo_path)

        # Load entity DataFrame
        entity_df = pd.read_parquet(entity_df_path)
        logger.info(f"Loaded entity DataFrame with {len(entity_df)} rows")

        # Get historical features
        training_df = fs.get_historical_features(
            entity_df=entity_df, features=feature_names
        ).to_df()

        # Save result
        output_path = entity_df_path.replace(".parquet", "_with_features.parquet")
        training_df.to_parquet(output_path)

        logger.info(f"Historical features saved to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to get historical features: {str(e)}")
        return None


def list_feature_views(repo_path: str = "feature_repo") -> List[str]:
    """
    List all available feature views

    Args:
        repo_path: Path to feature repository

    Returns:
        List of feature view names
    """
    try:
        fs = get_feature_store(repo_path)
        feature_views = fs.list_feature_views()

        fv_names = [fv.name for fv in feature_views]
        logger.info(f"Found {len(fv_names)} feature views: {fv_names}")

        return fv_names

    except Exception as e:
        logger.error(f"Failed to list feature views: {str(e)}")
        return []


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Feast feature materialization")
    parser.add_argument(
        "--action",
        choices=["materialize", "list", "get-online"],
        required=True,
        help="Action to perform",
    )
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--feature-views", nargs="+", help="Feature views to materialize"
    )
    parser.add_argument("--ticker", help="Ticker for online features")
    parser.add_argument(
        "--features", nargs="+", help="Feature names for online retrieval"
    )
    parser.add_argument(
        "--repo-path", default="feature_repo", help="Feature repository path"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    if args.action == "materialize":
        if not args.start_date:
            print("❌ --start-date required for materialization")
            exit(1)

        success = materialize_features(
            start_date=args.start_date,
            end_date=args.end_date,
            feature_views=args.feature_views,
            repo_path=args.repo_path,
        )

        if success:
            print("✅ Feature materialization completed")
        else:
            print("❌ Feature materialization failed")
            exit(1)

    elif args.action == "list":
        feature_views = list_feature_views(args.repo_path)
        print("Available feature views:")
        for fv in feature_views:
            print(f"  - {fv}")

    elif args.action == "get-online":
        if not args.ticker or not args.features:
            print("❌ --ticker and --features required for online retrieval")
            exit(1)

        features = get_online_features(
            entity_rows=[{"ticker": args.ticker}],
            feature_names=args.features,
            repo_path=args.repo_path,
        )

        if features:
            print(f"✅ Retrieved features for {args.ticker}:")
            for key, value in features.items():
                print(f"  {key}: {value}")
        else:
            print("❌ Failed to retrieve online features")
            exit(1)
