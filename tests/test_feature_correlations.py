"""Tests for cross-feature correlation to detect redundant features."""

from itertools import combinations
from pathlib import Path

import polars as pl
import pytest


def load_feature_list(features_list_path: str) -> list:
    """Load expected feature names from markdown file."""
    features = []
    with open(features_list_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("##"):
                features.append(line)
    return features


@pytest.fixture
def feature_df():
    """Load feature dataframe for testing."""
    feature_path = "data/Parquet_data/features_test.parquet"
    if not Path(feature_path).exists():
        pytest.skip(f"Feature file not found: {feature_path}")
    return pl.read_parquet(feature_path)


@pytest.mark.parametrize(
    "f1,f2", combinations(load_feature_list("docs/features_list.md"), 2)
)
def test_low_correlation(feature_df, f1, f2):
    """Test that feature pairs have correlation below threshold."""
    # Skip if either feature is missing
    if f1 not in feature_df.columns or f2 not in feature_df.columns:
        pytest.skip(f"Features {f1} or {f2} not found in dataframe")

    # Compute Pearson correlation
    corr_result = feature_df.select([pl.corr(f1, f2)]).to_series()

    # Handle null correlation (constant features)
    if corr_result.is_null().any():
        pytest.skip(
            f"Cannot compute correlation between {f1} and {f2} (constant features)"
        )

    corr = corr_result[0]

    # Assert correlation is below threshold
    assert abs(corr) < 0.95, f"Features {f1} and {f2} too highly correlated: {corr:.3f}"


def test_feature_variance():
    """Test that features have sufficient variance."""
    feature_path = "data/Parquet_data/features_test.parquet"
    if not Path(feature_path).exists():
        pytest.skip(f"Feature file not found: {feature_path}")

    df = pl.read_parquet(feature_path)
    features = load_feature_list("docs/features_list.md")

    low_variance_features = []
    for feature in features:
        if feature in df.columns:
            variance = df.select(pl.var(feature)).to_series()[0]
            if variance is not None and variance < 1e-6:
                low_variance_features.append(f"{feature} (var={variance:.2e})")

    assert (
        not low_variance_features
    ), f"Features with low variance: {low_variance_features}"
