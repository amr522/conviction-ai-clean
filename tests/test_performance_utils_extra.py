import polars as pl
import pytest

from src.utils.performance_utils import (enhance_gamma_detection,
                                         optimize_signal_generation)


def make_df():
    """Simple DataFrame with known values."""
    return pl.DataFrame(
        {"opt30_volume": [10, 20, 30, 40, 50], "opt30_net_gamma": [1, 2, 3, 4, 5]}
    )


def test_optimize_signal_generation():
    df = make_df()
    out = optimize_signal_generation(df, window_size=3)

    # rolling_vol_mean for [10,20,30] is 20
    assert out.select("rolling_vol_mean").item(2, 0) == pytest.approx(
        (10 + 20 + 30) / 3
    )

    # Check all expected columns are present
    expected_cols = [
        "rolling_vol_mean",
        "rolling_vol_std",
        "rolling_gamma_mean",
        "rolling_gamma_std",
    ]
    for col in expected_cols:
        assert col in out.columns


def test_enhance_gamma_detection():
    df = optimize_signal_generation(make_df(), window_size=3)
    out = enhance_gamma_detection(df, multiplier=1.0)

    # At index 3: gamma=4, rolling mean of [2,3,4]=3, so 4>3*1.0 = True
    assert out.select("gamma_squeeze_enhanced").item(3, 0) is True

    # At index 2: gamma=3, rolling mean of [1,2,3]=2, so 3>2*1.0 = True
    assert out.select("gamma_squeeze_enhanced").item(2, 0) is True


def test_enhance_gamma_detection_high_multiplier():
    df = optimize_signal_generation(make_df(), window_size=3)
    out = enhance_gamma_detection(df, multiplier=5.0)

    # With high multiplier, should have fewer/no squeezes
    squeeze_count = out.select("gamma_squeeze_enhanced").to_series().sum()
    assert squeeze_count < 3  # Should be less than with multiplier=1.0
