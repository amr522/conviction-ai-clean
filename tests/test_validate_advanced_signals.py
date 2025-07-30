import json
import os
import tempfile

import polars as pl
import pytest

from src.validate_advanced_signals import (check_gamma_signals,
                                           test_volume_spikes,
                                           validate_advanced_signals,
                                           validate_flow_divergence)


@pytest.fixture
def sample_df(tmp_path):
    """Construct a tiny DataFrame covering all cases."""
    df = pl.DataFrame(
        {
            "ticker": ["A", "A", "B", "B"],
            "timestamp": [1, 2, 1, 2],
            "opt30_net_gamma": [1.0, None, 2.0, 3.0],
            "opt30_flow_divergence": [5.0, -2.0, 3.0, -1.0],
            "opt30_mid_price": [10, 12, 20, 18],
            "opt30_volume": [100, 200, 300, 400],
            "opt30_vol_mean_5": [50, 50, 100, 100],
            "opt30_vol_std_5": [10, 10, 20, 20],
            "opt30_vol_spike": [True, False, True, False],
        }
    )
    file = tmp_path / "df.parquet"
    df.write_parquet(file)
    return str(file)


def test_gamma_coverage(sample_df):
    df = pl.read_parquet(sample_df)
    cov = check_gamma_signals(df)
    assert cov == pytest.approx(3 / 4)


def test_flow_accuracy(sample_df):
    df = pl.read_parquet(sample_df)
    acc = validate_flow_divergence(df)
    # A: flow positive [True,False], returns [True,False] → both correct
    # B: [True,False] → [True,False] → correct
    assert acc == pytest.approx(1.0)


def test_volume_spikes(sample_df):
    df = pl.read_parquet(sample_df)
    det = test_volume_spikes(df)
    # Both flagged spikes coincide with true_spike logic
    assert det == pytest.approx(1.0)


def test_validate_advanced_signals_pass(sample_df, tmp_path):
    out = tmp_path / "res.json"
    results = validate_advanced_signals(sample_df, 0.5)
    assert results["gamma_coverage"] > 0.0


def test_validate_advanced_signals_fail(sample_df):
    # threshold too high
    with pytest.raises(SystemExit):
        validate_advanced_signals(sample_df, 1.0)
