#!/usr/bin/env python3
"""
Test Great Expectations data quality validation
"""
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from validate_data_quality import (DataQualityError,
                                   generate_validation_report,
                                   validate_dataset)


class TestDataQuality:
    def setup_method(self):
        """Setup test data and temporary directories"""
        self.temp_dir = tempfile.mkdtemp()

        # Create valid test data
        self.valid_options_daily = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=100, freq="D"),
                "symbol": ["AAPL"] * 100,
                "optd_close": [150.0 + i for i in range(100)],
                "optd_volume": [1000 + i * 10 for i in range(100)],
                "optd_moneyness": [1.0 + i * 0.01 for i in range(100)],
            }
        )

        # Create invalid test data (with violations)
        self.invalid_options_daily = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=100, freq="D"),
                "symbol": ["AAPL"] * 100,
                "optd_close": [-10.0] * 100,  # Invalid negative prices
                "optd_volume": [-100] * 100,  # Invalid negative volume
                "optd_moneyness": [50.0] * 100,  # Invalid high moneyness
            }
        )

    @patch("validate_data_quality.get_data_context")
    def test_validate_dataset_success(self, mock_context):
        """Test successful dataset validation"""
        # Mock Great Expectations context and components
        mock_gx_context = MagicMock()
        mock_context.return_value = mock_gx_context

        # Mock expectation suite
        mock_suite = MagicMock()
        mock_gx_context.get_expectation_suite.return_value = mock_suite

        # Mock datasource
        mock_datasource = MagicMock()
        mock_gx_context.get_datasource.return_value = mock_datasource

        # Mock successful validation results
        mock_results = {
            "success": True,
            "run_results": {
                "test_run": {
                    "validation_result": {
                        "results": [
                            {
                                "success": True,
                                "expectation_config": {
                                    "expectation_type": "expect_column_to_exist",
                                    "kwargs": {"column": "timestamp"},
                                },
                            }
                        ]
                    }
                }
            },
        }

        # Mock checkpoint
        with patch("validate_data_quality.SimpleCheckpoint") as mock_checkpoint_class:
            mock_checkpoint = MagicMock()
            mock_checkpoint.run.return_value = mock_results
            mock_checkpoint_class.return_value = mock_checkpoint

            # Save test data
            test_file = os.path.join(self.temp_dir, "test_data.parquet")
            self.valid_options_daily.to_parquet(test_file)

            # Run validation
            result = validate_dataset(test_file, "options_daily", "2025-01-16")

            assert result["success"] is True
            assert result["dataset_type"] == "options_daily"
            assert result["date"] == "2025-01-16"

    @patch("validate_data_quality.get_data_context")
    def test_validate_dataset_failure(self, mock_context):
        """Test dataset validation failure"""
        # Mock Great Expectations context
        mock_gx_context = MagicMock()
        mock_context.return_value = mock_gx_context

        # Mock expectation suite
        mock_suite = MagicMock()
        mock_gx_context.get_expectation_suite.return_value = mock_suite

        # Mock datasource
        mock_datasource = MagicMock()
        mock_gx_context.get_datasource.return_value = mock_datasource

        # Mock failed validation results
        mock_results = {
            "success": False,
            "run_results": {
                "test_run": {
                    "validation_result": {
                        "results": [
                            {
                                "success": False,
                                "expectation_config": {
                                    "expectation_type": "expect_column_values_to_be_between",
                                    "kwargs": {
                                        "column": "optd_close",
                                        "min_value": 0.01,
                                    },
                                },
                                "result": {"observed_value": -10.0},
                            }
                        ]
                    }
                }
            },
        }

        # Mock checkpoint
        with patch("validate_data_quality.SimpleCheckpoint") as mock_checkpoint_class:
            mock_checkpoint = MagicMock()
            mock_checkpoint.run.return_value = mock_results
            mock_checkpoint_class.return_value = mock_checkpoint

            # Save test data
            test_file = os.path.join(self.temp_dir, "test_data.parquet")
            self.invalid_options_daily.to_parquet(test_file)

            # Run validation - should raise DataQualityError
            with pytest.raises(DataQualityError) as exc_info:
                validate_dataset(test_file, "options_daily", "2025-01-16")

            assert "Validation failed for options_daily" in str(exc_info.value)

    def test_generate_validation_report(self):
        """Test validation report generation"""
        # Mock validation results
        validation_results = {
            "success": True,
            "dataset_type": "options_daily",
            "date": "2025-01-16",
            "file_path": "/test/path.parquet",
            "results": {
                "run_results": {
                    "test_run": {
                        "validation_result": {
                            "results": [
                                {
                                    "success": True,
                                    "expectation_config": {
                                        "expectation_type": "expect_column_to_exist",
                                        "kwargs": {"column": "timestamp"},
                                    },
                                },
                                {
                                    "success": False,
                                    "expectation_config": {
                                        "expectation_type": "expect_column_values_to_be_between",
                                        "kwargs": {"column": "optd_close"},
                                    },
                                },
                            ]
                        }
                    }
                }
            },
        }

        # Generate report
        report_path = generate_validation_report(validation_results, self.temp_dir)

        # Check report was created
        assert os.path.exists(report_path)
        assert report_path.endswith("ge_validation_options_daily_2025-01-16.html")

        # Check report content
        with open(report_path, "r") as f:
            content = f.read()
            assert "Data Quality Report" in content
            assert "options_daily" in content
            assert "2025-01-16" in content
            assert "expect_column_to_exist" in content

    @patch("validate_data_quality.validate_dataset")
    def test_validate_pipeline_outputs(self, mock_validate):
        """Test validation of all pipeline outputs"""
        from validate_data_quality import validate_pipeline_outputs

        # Mock successful validation
        mock_validate.return_value = {
            "success": True,
            "dataset_type": "options_daily",
            "date": "2025-01-16",
            "file_path": "test.parquet",
        }

        # Create test files
        test_files = [
            "options_daily_clean.parquet",
            "options_30min_clean.parquet",
            "stocks_daily_clean.parquet",
            "stocks_30min_clean.parquet",
        ]

        for filename in test_files:
            filepath = os.path.join(self.temp_dir, filename)
            self.valid_options_daily.to_parquet(filepath)

        # Run validation
        results = validate_pipeline_outputs("2025-01-16", self.temp_dir, self.temp_dir)

        # Check results
        assert len(results) == 4
        assert all(r.get("success", False) for r in results.values())

        # Verify validate_dataset was called for each file
        assert mock_validate.call_count == 4


def test_data_quality_imports():
    """Test that data quality utilities import correctly"""
    try:
        from validate_data_quality import (DataQualityError,
                                           generate_validation_report,
                                           validate_dataset)

        assert callable(validate_dataset)
        assert issubclass(DataQualityError, Exception)
        assert callable(generate_validation_report)
    except ImportError as e:
        pytest.fail(f"Data quality import failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
