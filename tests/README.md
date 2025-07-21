# Automated Tests for Conviction-AI

This directory contains automated tests for the Conviction-AI project. These tests validate the functionality of various scripts in the codebase, ensuring they work as expected and catch regressions before they make it to production.

## Tests Overview

- **test_train_local.py**: Tests for training models locally including feature preparation, cross-validation, and ensemble model functionality.
- **test_gpu_optimizer.py**: Tests for GPU memory optimization functions, ensuring efficient GPU utilization during training.
- **test_fix_data_leakage.py**: Tests for data leakage detection and fixing functionality.
- **test_predict_volatility.py**: Tests for the volatility prediction pipeline.
- **test_fix_ensemble_issue.py**: Tests for identifying and fixing issues with ensemble models.
- **test_check_parquet_columns.py**: Tests for validating parquet file schemas and contents.

## Prerequisites

Install the required packages for running the tests:

```bash
pip install -r requirements.txt
pip install pytest pytest-cov
```

## Running Tests

To run all tests:

```bash
pytest -v
```

To run a specific test file:

```bash
pytest -v tests/test_train_local.py
```

To generate a coverage report:

```bash
pytest --cov=. --cov-report=html
```

Then open `htmlcov/index.html` in your browser to view the coverage report.

## Continuous Integration

These tests can be integrated into a CI/CD pipeline using GitHub Actions. A sample workflow is provided in `.github/workflows/test.yml`.

## Writing New Tests

When adding new functionality to the codebase, please also add corresponding tests. Follow these guidelines:

1. Create a new file named `test_<script_name>.py` for each script being tested.
2. Use pytest fixtures for common setup and teardown operations.
3. Mock external dependencies like file I/O, GPU operations, etc.
4. Include tests for both success cases and error handling.
5. Aim for at least 80% code coverage.

## Test Fixtures

Common test fixtures are defined in `conftest.py`. These include:

- Mock SparkSession and GlueContext for AWS Glue tests
- Mock boto3 clients for AWS service tests
- Sample data frames for different data types
- Mock GPU/hardware utilities

Feel free to add new fixtures as needed for your tests.

## Troubleshooting

If you encounter issues running the tests:

1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Check that pytest is installed: `pip install pytest pytest-cov`
3. Verify that your Python environment has access to the required modules
4. For GPU-related tests, some tests may be skipped if no GPU is available

## Contact

If you have questions about these tests or need help writing new ones, please contact the team.
