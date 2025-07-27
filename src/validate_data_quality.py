#!/usr/bin/env python3
"""
Great Expectations data quality validation
"""
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import great_expectations as gx
from great_expectations.checkpoint import SimpleCheckpoint
from great_expectations.core.batch import RuntimeBatchRequest

logger = logging.getLogger(__name__)

class DataQualityError(Exception):
    """Custom exception for data quality validation failures"""
    pass

def get_data_context() -> gx.DataContext:
    """
    Get or create Great Expectations data context
    
    Returns:
        Great Expectations DataContext
    """
    try:
        # Try to get existing context
        context = gx.get_context()
        logger.info("Using existing Great Expectations context")
        return context
    except Exception:
        # Initialize new context if none exists
        logger.info("Initializing new Great Expectations context")
        context = gx.get_context(project_root_dir=".")
        return context

def create_expectation_suite(
    context: gx.DataContext,
    suite_name: str,
    dataset_type: str
) -> None:
    """
    Create expectation suite for dataset type
    
    Args:
        context: Great Expectations context
        suite_name: Name of expectation suite
        dataset_type: Type of dataset (options_daily, options_30min, etc.)
    """
    try:
        # Try to get existing suite
        suite = context.get_expectation_suite(suite_name)
        logger.info(f"Using existing expectation suite: {suite_name}")
        return
    except Exception:
        # Create new suite
        logger.info(f"Creating new expectation suite: {suite_name}")
        suite = context.create_expectation_suite(suite_name)
    
    # Define expectations based on dataset type
    if dataset_type == "options_daily":
        expectations = [
            # Core columns must exist
            {"expectation_type": "expect_column_to_exist", "kwargs": {"column": "timestamp"}},
            {"expectation_type": "expect_column_to_exist", "kwargs": {"column": "symbol"}},
            {"expectation_type": "expect_column_to_exist", "kwargs": {"column": "optd_close"}},
            {"expectation_type": "expect_column_to_exist", "kwargs": {"column": "optd_volume"}},
            
            # Data types
            {"expectation_type": "expect_column_values_to_be_of_type", "kwargs": {"column": "optd_close", "type_": "float64"}},
            {"expectation_type": "expect_column_values_to_be_of_type", "kwargs": {"column": "optd_volume", "type_": "int64"}},
            
            # Null checks
            {"expectation_type": "expect_column_values_to_not_be_null", "kwargs": {"column": "timestamp"}},
            {"expectation_type": "expect_column_values_to_not_be_null", "kwargs": {"column": "symbol"}},
            
            # Range checks
            {"expectation_type": "expect_column_values_to_be_between", "kwargs": {"column": "optd_close", "min_value": 0.01, "max_value": 10000}},
            {"expectation_type": "expect_column_values_to_be_between", "kwargs": {"column": "optd_volume", "min_value": 0, "max_value": 1000000}},
            
            # Moneyness should be reasonable
            {"expectation_type": "expect_column_values_to_be_between", "kwargs": {"column": "optd_moneyness", "min_value": 0.1, "max_value": 10.0}},
        ]
    
    elif dataset_type == "options_30min":
        expectations = [
            # Core columns must exist
            {"expectation_type": "expect_column_to_exist", "kwargs": {"column": "timestamp"}},
            {"expectation_type": "expect_column_to_exist", "kwargs": {"column": "symbol"}},
            {"expectation_type": "expect_column_to_exist", "kwargs": {"column": "opt30_close"}},
            {"expectation_type": "expect_column_to_exist", "kwargs": {"column": "opt30_volume"}},
            
            # Data types
            {"expectation_type": "expect_column_values_to_be_of_type", "kwargs": {"column": "opt30_close", "type_": "float64"}},
            {"expectation_type": "expect_column_values_to_be_of_type", "kwargs": {"column": "opt30_volume", "type_": "int64"}},
            
            # Null checks
            {"expectation_type": "expect_column_values_to_not_be_null", "kwargs": {"column": "timestamp"}},
            {"expectation_type": "expect_column_values_to_not_be_null", "kwargs": {"column": "symbol"}},
            
            # Range checks
            {"expectation_type": "expect_column_values_to_be_between", "kwargs": {"column": "opt30_close", "min_value": 0.01, "max_value": 10000}},
            {"expectation_type": "expect_column_values_to_be_between", "kwargs": {"column": "opt30_volume", "min_value": 0, "max_value": 100000}},
            
            # Flow analysis checks
            {"expectation_type": "expect_column_values_to_be_between", "kwargs": {"column": "opt30_call_flow", "min_value": 0, "max_value": 1000000}},
            {"expectation_type": "expect_column_values_to_be_between", "kwargs": {"column": "opt30_put_flow", "min_value": 0, "max_value": 1000000}},
        ]
    
    elif dataset_type == "stocks_daily":
        expectations = [
            # Core columns must exist
            {"expectation_type": "expect_column_to_exist", "kwargs": {"column": "timestamp"}},
            {"expectation_type": "expect_column_to_exist", "kwargs": {"column": "symbol"}},
            {"expectation_type": "expect_column_to_exist", "kwargs": {"column": "close"}},
            {"expectation_type": "expect_column_to_exist", "kwargs": {"column": "volume"}},
            
            # Data types
            {"expectation_type": "expect_column_values_to_be_of_type", "kwargs": {"column": "close", "type_": "float64"}},
            {"expectation_type": "expect_column_values_to_be_of_type", "kwargs": {"column": "volume", "type_": "int64"}},
            
            # Null checks
            {"expectation_type": "expect_column_values_to_not_be_null", "kwargs": {"column": "timestamp"}},
            {"expectation_type": "expect_column_values_to_not_be_null", "kwargs": {"column": "symbol"}},
            
            # Range checks
            {"expectation_type": "expect_column_values_to_be_between", "kwargs": {"column": "close", "min_value": 0.01, "max_value": 10000}},
            {"expectation_type": "expect_column_values_to_be_between", "kwargs": {"column": "volume", "min_value": 0, "max_value": 1000000000}},
        ]
    
    else:
        # Generic expectations
        expectations = [
            {"expectation_type": "expect_table_row_count_to_be_between", "kwargs": {"min_value": 1, "max_value": 10000000}},
        ]
    
    # Add expectations to suite
    for exp in expectations:
        suite.add_expectation(gx.expectations.registry.get_expectation_class_from_expectation_type(exp["expectation_type"])(**exp["kwargs"]))
    
    # Save suite
    context.save_expectation_suite(suite)
    logger.info(f"Created expectation suite '{suite_name}' with {len(expectations)} expectations")

def validate_dataset(
    file_path: str,
    dataset_type: str,
    date: str,
    context: Optional[gx.DataContext] = None
) -> Dict:
    """
    Validate dataset using Great Expectations
    
    Args:
        file_path: Path to dataset file
        dataset_type: Type of dataset for expectation suite selection
        date: Processing date for reporting
        context: Great Expectations context (optional)
        
    Returns:
        Validation results dictionary
        
    Raises:
        DataQualityError: If validation fails
    """
    if context is None:
        context = get_data_context()
    
    suite_name = f"{dataset_type}_suite"
    
    # Create expectation suite if it doesn't exist
    create_expectation_suite(context, suite_name, dataset_type)
    
    try:
        # Create datasource if it doesn't exist
        datasource_name = "parquet_datasource"
        try:
            datasource = context.get_datasource(datasource_name)
        except Exception:
            logger.info(f"Creating datasource: {datasource_name}")
            datasource = context.sources.add_pandas_filesystem(
                name=datasource_name,
                base_directory=str(Path(file_path).parent)
            )
        
        # Create batch request
        batch_request = RuntimeBatchRequest(
            datasource_name=datasource_name,
            data_connector_name="default_runtime_data_connector_name",
            data_asset_name=f"{dataset_type}_{date}",
            runtime_parameters={"path": file_path},
            batch_identifiers={"default_identifier_name": f"{dataset_type}_{date}"}
        )
        
        # Create and run checkpoint
        checkpoint_name = f"{dataset_type}_checkpoint"
        checkpoint = SimpleCheckpoint(
            name=checkpoint_name,
            data_context=context,
            validations=[
                {
                    "batch_request": batch_request,
                    "expectation_suite_name": suite_name
                }
            ]
        )
        
        # Run validation
        logger.info(f"Running validation for {dataset_type} on {date}")
        results = checkpoint.run()
        
        # Check if validation passed
        success = results["success"]
        
        if not success:
            logger.error(f"Data quality validation failed for {dataset_type}")
            # Get detailed failure information
            validation_result = results["run_results"][list(results["run_results"].keys())[0]]
            failed_expectations = []
            
            for result in validation_result["validation_result"]["results"]:
                if not result["success"]:
                    failed_expectations.append({
                        "expectation_type": result["expectation_config"]["expectation_type"],
                        "column": result["expectation_config"]["kwargs"].get("column", "N/A"),
                        "observed_value": result["result"].get("observed_value", "N/A")
                    })
            
            error_msg = f"Validation failed for {dataset_type}: {len(failed_expectations)} expectations failed"
            logger.error(error_msg)
            for failure in failed_expectations[:5]:  # Show first 5 failures
                logger.error(f"  - {failure['expectation_type']} on {failure['column']}: {failure['observed_value']}")
            
            raise DataQualityError(error_msg)
        
        logger.info(f"Data quality validation passed for {dataset_type}")
        
        return {
            "success": success,
            "dataset_type": dataset_type,
            "date": date,
            "file_path": file_path,
            "results": results
        }
        
    except Exception as e:
        if isinstance(e, DataQualityError):
            raise
        logger.error(f"Error during validation: {str(e)}")
        raise DataQualityError(f"Validation error for {dataset_type}: {str(e)}")

def generate_validation_report(
    validation_results: Dict,
    output_dir: str = "metrics"
) -> str:
    """
    Generate HTML validation report
    
    Args:
        validation_results: Results from validate_dataset
        output_dir: Directory to save report
        
    Returns:
        Path to generated report
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        dataset_type = validation_results["dataset_type"]
        date = validation_results["date"]
        
        # Generate report filename
        report_path = os.path.join(output_dir, f"ge_validation_{dataset_type}_{date}.html")
        
        # Get validation result details
        results = validation_results["results"]
        run_id = list(results["run_results"].keys())[0]
        validation_result = results["run_results"][run_id]["validation_result"]
        
        # Create simple HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Quality Report - {dataset_type} ({date})</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
                .success {{ color: green; }}
                .failure {{ color: red; }}
                .expectation {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Data Quality Report</h1>
                <p><strong>Dataset:</strong> {dataset_type}</p>
                <p><strong>Date:</strong> {date}</p>
                <p><strong>Status:</strong> <span class="{'success' if validation_results['success'] else 'failure'}">
                    {'PASSED' if validation_results['success'] else 'FAILED'}
                </span></p>
            </div>
            
            <h2>Validation Results</h2>
        """
        
        # Add expectation results
        for result in validation_result["results"]:
            expectation_type = result["expectation_config"]["expectation_type"]
            column = result["expectation_config"]["kwargs"].get("column", "N/A")
            success = result["success"]
            
            html_content += f"""
            <div class="expectation">
                <h3 class="{'success' if success else 'failure'}">
                    {'✅' if success else '❌'} {expectation_type}
                </h3>
                <p><strong>Column:</strong> {column}</p>
                <p><strong>Status:</strong> {'PASSED' if success else 'FAILED'}</p>
            </div>
            """
        
        html_content += """
            </body>
        </html>
        """
        
        # Write report
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Validation report generated: {report_path}")
        return report_path
        
    except Exception as e:
        logger.error(f"Failed to generate validation report: {str(e)}")
        return ""

def validate_pipeline_outputs(
    date: str,
    output_dir: str = "staged",
    report_dir: str = "metrics"
) -> Dict[str, bool]:
    """
    Validate all pipeline outputs for a given date
    
    Args:
        date: Processing date
        output_dir: Directory containing output files
        report_dir: Directory to save validation reports
        
    Returns:
        Dictionary of validation results by dataset type
    """
    context = get_data_context()
    results = {}
    
    # Define datasets to validate
    datasets = [
        ("options_daily", f"{output_dir}/options_daily_clean.parquet"),
        ("options_30min", f"{output_dir}/options_30min_clean.parquet"),
        ("stocks_daily", f"{output_dir}/stocks_daily_clean.parquet"),
        ("stocks_30min", f"{output_dir}/stocks_30min_clean.parquet"),
    ]
    
    for dataset_type, file_path in datasets:
        if os.path.exists(file_path):
            try:
                validation_result = validate_dataset(file_path, dataset_type, date, context)
                
                # Generate report
                report_path = generate_validation_report(validation_result, report_dir)
                validation_result["report_path"] = report_path
                
                results[dataset_type] = validation_result
                logger.info(f"✅ Validation passed: {dataset_type}")
                
            except DataQualityError as e:
                logger.error(f"❌ Validation failed: {dataset_type} - {str(e)}")
                results[dataset_type] = {
                    "success": False,
                    "error": str(e),
                    "dataset_type": dataset_type,
                    "date": date,
                    "file_path": file_path
                }
        else:
            logger.warning(f"⚠️ File not found: {file_path}")
            results[dataset_type] = {
                "success": False,
                "error": "File not found",
                "dataset_type": dataset_type,
                "date": date,
                "file_path": file_path
            }
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate data quality with Great Expectations")
    parser.add_argument("--date", required=True, help="Processing date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", default="staged", help="Directory containing output files")
    parser.add_argument("--report-dir", default="metrics", help="Directory to save reports")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run validation
    results = validate_pipeline_outputs(args.date, args.output_dir, args.report_dir)
    
    # Print summary
    total = len(results)
    passed = sum(1 for r in results.values() if r.get("success", False))
    
    print(f"\nValidation Summary for {args.date}:")
    print(f"  Total datasets: {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {total - passed}")
    
    for dataset_type, result in results.items():
        status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
        print(f"  {dataset_type}: {status}")
        if not result.get("success", False) and "error" in result:
            print(f"    Error: {result['error']}")
    
    # Exit with error if any validation failed
    if passed < total:
        exit(1)