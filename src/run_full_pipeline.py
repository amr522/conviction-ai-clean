#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys

# Optionally disable XRay if not configured
if (
    os.getenv("AWS_XRAY_DAEMON_ADDRESS") in (None, "none")
    or os.getenv("AWS_XRAY_SDK_ENABLED") == "false"
):
    os.environ["AWS_XRAY_SDK_ENABLED"] = "false"

    # Mock XRay components
    class MockXRayRecorder:
        def configure(self, **kwargs):
            pass

        def capture(self, name):
            return lambda f: f

        def begin_subsegment(self, name):
            return self

        def end_subsegment(self):
            pass

        def put_annotation(self, key, value):
            pass

        def put_metadata(self, key, value):
            pass

    xray_recorder = MockXRayRecorder()

    def patch_all():
        pass

else:
    # AWS X-Ray tracing setup
    from aws_xray_sdk.core import patch_all, xray_recorder

    # Patch AWS services and HTTP libraries
    patch_all()
    xray_recorder.configure(
        service="conviction-ai-pipeline",
        plugins=("EC2Plugin", "ECSPlugin"),
        daemon_address="127.0.0.1:2000",
    )

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_intraday_dataset import run as build_intraday
from clean_options_30min import run as clean_options_30min
from clean_options_daily import run as clean_options_daily
# OpenLineage imports
from utils.lineage_utils import LineageTracker
from utils.profiling import (clear_profile_results, enable_profiling,
                             save_profile_report)
from validate_schemas import validate_parquet_schema


@xray_recorder.capture("etl_pipeline")
def run_full_pipeline(
    date: str,
    dry_run: bool = False,
    flow_window: int = 1,
    gamma_squeeze_multiplier: float = 2.0,
    daily_vol_spike_multiplier: float = 2.0,
    check_schema: bool = False,
    profile: bool = False,
    profile_lines: bool = False,
    use_delta: bool = False,
    use_raw_macro: bool = False,
    raw_fred_csv: str = None,
    raw_vix_json: str = None,
    raw_dxy_csv: str = None,
    raw_news_dir: str = None,
):
    """Run the full pipeline with advanced signal parameters"""

    # Enable profiling if requested
    if profile or profile_lines:
        enable_profiling(profile_lines=profile_lines)
        clear_profile_results()
        print(f"🔍 PROFILING ENABLED (lines: {profile_lines})")

    print(f"=== RUNNING FULL PIPELINE FOR {date} ===")
    print(f"Flow window: {flow_window}")
    print(f"Gamma squeeze multiplier: {gamma_squeeze_multiplier}")
    print(f"Daily vol spike multiplier: {daily_vol_spike_multiplier}")
    print(f"Dry run: {dry_run}")
    print(f"Schema validation: {check_schema}")
    print(f"Profiling: {profile} (lines: {profile_lines})")
    print(f"Delta Lake: {use_delta}")

    results = {}

    try:
        # Initialize lineage tracker
        lineage = LineageTracker()

        # Step 1: Clean 30-minute options data with advanced signals
        print("\n" + "=" * 50)
        print("STEP 1: Cleaning 30-minute options data...")

        lineage.start_run(
            "clean_options_30min",
            inputs=["data/Parquet_data/raw/options_30min"],
            outputs=["staged/options_30min_clean.parquet"],
        )

        subsegment = xray_recorder.begin_subsegment("clean_options_30min")
        try:
            xray_recorder.put_annotation("date", date)
            xray_recorder.put_annotation("step", "options_30min_cleaning")

            options_30min_result = clean_options_30min(
                date=date,
                dry_run=dry_run,
                flow_window=flow_window,
                gamma_squeeze_multiplier=gamma_squeeze_multiplier,
            )

            xray_recorder.put_metadata("options_30min_result", options_30min_result)
            results["options_30min"] = options_30min_result
            print(f"✅ 30-minute options: {options_30min_result['rows_processed']} rows")
            lineage.complete_run(success=True)
        except Exception as e:
            lineage.complete_run(success=False)
            raise e
        finally:
            xray_recorder.end_subsegment()

        # Validate schema if requested
        if check_schema and not dry_run:
            try:
                validate_parquet_schema(
                    "staged/options_30min_clean.parquet",
                    "schemas/option_parquet_schema.json",
                    "options_30min",
                )
            except Exception as e:
                print(f"⚠️  Schema validation failed for options_30min: {e}")

        # Step 2: Clean macro data
        print("\n" + "=" * 50)
        print("STEP 2: Cleaning macro data...")

        subsegment = xray_recorder.begin_subsegment("clean_macro_data")
        try:
            xray_recorder.put_annotation("step", "macro_data_cleaning")

            macro_cmd = [sys.executable, "src/clean_macro_data.py"]
            if use_raw_macro:
                macro_cmd.append("--use-raw-macro")
            if raw_fred_csv:
                macro_cmd.extend(["--raw-fred-csv", raw_fred_csv])
            if raw_vix_json:
                macro_cmd.extend(["--raw-vix-json", raw_vix_json])
            if raw_dxy_csv:
                macro_cmd.extend(["--raw-dxy-csv", raw_dxy_csv])
            if raw_news_dir:
                macro_cmd.extend(["--raw-news-dir", raw_news_dir])

            if not dry_run:
                result = subprocess.run(macro_cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print("✅ Macro data processed successfully")
                    results["macro_data"] = {
                        "status": "success",
                        "output": result.stdout,
                    }
                else:
                    print(f"❌ Macro data processing failed: {result.stderr}")
                    results["macro_data"] = {"status": "failed", "error": result.stderr}
            else:
                print("Skipping macro data processing (dry run)")
                results["macro_data"] = {"status": "skipped", "reason": "dry_run"}
        finally:
            xray_recorder.end_subsegment()

        # Step 3: Clean daily options data
        print("\n" + "=" * 50)
        print("STEP 3: Cleaning daily options data...")

        lineage.start_run(
            "clean_options_daily",
            inputs=["data/Parquet_data/raw/options_daily"],
            outputs=["staged/options_daily_clean.parquet"],
        )

        subsegment = xray_recorder.begin_subsegment("clean_options_daily")
        try:
            xray_recorder.put_annotation("step", "options_daily_cleaning")

            options_daily_result = clean_options_daily(
                date=date,
                dry_run=dry_run,
                daily_vol_spike_multiplier=daily_vol_spike_multiplier,
            )

            xray_recorder.put_metadata("options_daily_result", options_daily_result)
            results["options_daily"] = options_daily_result
            print(f"✅ Daily options: {options_daily_result['rows_processed']} rows")
            lineage.complete_run(success=True)
        except Exception as e:
            lineage.complete_run(success=False)
            raise e
        finally:
            xray_recorder.end_subsegment()

        # Validate schema if requested
        if check_schema and not dry_run:
            try:
                validate_parquet_schema(
                    "staged/options_daily_clean.parquet",
                    "schemas/option_parquet_schema.json",
                    "options_daily",
                )
            except Exception as e:
                print(f"⚠️  Schema validation failed for options_daily: {e}")

        # Step 4: Build daily master dataset (if not dry run)
        if not dry_run:
            print("\n" + "=" * 50)
            print("STEP 4: Building daily master dataset...")
            from build_daily_master import run as build_daily_master

            daily_master_result = build_daily_master(
                date=date,
                dry_run=dry_run,
                use_raw_macro=use_raw_macro,
                raw_fred_csv=raw_fred_csv,
                raw_vix_json=raw_vix_json,
                raw_dxy_csv=raw_dxy_csv,
                raw_news_dir=raw_news_dir,
            )
            results["daily_master"] = daily_master_result
            print(f"✅ Daily master: {daily_master_result['rows_processed']} rows")

            print("\n" + "=" * 50)
            print("STEP 5: Building intraday master dataset...")
            intraday_result = build_intraday(date=date, dry_run=dry_run)
            results["intraday"] = intraday_result
            print(f"✅ Intraday master: {intraday_result['rows_processed']} rows")

            # Generate feature parquet after building masters
            print("\n" + "=" * 50)
            print("STEP 6: Generating feature parquet...")
            import polars as pl

            from calculate_features import calculate_all_features

            # Load master datasets
            daily_master_df = pl.read_parquet("staged/daily_master.parquet")
            intraday_master_df = pl.read_parquet("datasets/intraday_master.parquet")

            # Calculate features
            feats = calculate_all_features(daily_master_df, intraday_master_df)
            feats_path = f"data/Parquet_data/features_{date}.parquet"
            feats.write_parquet(feats_path)
            print(f"✅ Features written to {feats_path}")
            results["feature_parquet"] = {"status": "success", "rows": feats.shape[0]}

            # Generate labels automatically
            print("\n" + "=" * 50)
            print("STEP 6.3: Generating labels...")

            labels_path = f"data/Parquet_data/labels_{date}.parquet"
            labels_cmd = [
                sys.executable,
                "src/generate_labels.py",
                "--date",
                date,
                "--output-path",
                labels_path,
            ]

            result = subprocess.run(labels_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Labels generated: {labels_path}")
                results["labels"] = {"status": "success", "output": result.stdout}
            else:
                print(f"❌ Label generation failed: {result.stderr}")
                results["labels"] = {"status": "failed", "error": result.stderr}

            # Generate training dataset if labels exist
            from pathlib import Path

            train_path = f"data/Parquet_data/train_dataset_{date}.parquet"

            if Path(labels_path).exists():
                print("\n" + "=" * 50)
                print("STEP 6.5: Generating training dataset...")

                train_cmd = [
                    sys.executable,
                    "src/generate_training_dataset.py",
                    "--feature-path",
                    feats_path,
                    "--label-path",
                    labels_path,
                    "--output-path",
                    train_path,
                ]

                result = subprocess.run(train_cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✅ Training dataset created: {train_path}")
                    results["training_dataset"] = {
                        "status": "success",
                        "output": result.stdout,
                    }
                else:
                    print(f"❌ Training dataset generation failed: {result.stderr}")
                    results["training_dataset"] = {
                        "status": "failed",
                        "error": result.stderr,
                    }
            else:
                print(
                    f"⚠️ Labels file not found: {labels_path}, skipping training dataset generation"
                )
                results["training_dataset"] = {
                    "status": "skipped",
                    "reason": "labels_not_found",
                }

            print("\n" + "=" * 50)
            print("STEP 7: Calculating features (legacy)...")

            window_days = int(os.getenv("WINDOW_DAYS", "30"))
            use_gpu = os.getenv("USE_GPU", "false").lower() == "true"
            n_jobs = int(os.getenv("N_JOBS", "1"))

            features_cmd = [
                sys.executable,
                "src/calculate_features.py",
                "--daily-master-path",
                "staged/daily_master.parquet",
                "--intraday-master-path",
                "datasets/intraday_master.parquet",
                "--output-path",
                f"datasets/features_{date}.parquet",
                "--date",
                date,
                "--window-days",
                str(window_days),
                "--n-jobs",
                str(n_jobs),
            ]

            if use_gpu:
                features_cmd.append("--use-gpu")

            result = subprocess.run(features_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Features calculated successfully")
                results["features"] = {"status": "success", "output": result.stdout}
            else:
                print(f"❌ Feature calculation failed: {result.stderr}")
                results["features"] = {"status": "failed", "error": result.stderr}
                raise Exception(f"Feature calculation failed: {result.stderr}")
        else:
            print("\n" + "=" * 50)
            print("STEP 4: Skipping daily master build (dry run)")
            results["daily_master"] = {"status": "skipped", "reason": "dry_run"}

            print("\n" + "=" * 50)
            print("STEP 5: Skipping intraday build (dry run)")
            results["intraday"] = {"status": "skipped", "reason": "dry_run"}

            print("\n" + "=" * 50)
            print("STEP 6: Skipping feature parquet generation (dry run)")
            results["feature_parquet"] = {"status": "skipped", "reason": "dry_run"}

            print("\n" + "=" * 50)
            print("STEP 6.3: Skipping label generation (dry run)")
            results["labels"] = {"status": "skipped", "reason": "dry_run"}

            print("\n" + "=" * 50)
            print("STEP 6.5: Skipping training dataset generation (dry run)")
            results["training_dataset"] = {"status": "skipped", "reason": "dry_run"}

            print("\n" + "=" * 50)
            print("STEP 7: Skipping feature calculation (dry run)")
            results["features"] = {"status": "skipped", "reason": "dry_run"}

        # Step 8: Register tables in AWS Glue Data Catalog
        if not dry_run:
            print("\n" + "=" * 50)
            print("STEP 8: Registering tables in AWS Glue Data Catalog...")
            try:
                from utils.glue_catalog import register_pipeline_tables

                s3_bucket = os.getenv("S3_BUCKET_NAME", "conviction-ai-data")
                s3_prefix = os.getenv("S3_PREFIX", "processed/")
                glue_database = os.getenv("GLUE_DATABASE", "conviction_ai")
                aws_region = os.getenv("AWS_REGION", "us-east-1")

                catalog_results = register_pipeline_tables(
                    s3_bucket=s3_bucket,
                    s3_prefix=s3_prefix,
                    database=glue_database,
                    region=aws_region,
                )

                success_count = sum(catalog_results.values())
                total_count = len(catalog_results)

                if success_count == total_count:
                    print(f"✅ All {total_count} tables registered in Glue catalog")
                else:
                    print(
                        f"⚠️ {success_count}/{total_count} tables registered successfully"
                    )

                results["glue_catalog"] = catalog_results

            except Exception as e:
                print(f"⚠️ Glue catalog registration failed: {e}")
                results["glue_catalog"] = {"error": str(e)}

            # Step 9: Write Delta tables if requested
            if use_delta:
                print("\n" + "=" * 50)
                print("STEP 9: Converting to Delta Lake format...")
                try:
                    import pandas as pd

                    from utils.delta_writer import write_delta_table

                    s3_bucket = os.getenv("S3_BUCKET_NAME", "conviction-ai-data")
                    s3_prefix = os.getenv("S3_PREFIX", "delta/")

                    delta_tables = [
                        (
                            "staged/stocks_daily_clean.parquet",
                            f"s3a://{s3_bucket}/{s3_prefix}stocks_daily.delta",
                        ),
                        (
                            "staged/options_daily_clean.parquet",
                            f"s3a://{s3_bucket}/{s3_prefix}options_daily.delta",
                        ),
                        (
                            "staged/stocks_30min_clean.parquet",
                            f"s3a://{s3_bucket}/{s3_prefix}stocks_30min.delta",
                        ),
                        (
                            "staged/options_30min_clean.parquet",
                            f"s3a://{s3_bucket}/{s3_prefix}options_30min.delta",
                        ),
                        (
                            "datasets/intraday_master.parquet",
                            f"s3a://{s3_bucket}/{s3_prefix}intraday_master.delta",
                        ),
                    ]

                    delta_results = {}
                    for parquet_path, delta_path in delta_tables:
                        if os.path.exists(parquet_path):
                            df = pd.read_parquet(parquet_path)
                            success = write_delta_table(
                                df,
                                delta_path,
                                partition_cols=["year", "month"]
                                if "year" in df.columns
                                else None,
                                merge_schema=True,
                            )
                            delta_results[os.path.basename(delta_path)] = success
                            if success:
                                print(f"✅ Delta table: {os.path.basename(delta_path)}")
                            else:
                                print(f"❌ Failed: {os.path.basename(delta_path)}")

                    results["delta_tables"] = delta_results

                except Exception as e:
                    print(f"⚠️ Delta table conversion failed: {e}")
                    results["delta_tables"] = {"error": str(e)}
            else:
                results["delta_tables"] = {
                    "status": "skipped",
                    "reason": "not_requested",
                }

            # Step 11: Materialize features to Feast
            print("\n" + "=" * 50)
            print("STEP 11: Materializing features to Feast feature store...")
            try:
                from feast_materialize import materialize_features

                # Materialize features for the current date
                success = materialize_features(
                    start_date=date, end_date=date, repo_path="feature_repo"
                )

                if success:
                    print(f"✅ Features materialized to Feast for {date}")
                else:
                    print(f"⚠️ Feature materialization failed for {date}")

                results["feast_materialization"] = {"success": success, "date": date}

            except Exception as e:
                print(f"⚠️ Feast materialization failed: {e}")
                results["feast_materialization"] = {"error": str(e)}

        # Step 10: Data Quality Validation
        if not dry_run:
            print("\n" + "=" * 50)
            print("STEP 10: Running data quality validation...")
            try:
                from validate_data_quality import validate_pipeline_outputs

                validation_results = validate_pipeline_outputs(
                    date=date, output_dir="staged", report_dir="metrics"
                )

                passed_count = sum(
                    1 for r in validation_results.values() if r.get("success", False)
                )
                total_count = len(validation_results)

                if passed_count == total_count:
                    print(
                        f"✅ All {total_count} datasets passed data quality validation"
                    )
                else:
                    print(f"⚠️ {passed_count}/{total_count} datasets passed validation")

                    # Send Slack alert for validation failures
                    failed_datasets = [
                        k
                        for k, v in validation_results.items()
                        if not v.get("success", False)
                    ]
                    if failed_datasets:
                        try:
                            from utils.slack_notify import notify_security

                            notify_security(
                                "DATA_QUALITY_FAILED",
                                f"Data quality validation failed for {date}: {', '.join(failed_datasets)}",
                            )
                        except Exception as e:
                            logger.warning(f"Failed to send Slack notification: {e}")

                results["data_quality"] = validation_results

            except Exception as e:
                print(f"⚠️ Data quality validation failed: {e}")
                results["data_quality"] = {"error": str(e)}
        else:
            print("\n" + "=" * 50)
            print("STEP 8: Skipping Glue catalog registration (dry run)")
            results["glue_catalog"] = {"status": "skipped", "reason": "dry_run"}
            results["delta_tables"] = {"status": "skipped", "reason": "dry_run"}

            print("\n" + "=" * 50)
            print("STEP 10: Skipping data quality validation (dry run)")
            results["data_quality"] = {"status": "skipped", "reason": "dry_run"}

            print("\n" + "=" * 50)
            print("STEP 11: Skipping Feast materialization (dry run)")
            results["feast_materialization"] = {
                "status": "skipped",
                "reason": "dry_run",
            }

        print("\n" + "=" * 50)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")

        # Save profiling report if enabled
        if profile or profile_lines:
            report_path = save_profile_report(date)
            print(f"📊 Profile report saved: {report_path}")

        return {"status": "success", "date": date, "results": results}

    except Exception as e:
        print(f"\n❌ PIPELINE FAILED: {e}")
        return {"status": "failed", "date": date, "error": str(e), "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run full options pipeline with advanced signals"
    )
    parser.add_argument(
        "--date", type=str, required=True, help="Processing date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run without writing files"
    )
    parser.add_argument(
        "--flow-window",
        type=int,
        default=1,
        help="Flow divergence smoothing window (default: 1)",
    )
    parser.add_argument(
        "--gamma-squeeze-multiplier",
        type=float,
        default=2.0,
        help="Gamma squeeze threshold multiplier (default: 2.0)",
    )
    parser.add_argument(
        "--daily-vol-spike-multiplier",
        type=float,
        default=2.0,
        help="Daily volume spike threshold multiplier (default: 2.0)",
    )
    parser.add_argument(
        "--check-schema",
        action="store_true",
        help="Validate Parquet schemas against specification",
    )
    parser.add_argument(
        "--profile", action="store_true", help="Enable performance profiling"
    )
    parser.add_argument(
        "--profile-lines",
        action="store_true",
        help="Enable line-by-line profiling with kernprof",
    )
    parser.add_argument(
        "--use-delta",
        action="store_true",
        help="Write outputs as Delta tables instead of Parquet",
    )
    parser.add_argument(
        "--use-raw-macro",
        action="store_true",
        help="Force use of raw macro data sources",
    )
    parser.add_argument(
        "--raw-fred-csv",
        default="/Users/amroheidak/Desktop/conviction-ai-clean/data/Parquet_data/Raw/FRED.csv",
    )
    parser.add_argument(
        "--raw-vix-json",
        default="/Users/amroheidak/Desktop/conviction-ai-clean/data/Parquet_data/Raw/vix_data.json",
    )
    parser.add_argument(
        "--raw-dxy-csv",
        default="/Users/amroheidak/Desktop/conviction-ai-clean/data/Parquet_data/Raw/DXY.csv",
    )
    parser.add_argument(
        "--raw-news-dir",
        default="/Users/amroheidak/Desktop/conviction-ai-clean/data/Parquet_data/Raw/news",
    )

    args = parser.parse_args()

    # Handle line profiling with kernprof
    if args.profile_lines:
        print("Running with line profiling...")
        cmd = [
            "kernprof",
            "-l",
            "-v",
            __file__,
            "--date",
            args.date,
            "--profile",  # Enable regular profiling too
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        if args.check_schema:
            cmd.append("--check-schema")

        # Add other parameters
        cmd.extend(["--flow-window", str(args.flow_window)])
        cmd.extend(["--gamma-squeeze-multiplier", str(args.gamma_squeeze_multiplier)])
        cmd.extend(
            ["--daily-vol-spike-multiplier", str(args.daily_vol_spike_multiplier)]
        )

        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        # Process line profiler output
        lprof_file = f"{__file__}.lprof"
        if os.path.exists(lprof_file):
            line_profile_output = f"logs/line_profile_{args.date}.txt"
            os.makedirs("logs", exist_ok=True)
            with open(line_profile_output, "w") as f:
                line_result = subprocess.run(
                    ["python", "-m", "line_profiler", lprof_file],
                    stdout=f,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            print(f"Line profile saved to: {line_profile_output}")
            os.remove(lprof_file)  # Clean up

        sys.exit(result.returncode)

    result = run_full_pipeline(
        date=args.date,
        dry_run=args.dry_run,
        flow_window=args.flow_window,
        gamma_squeeze_multiplier=args.gamma_squeeze_multiplier,
        daily_vol_spike_multiplier=args.daily_vol_spike_multiplier,
        check_schema=args.check_schema,
        profile=args.profile,
        profile_lines=args.profile_lines,
        use_delta=args.use_delta,
        use_raw_macro=args.use_raw_macro,
        raw_fred_csv=args.raw_fred_csv,
        raw_vix_json=args.raw_vix_json,
        raw_dxy_csv=args.raw_dxy_csv,
        raw_news_dir=args.raw_news_dir,
    )

    print(f"\nFinal result: {result}")
