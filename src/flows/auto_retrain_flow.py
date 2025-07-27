#!/usr/bin/env python3
"""
Prefect Auto-Retrain Flow
Runs daily ETL + training + drift check, triggers backfill only if drift detected
"""
import os
import subprocess
from datetime import date, datetime
from pathlib import Path

from prefect import flow, get_run_logger, task
from prefect.tasks import shell_run_command


@task(name="run-etl-and-train", retries=2, retry_delay_seconds=300)
def run_etl_and_train(dt: str) -> bool:
    """Run ETL and training pipeline for given date"""
    logger = get_run_logger()

    try:
        logger.info(f"Starting ETL and training for {dt}")

        # Run training script
        cmd = f"./scripts/run_and_train.sh {dt}"
        result = shell_run_command(command=cmd, return_all=True)

        if result.return_code == 0:
            logger.info(f"ETL and training completed successfully for {dt}")
            return True
        else:
            logger.error(f"ETL and training failed for {dt}: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"Error in ETL and training: {str(e)}")
        return False


@task(name="check-drift-status")
def check_drift_status(dt: str) -> bool:
    """Check if data drift was detected from logs"""
    logger = get_run_logger()

    try:
        # Check Evidently log for drift status
        log_path = "logs/evidently_log.txt"

        if not os.path.exists(log_path):
            logger.warning(f"Evidently log not found at {log_path}")
            return False

        with open(log_path, "r") as f:
            log_content = f.read()

        # Look for drift detection in logs
        drift_detected = f"Drift detected: True" in log_content

        if drift_detected:
            logger.warning(f"⚠️ Data drift detected for {dt}")
        else:
            logger.info(f"✅ No data drift detected for {dt}")

        return drift_detected

    except Exception as e:
        logger.error(f"Error checking drift status: {str(e)}")
        return False


@task(name="trigger-backfill", retries=1, retry_delay_seconds=600)
def trigger_backfill(dt: str) -> bool:
    """Trigger historical backfill and full retrain"""
    logger = get_run_logger()

    try:
        logger.info(f"Starting historical backfill up to {dt}")

        # Run backfill from start of year to current date
        start_date = f"{dt[:4]}-01-01"  # Start of current year
        cmd = f"./scripts/backfill_flow.sh {start_date} {dt} 24"

        result = shell_run_command(command=cmd, return_all=True)

        if result.return_code == 0:
            logger.info(f"Historical backfill completed successfully")
            return True
        else:
            logger.error(f"Historical backfill failed: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"Error in historical backfill: {str(e)}")
        return False


@task(name="send-drift-notification")
def send_drift_notification(
    dt: str, drift_detected: bool, backfill_success: bool = None
):
    """Send Slack notification about drift status and actions taken"""
    logger = get_run_logger()

    try:
        # Source Slack notification helper
        if drift_detected:
            if backfill_success:
                message = f"⚠️ Data drift detected for {dt}. Historical backfill completed successfully."
                status = "DRIFT_RESOLVED"
            else:
                message = f"🚨 Data drift detected for {dt}. Historical backfill FAILED."
                status = "DRIFT_BACKFILL_FAILED"
        else:
            message = f"✅ No data drift detected for {dt}. Normal training completed."
            status = "NO_DRIFT"

        # Use existing Slack notification function
        slack_cmd = f'./scripts/slack_notify.sh "{status}" "{message}"'
        result = shell_run_command(command=slack_cmd, return_all=True)

        if result.return_code == 0:
            logger.info("Slack notification sent successfully")
        else:
            logger.warning("Failed to send Slack notification")

    except Exception as e:
        logger.warning(f"Error sending notification: {str(e)}")


@flow(name="Auto-Retrain Pipeline", log_prints=True)
def auto_retrain_flow(target_date: str = None):
    """
    Main auto-retrain flow that runs daily

    Args:
        target_date: Date to process (YYYY-MM-DD). Defaults to today.
    """
    logger = get_run_logger()

    # Use provided date or default to today
    if target_date is None:
        target_date = date.today().isoformat()

    logger.info(f"🚀 Starting Auto-Retrain Pipeline for {target_date}")

    # Step 1: Run ETL and training
    training_success = run_etl_and_train(target_date)

    if not training_success:
        logger.error(f"❌ Training failed for {target_date}, aborting pipeline")
        send_drift_notification(target_date, drift_detected=False)
        return {"status": "failed", "reason": "training_failed"}

    # Step 2: Check for data drift
    drift_detected = check_drift_status(target_date)

    # Step 3: Conditional backfill based on drift
    if drift_detected:
        logger.warning(
            f"⚠️ Data drift detected for {target_date}, triggering historical backfill"
        )

        backfill_success = trigger_backfill(target_date)

        if backfill_success:
            logger.info(f"✅ Historical backfill completed successfully")
            send_drift_notification(
                target_date, drift_detected=True, backfill_success=True
            )
            return {
                "status": "success",
                "drift_detected": True,
                "backfill_completed": True,
            }
        else:
            logger.error(f"❌ Historical backfill failed")
            send_drift_notification(
                target_date, drift_detected=True, backfill_success=False
            )
            return {
                "status": "partial_success",
                "drift_detected": True,
                "backfill_completed": False,
            }
    else:
        logger.info(
            f"✅ No data drift detected for {target_date}, skipping historical backfill"
        )
        send_drift_notification(target_date, drift_detected=False)
        return {
            "status": "success",
            "drift_detected": False,
            "backfill_completed": False,
        }


if __name__ == "__main__":
    # For local testing
    import argparse

    parser = argparse.ArgumentParser(description="Run Auto-Retrain Pipeline")
    parser.add_argument("--date", help="Target date (YYYY-MM-DD)")

    args = parser.parse_args()

    result = auto_retrain_flow(args.date)
    print(f"Pipeline result: {result}")
