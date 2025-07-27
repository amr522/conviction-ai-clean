#!/usr/bin/env bash
# Usage: START_DATE=2021-01-01 END_DATE=2025-07-01 bash run_historical_pipeline.sh
# Requires: pip install pandas_market_calendars

# Default values
START_DATE=${START_DATE:-2021-01-01}
END_DATE=${END_DATE:-2025-07-01}

echo "🚀 Starting Historical Pipeline Run"
echo "Date range: $START_DATE to $END_DATE"
echo "=========================================="

# Fetch NYSE trading calendar
echo "📅 Fetching NYSE trading calendar..."
market_cal="NYSE"

# Try to get trading dates using pandas_market_calendars
trading_dates=$(python - << 'PYCODE'
try:
    import pandas_market_calendars as mcal
    cal = mcal.get_calendar('NYSE')
    schedule = cal.schedule(start_date='${START_DATE}', end_date='${END_DATE}')
    print('\n'.join(schedule.index.strftime('%Y-%m-%d')))
except ImportError:
    print("FALLBACK")
PYCODE
)

# Check if pandas_market_calendars is available
if [ "$trading_dates" = "FALLBACK" ]; then
    echo "⚠ pandas_market_calendars not installed, using weekend fallback"
    echo "Install with: pip install pandas_market_calendars"

    # Fallback: iterate through all dates and skip weekends
    start_seconds=$(date -j -f "%Y-%m-%d" "$START_DATE" "+%s")
    end_seconds=$(date -j -f "%Y-%m-%d" "$END_DATE" "+%s")
    current_seconds=$start_seconds

    while [ $current_seconds -le $end_seconds ]; do
        current_date=$(date -j -f "%s" "$current_seconds" "+%Y-%m-%d")
        day_of_week=$(date -j -f "%s" "$current_seconds" "+%u")

        # Skip weekends (6=Saturday, 7=Sunday)
        if [ "$day_of_week" -eq 6 ] || [ "$day_of_week" -eq 7 ]; then
            echo "⏭ Skipping weekend $current_date"
        else
            echo "▶ Processing $current_date"

            # Run full pipeline with error handling
            if python src/run_full_pipeline.py --date "$current_date"; then
                if python src/validate_pipeline.py --date "$current_date"; then
                    echo "✓ Completed $current_date"
                else
                    echo "⚠ Validation failed for $current_date"
                fi
            else
                echo "⚠ No data available for $current_date (skipping)"
            fi
        fi

        current_seconds=$((current_seconds + 86400))
    done
else
    echo "✅ Using NYSE trading calendar"

    # Process only trading dates
    echo "$trading_dates" | while read -r current_date; do
        echo "▶ Processing $current_date (trading day)"

        # Run full pipeline with error handling
        if python src/run_full_pipeline.py --date "$current_date"; then
            if python src/validate_pipeline.py --date "$current_date"; then
                echo "✓ Completed $current_date"
            else
                echo "⚠ Validation failed for $current_date"
            fi
        else
            echo "⚠ No data available for $current_date (skipping)"
        fi
    done
fi

echo "=========================================="
echo "🎉 Historical pipeline run completed!"
echo "Processed date range: $START_DATE to $END_DATE"
