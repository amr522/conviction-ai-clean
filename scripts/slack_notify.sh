#!/bin/bash

TITLE="$1"
MESSAGE="$2"
WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"

if [ -z "$WEBHOOK_URL" ]; then
    echo "SLACK_WEBHOOK_URL not set, skipping notification"
    exit 0
fi

curl -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"$TITLE\n$MESSAGE\"}" \
    "$WEBHOOK_URL"