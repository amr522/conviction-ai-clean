#!/bin/bash

# Production monitoring script
set -e

NAMESPACE=${1:-production}

echo "📊 Monitoring production deployment..."

# Check Prometheus alerts
echo "Checking Prometheus alerts..."
if kubectl -n monitoring get prometheusrule conviction-ai-pipeline-alerts >/dev/null 2>&1; then
    echo "✅ Prometheus alerts configured"
else
    echo "⚠️ Prometheus alerts not found"
fi

# Check chaos exporter metrics
echo "Checking chaos exporter..."
if kubectl -n $NAMESPACE get deployment conviction-ai-pipeline-chaos-exporter >/dev/null 2>&1; then
    echo "✅ Chaos exporter running"
else
    echo "ℹ️ Chaos exporter disabled (expected in production)"
fi

# Check lineage explorer metrics
echo "Checking lineage metrics..."
if kubectl -n $NAMESPACE get servicemonitor conviction-ai-pipeline-lineage-sm >/dev/null 2>&1; then
    echo "✅ Lineage metrics configured"
else
    echo "⚠️ Lineage metrics not found"
fi

# Port forward for local access (optional)
echo "To access Grafana dashboards locally:"
echo "kubectl port-forward -n monitoring svc/grafana 3000:80"
echo "kubectl port-forward -n monitoring svc/prometheus 9090:9090"

echo "✅ Production monitoring check completed"