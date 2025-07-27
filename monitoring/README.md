# Production Monitoring Setup

This directory contains monitoring and alerting configuration for the Conviction-AI ETL pipeline.

## Components

### Metrics Exporter
- **File**: `src/metrics_exporter.py`
- **Port**: 8000
- **Metrics**: Pipeline status, file counts, data freshness, runtime

### Prometheus
- **Port**: 9090
- **Config**: `prometheus/prometheus.yml`
- **Alerts**: `alerts/rules.yml`

### Grafana
- **Port**: 3000
- **Dashboard**: `dashboards/vol-pipeline.json`
- **Default Login**: admin/admin

### Alertmanager
- **Port**: 9093
- **Config**: `alertmanager/alertmanager.yml`
- **Telegram Integration**: Routes alerts to configured Telegram chat

## Quick Start

### 1. Start Monitoring Stack

```bash
cd monitoring
docker-compose -f docker-compose.monitoring.yml up -d
```

### 2. Access Dashboards

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000
- **Alertmanager**: http://localhost:9093
- **Metrics**: http://localhost:8000/metrics

### 3. Configure Telegram Bot

```bash
# Set environment variables before starting
export TELEGRAM_BOT_TOKEN=123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11
export TELEGRAM_CHAT_ID=-1001234567890
cd monitoring
docker-compose -f docker-compose.monitoring.yml up -d
```

**Getting Telegram credentials:**
1. Create a bot by messaging @BotFather on Telegram
2. Get your bot token from BotFather
3. Add the bot to your group/channel and get the chat ID
4. Use a tool like @userinfobot to find your chat ID

### 4. Import Grafana Dashboard

1. Open Grafana (http://localhost:3000)
2. Login with admin/admin
3. Go to Dashboards → Import
4. Upload `dashboards/vol-pipeline.json`

## Alerts

### Critical Alerts
- **PipelineFailure**: Last run failed (exit code 1)
- **MetricsExporterDown**: Metrics service unavailable

### Warning Alerts
- **MissingOutput**: No files in master directory
- **DataLag**: Data older than 25 hours
- **LongRuntime**: Pipeline took >1 hour

**Telegram alerts** will be sent to your configured chat with formatted messages including alert details, timestamps, and severity indicators.

## Metrics Available

```
vol_pipeline_files_total{directory="datasets|master|staged"}
vol_pipeline_size_mb{directory="datasets|master|staged"}
vol_pipeline_runtime_seconds
vol_pipeline_data_age_hours
vol_pipeline_last_run_status
vol_pipeline_up
```

## Production Deployment

### Kubernetes

Deploy metrics exporter alongside the main pipeline:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vol-metrics-exporter
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vol-metrics-exporter
  template:
    spec:
      containers:
      - name: metrics-exporter
        image: your-registry/metrics-exporter:1.0.0
        ports:
        - containerPort: 8000
        volumeMounts:
        - name: datasets
          mountPath: /app/datasets
          readOnly: true
```

### Prometheus Configuration

Add to your existing Prometheus config:

```yaml
scrape_configs:
  - job_name: 'vol-pipeline'
    static_configs:
      - targets: ['vol-metrics-exporter:8000']
```

## Troubleshooting

### Metrics Not Updating
- Check if pipeline is writing success/failure flags
- Verify volume mounts for data directories
- Check metrics exporter logs: `docker logs vol-metrics-exporter`

### Alerts Not Firing
- Verify Prometheus can scrape metrics endpoint
- Check alert rule syntax in `alerts/rules.yml`
- Ensure alertmanager is configured

### Dashboard Issues
- Verify Prometheus data source in Grafana
- Check metric names match dashboard queries
- Ensure time range covers recent data