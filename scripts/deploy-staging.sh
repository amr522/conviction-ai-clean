#!/usr/bin/env bash
set -euo pipefail

NAMESPACE=${1:-staging}

echo "🚀 Deploying to $NAMESPACE namespace..."

# Create namespace if it doesn't exist
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply staging configuration
kubectl apply -n $NAMESPACE -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: conviction-ai-config
data:
  ENVIRONMENT: "staging"
  LOG_LEVEL: "INFO"
  PYTHONPATH: "/app/src"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: conviction-ai-pipeline
  labels:
    app: conviction-ai-pipeline
spec:
  replicas: 1
  selector:
    matchLabels:
      app: conviction-ai-pipeline
  template:
    metadata:
      labels:
        app: conviction-ai-pipeline
    spec:
      containers:
      - name: pipeline
        image: conviction-ai-pipeline:latest
        imagePullPolicy: Never
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: conviction-ai-config
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: conviction-ai-pipeline
spec:
  selector:
    app: conviction-ai-pipeline
  ports:
  - port: 8000
    targetPort: 8000
    name: http
EOF

echo "✅ Deployment to $NAMESPACE completed"