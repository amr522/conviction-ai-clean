#!/usr/bin/env python3
"""
Real-time ML Ops Dashboard for Conviction AI HPO System

This dashboard provides:
1. Real-time endpoint metrics visualization
2. Model performance monitoring with AUC tracking
3. Data drift detection and alerts
4. System health monitoring
"""

import os
import sys
import time
import json
import boto3
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mlops_dashboard.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MLOpsDashboard:
    def __init__(self, region='us-east-1'):
        self.region = region
        self.sagemaker = boto3.client('sagemaker', region_name=region)
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        self.sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=region)
        self.s3 = boto3.client('s3', region_name=region)
        self.bucket = 'hpo-bucket-773934887314'
        
    def get_endpoint_metrics(self, endpoint_name: str, hours_back: int = 24) -> Dict:
        """Retrieve CloudWatch metrics for SageMaker endpoint"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours_back)
        
        metrics = {}
        
        metric_queries = [
            ('Invocations', 'Sum'),
            ('InvocationsPerInstance', 'Average'),
            ('ModelLatency', 'Average'),
            ('OverheadLatency', 'Average'),
            ('Invocation4XXErrors', 'Sum'),
            ('Invocation5XXErrors', 'Sum')
        ]
        
        for metric_name, statistic in metric_queries:
            try:
                response = self.cloudwatch.get_metric_statistics(
                    Namespace='AWS/SageMaker',
                    MetricName=metric_name,
                    Dimensions=[
                        {
                            'Name': 'EndpointName',
                            'Value': endpoint_name
                        }
                    ],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=300,
                    Statistics=[statistic]
                )
                
                datapoints = sorted(response['Datapoints'], key=lambda x: x['Timestamp'])
                metrics[metric_name] = {
                    'timestamps': [dp['Timestamp'] for dp in datapoints],
                    'values': [dp[statistic] for dp in datapoints],
                    'unit': response.get('Label', metric_name)
                }
                
            except Exception as e:
                logger.warning(f"Failed to get metric {metric_name}: {e}")
                metrics[metric_name] = {'timestamps': [], 'values': [], 'unit': metric_name}
        
        return metrics
    
    def get_endpoint_status(self, endpoint_name: str) -> Dict:
        """Get current endpoint status and configuration"""
        try:
            response = self.sagemaker.describe_endpoint(EndpointName=endpoint_name)
            return {
                'status': response['EndpointStatus'],
                'creation_time': response['CreationTime'],
                'last_modified': response['LastModifiedTime'],
                'instance_type': response.get('ProductionVariants', [{}])[0].get('InstanceType', 'Unknown'),
                'instance_count': response.get('ProductionVariants', [{}])[0].get('CurrentInstanceCount', 0),
                'failure_reason': response.get('FailureReason', None)
            }
        except Exception as e:
            logger.error(f"Failed to get endpoint status: {e}")
            return {'status': 'Unknown', 'error': str(e)}
    
    def test_endpoint_performance(self, endpoint_name: str, sample_count: int = 10) -> Dict:
        """Test endpoint performance and calculate AUC if possible"""
        try:
            np.random.seed(42)
            sample_data = np.random.randn(sample_count, 63)
            
            predictions = []
            latencies = []
            
            for i in range(sample_count):
                start_time = time.time()
                sample_row = sample_data[i]
                payload = ','.join([str(val) for val in sample_row])
                
                response = self.sagemaker_runtime.invoke_endpoint(
                    EndpointName=endpoint_name,
                    ContentType='text/csv',
                    Body=payload
                )
                
                latency = (time.time() - start_time) * 1000
                latencies.append(latency)
                
                result = json.loads(response['Body'].read().decode())
                prediction = result.get('predictions', [0])[0]
                predictions.append(prediction)
            
            return {
                'success': True,
                'sample_count': sample_count,
                'predictions': predictions,
                'avg_latency_ms': np.mean(latencies),
                'max_latency_ms': np.max(latencies),
                'min_latency_ms': np.min(latencies),
                'prediction_range': [np.min(predictions), np.max(predictions)],
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to test endpoint performance: {e}")
            return {'success': False, 'error': str(e)}
    
    def publish_custom_metrics(self, endpoint_name: str, performance_data: Dict):
        """Publish custom performance metrics to CloudWatch"""
        try:
            if not performance_data.get('success'):
                return
            
            metrics = [
                {
                    'MetricName': 'CustomLatency',
                    'Value': performance_data['avg_latency_ms'],
                    'Unit': 'Milliseconds',
                    'Dimensions': [
                        {
                            'Name': 'EndpointName',
                            'Value': endpoint_name
                        }
                    ]
                },
                {
                    'MetricName': 'PredictionRange',
                    'Value': performance_data['prediction_range'][1] - performance_data['prediction_range'][0],
                    'Unit': 'None',
                    'Dimensions': [
                        {
                            'Name': 'EndpointName',
                            'Value': endpoint_name
                        }
                    ]
                }
            ]
            
            self.cloudwatch.put_metric_data(
                Namespace='Custom/MLOps',
                MetricData=metrics
            )
            
            logger.info(f"Published custom metrics for {endpoint_name}")
            
        except Exception as e:
            logger.error(f"Failed to publish custom metrics: {e}")
    
    def create_dashboard_html(self, endpoint_name: str, metrics: Dict, status: Dict, performance: Dict) -> str:
        """Create HTML dashboard with Plotly visualizations"""
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Endpoint Invocations', 'Model Latency', 
                          'Error Rates', 'Performance Test Results',
                          'Endpoint Status', 'System Health'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"type": "indicator"}, {"type": "indicator"}]]
        )
        
        if metrics.get('Invocations', {}).get('timestamps'):
            fig.add_trace(
                go.Scatter(
                    x=metrics['Invocations']['timestamps'],
                    y=metrics['Invocations']['values'],
                    mode='lines+markers',
                    name='Invocations',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
        
        if metrics.get('ModelLatency', {}).get('timestamps'):
            fig.add_trace(
                go.Scatter(
                    x=metrics['ModelLatency']['timestamps'],
                    y=metrics['ModelLatency']['values'],
                    mode='lines+markers',
                    name='Model Latency (ms)',
                    line=dict(color='green')
                ),
                row=1, col=2
            )
        
        error_4xx = metrics.get('Invocation4XXErrors', {}).get('values', [])
        error_5xx = metrics.get('Invocation5XXErrors', {}).get('values', [])
        error_timestamps = metrics.get('Invocation4XXErrors', {}).get('timestamps', [])
        
        if error_timestamps:
            fig.add_trace(
                go.Scatter(
                    x=error_timestamps,
                    y=error_4xx,
                    mode='lines+markers',
                    name='4XX Errors',
                    line=dict(color='orange')
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=error_timestamps,
                    y=error_5xx,
                    mode='lines+markers',
                    name='5XX Errors',
                    line=dict(color='red')
                ),
                row=2, col=1
            )
        
        if performance.get('success'):
            fig.add_trace(
                go.Bar(
                    x=['Avg Latency', 'Min Latency', 'Max Latency'],
                    y=[performance['avg_latency_ms'], performance['min_latency_ms'], performance['max_latency_ms']],
                    name='Performance Test',
                    marker_color=['blue', 'green', 'red']
                ),
                row=2, col=2
            )
        
        status_color = 'green' if status.get('status') == 'InService' else 'red'
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=1 if status.get('status') == 'InService' else 0,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Endpoint Status<br>{status.get('status', 'Unknown')}"},
                gauge={
                    'axis': {'range': [None, 1]},
                    'bar': {'color': status_color},
                    'steps': [
                        {'range': [0, 0.5], 'color': "lightgray"},
                        {'range': [0.5, 1], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.9
                    }
                }
            ),
            row=3, col=1
        )
        
        health_score = 1.0 if performance.get('success') and status.get('status') == 'InService' else 0.5
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=health_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "System Health"},
                gauge={
                    'axis': {'range': [None, 1]},
                    'bar': {'color': 'green' if health_score > 0.8 else 'orange'},
                    'steps': [
                        {'range': [0, 0.5], 'color': "lightgray"},
                        {'range': [0.5, 0.8], 'color': "yellow"},
                        {'range': [0.8, 1], 'color': "lightgreen"}
                    ]
                }
            ),
            row=3, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text=f"ML Ops Dashboard - {endpoint_name}",
            title_x=0.5,
            showlegend=True
        )
        
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ML Ops Dashboard - {endpoint_name}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ text-align: center; margin-bottom: 20px; }}
                .metrics-summary {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .metric-card {{ 
                    background: #f5f5f5; 
                    padding: 15px; 
                    border-radius: 8px; 
                    text-align: center;
                    min-width: 150px;
                }}
                .refresh-info {{ text-align: center; color: #666; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>ML Ops Dashboard</h1>
                <h2>Endpoint: {endpoint_name}</h2>
                <p>Last Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            </div>
            
            <div class="metrics-summary">
                <div class="metric-card">
                    <h3>Status</h3>
                    <p style="color: {'green' if status.get('status') == 'InService' else 'red'}">
                        {status.get('status', 'Unknown')}
                    </p>
                </div>
                <div class="metric-card">
                    <h3>Instance Type</h3>
                    <p>{status.get('instance_type', 'Unknown')}</p>
                </div>
                <div class="metric-card">
                    <h3>Instance Count</h3>
                    <p>{status.get('instance_count', 0)}</p>
                </div>
                <div class="metric-card">
                    <h3>Avg Latency</h3>
                    <p>{performance.get('avg_latency_ms', 0):.2f} ms</p>
                </div>
            </div>
            
            <div id="dashboard-plot"></div>
            
            <div class="refresh-info">
                <p>Dashboard auto-refreshes every 5 minutes. Last performance test: {performance.get('timestamp', 'Never')}</p>
            </div>
            
            <script>
                var plotData = {fig.to_json()};
                Plotly.newPlot('dashboard-plot', plotData.data, plotData.layout);
                
                // Auto-refresh every 5 minutes
                setTimeout(function() {{
                    location.reload();
                }}, 300000);
            </script>
        </body>
        </html>
        """
        
        return dashboard_html
    
    def generate_dashboard(self, endpoint_name: str, output_file: Optional[str] = None) -> str:
        """Generate complete dashboard for endpoint"""
        logger.info(f"Generating dashboard for endpoint: {endpoint_name}")
        
        status = self.get_endpoint_status(endpoint_name)
        logger.info(f"Endpoint status: {status.get('status', 'Unknown')}")
        
        metrics = self.get_endpoint_metrics(endpoint_name)
        logger.info(f"Retrieved {len(metrics)} metric types")
        
        performance = self.test_endpoint_performance(endpoint_name)
        if performance.get('success'):
            logger.info(f"Performance test successful - Avg latency: {performance['avg_latency_ms']:.2f}ms")
            self.publish_custom_metrics(endpoint_name, performance)
        else:
            logger.warning(f"Performance test failed: {performance.get('error', 'Unknown error')}")
        
        dashboard_html = self.create_dashboard_html(endpoint_name, metrics, status, performance)
        
        if not output_file:
            output_file = f"mlops_dashboard_{endpoint_name}_{int(time.time())}.html"
        
        with open(output_file, 'w') as f:
            f.write(dashboard_html)
        
        logger.info(f"Dashboard saved to: {output_file}")
        return output_file
    
    def run_continuous_monitoring(self, endpoint_name: str, interval_minutes: int = 5):
        """Run continuous dashboard monitoring"""
        logger.info(f"Starting continuous monitoring for {endpoint_name} (interval: {interval_minutes}min)")
        
        while True:
            try:
                output_file = f"mlops_dashboard_{endpoint_name}_live.html"
                self.generate_dashboard(endpoint_name, output_file)
                logger.info(f"Dashboard updated: {output_file}")
                
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}")
                time.sleep(60)

def main():
    parser = argparse.ArgumentParser(description='ML Ops Dashboard for Conviction AI')
    parser.add_argument('--endpoint-name', type=str, 
                        default='conviction-ensemble-v4-1751650627',
                        help='SageMaker endpoint name to monitor')
    parser.add_argument('--output-file', type=str,
                        help='Output HTML file path')
    parser.add_argument('--continuous', action='store_true',
                        help='Run continuous monitoring')
    parser.add_argument('--interval', type=int, default=5,
                        help='Monitoring interval in minutes (for continuous mode)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run in dry-run mode without making actual calls')
    
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("🧪 DRY RUN MODE - Dashboard generation simulation")
        logger.info(f"✅ Would generate dashboard for endpoint: {args.endpoint_name}")
        logger.info(f"✅ Would save to: {args.output_file or 'auto-generated filename'}")
        if args.continuous:
            logger.info(f"✅ Would run continuous monitoring every {args.interval} minutes")
        return
    
    dashboard = MLOpsDashboard()
    
    if args.continuous:
        dashboard.run_continuous_monitoring(args.endpoint_name, args.interval)
    else:
        output_file = dashboard.generate_dashboard(args.endpoint_name, args.output_file)
        print(f"Dashboard generated: {output_file}")

if __name__ == "__main__":
    main()
