#!/usr/bin/env python3
"""
Telegram alert workflow for drift, signal, and explainability alerts.
"""

import json
import os
import requests
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class AlertConfig:
    """Configuration for alert thresholds and settings."""
    drift_threshold: float = 0.1
    signal_threshold: float = 0.9
    explainability_threshold: float = 0.1
    bot_token: Optional[str] = None
    chat_id: Optional[str] = None


class TelegramAlerter:
    """Handles Telegram notifications for various alert types."""
    
    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        self.bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        
        if not self.bot_token or not self.chat_id:
            raise ValueError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be configured")
    
    def send_message(self, message: str, parse_mode: str = "Markdown") -> bool:
        """Send message to Telegram chat."""
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": parse_mode
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Failed to send Telegram alert: {e}")
            return False
    
    def drift_alert(self, drift_score: float, threshold: float, features: List[str]) -> bool:
        """Send data drift alert."""
        severity = "🔴 CRITICAL" if drift_score > threshold * 2 else "⚠️ WARNING"
        
        message = f"""
{severity} *Data Drift Detected*

📊 *Drift Score:* `{drift_score:.3f}` (threshold: `{threshold:.3f}`)
📈 *Affected Features:* {len(features)} features
🔍 *Top Drifted:* {', '.join(features[:3])}

🤖 *Pipeline:* Conviction-AI
⏰ *Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return self.send_message(message)
    
    def signal_alert(self, metric_name: str, score: float, threshold: float) -> bool:
        """Send signal quality alert."""
        message = f"""
⚠️ *Signal Quality Degraded*

📊 *Metric:* `{metric_name}`
📉 *Score:* `{score:.3f}` (threshold: `{threshold:.3f}`)
🔧 *Action:* Review signal validation

🤖 *Pipeline:* Conviction-AI
⏰ *Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return self.send_message(message)
    
    def explainability_alert(self, feature_changes: Dict[str, float], threshold: float) -> bool:
        """Send model explainability alert."""
        max_change = max(feature_changes.values())
        top_changes = sorted(feature_changes.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        
        change_text = "\n".join([f"• `{feat}`: {change:+.3f}" for feat, change in top_changes])
        
        message = f"""
🧠 *Feature Importance Drift*

📊 *Max Change:* `{max_change:.3f}`
📈 *Threshold:* `{threshold:.3f}`
🔢 *Features Changed:* {len(feature_changes)}

*Top Changes:*
{change_text}

🤖 *Pipeline:* Conviction-AI
⏰ *Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return self.send_message(message)


def check_and_alert_drift(drift_report_path: str, config: AlertConfig) -> bool:
    """Check drift report and send alert if needed."""
    try:
        with open(drift_report_path, 'r') as f:
            drift_data = json.load(f)
        
        max_drift = drift_data.get('max_drift_score', 0.0)
        drifted_features = drift_data.get('drifted_features', [])
        
        if max_drift > config.drift_threshold:
            alerter = TelegramAlerter(config.bot_token, config.chat_id)
            return alerter.drift_alert(max_drift, config.drift_threshold, drifted_features)
        
        return True
    except Exception as e:
        print(f"Failed to check drift: {e}")
        return False


def check_and_alert_signals(signal_metrics: Dict[str, float], config: AlertConfig) -> bool:
    """Check signal metrics and send alerts if needed."""
    alerter = TelegramAlerter(config.bot_token, config.chat_id)
    alerts_sent = 0
    
    for metric_name, score in signal_metrics.items():
        if score < config.signal_threshold:
            if alerter.signal_alert(metric_name, score, config.signal_threshold):
                alerts_sent += 1
    
    return alerts_sent == 0


def check_and_alert_explainability(shap_changes: Dict[str, float], config: AlertConfig) -> bool:
    """Check SHAP importance changes and send alert if needed."""
    max_change = max(abs(change) for change in shap_changes.values()) if shap_changes else 0.0
    
    if max_change > config.explainability_threshold:
        alerter = TelegramAlerter(config.bot_token, config.chat_id)
        return alerter.explainability_alert(shap_changes, config.explainability_threshold)
    
    return True


def send_message(status: str, payload: str) -> bool:
    """Simple function for shell script integration."""
    try:
        # Check for dummy/test values
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
        
        if bot_token in ['dummy', 'test', ''] or chat_id in ['dummy', 'test', '']:
            print(f"📱 [DRY RUN] Would send Telegram message:")
            print(f"Status: {status}")
            print(f"Payload: {payload}")
            return True
            
        alerter = TelegramAlerter()
        message = f"🤖 *{status}*\n\n{payload}"
        return alerter.send_message(message)
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")
        return False


def main():
    """CLI for testing Telegram alerts."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Telegram alerts")
    parser.add_argument("--type", choices=["drift", "signal", "explainability"], required=True)
    parser.add_argument("--bot-token", help="Telegram bot token")
    parser.add_argument("--chat-id", help="Telegram chat ID")
    
    args = parser.parse_args()
    
    config = AlertConfig(bot_token=args.bot_token, chat_id=args.chat_id)
    alerter = TelegramAlerter(config.bot_token, config.chat_id)
    
    if args.type == "drift":
        success = alerter.drift_alert(0.15, 0.1, ["feature1", "feature2", "feature3"])
    elif args.type == "signal":
        success = alerter.signal_alert("gamma_coverage", 0.85, 0.9)
    elif args.type == "explainability":
        changes = {"feature1": 0.15, "feature2": -0.12, "feature3": 0.08}
        success = alerter.explainability_alert(changes, 0.1)
    
    if success:
        print("✅ Alert sent successfully")
    else:
        print("❌ Failed to send alert")


if __name__ == "__main__":
    main()