#!/usr/bin/env python3
"""
Setup Sentry configuration for error tracking and performance monitoring
"""
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description="Setup Sentry configuration")
    parser.add_argument("--dsn", required=True, help="Sentry DSN URL")
    parser.add_argument("--environment", default="production", help="Environment name")
    parser.add_argument("--release", default="v1.0.0", help="Release version")
    parser.add_argument(
        "--traces-sample-rate",
        type=float,
        default=0.1,
        help="Traces sample rate (0.0-1.0)",
    )
    parser.add_argument(
        "--profiles-sample-rate",
        type=float,
        default=0.1,
        help="Profiles sample rate (0.0-1.0)",
    )
    parser.add_argument("--output-env", help="Output environment file path")
    parser.add_argument("--output-k8s", help="Output Kubernetes secret YAML")

    args = parser.parse_args()

    # Validate sample rates
    if not (0.0 <= args.traces_sample_rate <= 1.0):
        raise ValueError("Traces sample rate must be between 0.0 and 1.0")
    if not (0.0 <= args.profiles_sample_rate <= 1.0):
        raise ValueError("Profiles sample rate must be between 0.0 and 1.0")

    config = {
        "SENTRY_DSN": args.dsn,
        "SENTRY_TRACES_SAMPLE_RATE": str(args.traces_sample_rate),
        "SENTRY_PROFILES_SAMPLE_RATE": str(args.profiles_sample_rate),
        "ENVIRONMENT": args.environment,
        "RELEASE": args.release,
    }

    print("🔧 Sentry Configuration:")
    print(f"DSN: {args.dsn}")
    print(f"Environment: {args.environment}")
    print(f"Release: {args.release}")
    print(f"Traces Sample Rate: {args.traces_sample_rate}")
    print(f"Profiles Sample Rate: {args.profiles_sample_rate}")

    # Output environment file
    if args.output_env:
        with open(args.output_env, "w") as f:
            for key, value in config.items():
                f.write(f"{key}={value}\n")
        print(f"📝 Environment file written to: {args.output_env}")

    # Output Kubernetes secret YAML
    if args.output_k8s:
        import base64

        k8s_secret = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {"name": "sentry-config", "namespace": "default"},
            "type": "Opaque",
            "data": {},
        }

        # Base64 encode values for Kubernetes secret
        for key, value in config.items():
            encoded_value = base64.b64encode(value.encode()).decode()
            k8s_secret["data"][key.lower().replace("_", "-")] = encoded_value

        with open(args.output_k8s, "w") as f:
            import yaml

            yaml.dump(k8s_secret, f, default_flow_style=False)

        print(f"📝 Kubernetes secret written to: {args.output_k8s}")

    # Test Sentry connection
    print("\n🧪 Testing Sentry connection...")
    try:
        import sentry_sdk
        from sentry_sdk.integrations.logging import LoggingIntegration

        sentry_sdk.init(
            dsn=args.dsn,
            integrations=[LoggingIntegration(level=None, event_level=None)],
            traces_sample_rate=args.traces_sample_rate,
            profiles_sample_rate=args.profiles_sample_rate,
            environment=args.environment,
            release=args.release,
            debug=True,
        )

        # Send test message
        sentry_sdk.capture_message("Sentry configuration test", level="info")
        print("✅ Sentry connection test successful!")
        print("Check your Sentry dashboard for the test message.")

    except Exception as e:
        print(f"❌ Sentry connection test failed: {str(e)}")
        return 1

    print("\n📚 Next steps:")
    print("1. Set environment variables in your deployment")
    print("2. Restart your FastAPI service")
    print("3. Check Sentry dashboard for incoming events")
    print("4. Configure alerts and notifications in Sentry")

    return 0


if __name__ == "__main__":
    exit(main())
