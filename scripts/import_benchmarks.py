"""
Import benchmark results into the dashboard database.
"""

import json
import hashlib
import hmac
from datetime import datetime
from pathlib import Path
import requests


def generate_signature(payload: str, secret: str) -> str:
    """Generate HMAC-SHA256 signature."""
    return hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256,
    ).hexdigest()


def import_benchmark_results(
    results_file: str = "benchmark_results.json",
    dashboard_url: str = "http://localhost:8000",
    cli_secret: str = "test-cli-secret",
):
    """Import benchmark results into dashboard."""

    # Load results
    with open(results_file, "r") as f:
        results = json.load(f)

    print(f"Loading results from {results_file}")
    print(f"Models: {list(results.keys())}")

    # Import each model's results
    for model_name, model_results in results.items():
        print(f"\n{'='*60}")
        print(f"Importing: {model_name}")
        print(f"{'='*60}")

        # Find best metrics across all prompt lengths
        best_tps = 0
        best_tps_prompt = ""
        best_ttft = float('inf')
        best_ttft_prompt = ""

        for prompt_key, data in model_results.items():
            tps_stats = data.get("tps_stats", {})
            ttft_stats = data.get("ttft_stats", {})

            mean_tps = tps_stats.get("mean", 0)
            mean_ttft = ttft_stats.get("mean", 0)

            if mean_tps > best_tps:
                best_tps = mean_tps
                best_tps_prompt = prompt_key

            if mean_ttft > 0 and mean_ttft < best_ttft:
                best_ttft = mean_ttft
                best_ttft_prompt = prompt_key

        # Prepare payload for dashboard API
        build_name = f"{model_name.replace('/', '_').replace(':', '_')}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        payload = {
            "model_id": model_name,
            "run_type": "benchmark",
            "build_name": build_name,
            "config": {
                "iterations": 5,
                "output_tokens": 32,
                "prompt_lengths": [64, 128, 256],
            },
            "device": "gpu",
            "backend": "llamacpp",
            "dtype": "int4",
            "metrics": [],
            "status": "completed",
            "started_at": datetime.now().isoformat(),
            "completed_at": datetime.now().isoformat(),
            "duration_seconds": sum(
                data.get("ttft_stats", {}).get("mean", 0)
                for data in model_results.values()
            ),
        }

        # Add best TPS metric
        if best_tps > 0:
            payload["metrics"].append({
                "name": "token_generation_tokens_per_second",
                "value": round(best_tps, 2),
                "unit": "tokens/s",
                "category": "performance",
                "display_name": "Token Generation Speed",
            })
            print(f"  Best TPS: {best_tps:.2f} tok/s ({best_tps_prompt})")

        # Add best TTFT metric
        if best_ttft > 0 and best_ttft != float('inf'):
            payload["metrics"].append({
                "name": "seconds_to_first_token",
                "value": round(best_ttft, 4),
                "unit": "seconds",
                "category": "performance",
                "display_name": "Time to First Token",
            })
            print(f"  Best TTFT: {best_ttft:.4f}s ({best_ttft_prompt})")

        # Add per-prompt-length metrics
        for prompt_key, data in model_results.items():
            tps_stats = data.get("tps_stats", {})
            ttft_stats = data.get("ttft_stats", {})

            if tps_stats.get("mean", 0) > 0:
                payload["metrics"].append({
                    "name": f"tps_{prompt_key}",
                    "value": round(tps_stats["mean"], 2),
                    "unit": "tokens/s",
                    "category": "performance",
                    "display_name": f"TPS ({prompt_key})",
                })

        # Generate signature
        payload_json = json.dumps(payload)
        signature = generate_signature(payload_json, cli_secret)

        # Try to upload to dashboard
        url = f"{dashboard_url}/api/v1/import/evaluation"
        headers = {
            "Content-Type": "application/json",
            "X-CLI-Signature": signature,
        }

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            if resp.status_code in [200, 201]:
                result = resp.json()
                print(f"  Upload successful: {result}")
            else:
                print(f"  Upload failed: {resp.status_code} - {resp.text[:200]}")
        except Exception as e:
            print(f"  Upload error: {e}")

        # Also save as YAML for manual import
        yaml_data = {
            "checkpoint": model_name,
            "device": "gpu",
            "backend": "llamacpp",
            "dtype": "int4",
            "iterations": 5,
            "output_tokens": 32,
            "timestamp": datetime.now().isoformat(),
            "token_generation_tokens_per_second": round(best_tps, 2),
            "seconds_to_first_token": round(best_ttft, 4),
            "std_dev_tokens_per_second": round(
                model_results.get(best_tps_prompt, {}).get("tps_stats", {}).get("std_dev", 0),
                2,
            ),
        }

        print(f"  Data prepared for import")

    print(f"\n{'='*60}")
    print("IMPORT SUMMARY")
    print(f"{'='*60}")

    # Print comparison
    print("\nModel Comparison (Best TPS):")
    for model_name, model_results in results.items():
        best_tps = max(
            data.get("tps_stats", {}).get("mean", 0)
            for data in model_results.values()
        )
        print(f"  {model_name}: {best_tps:.2f} tok/s")

    # Determine winner
    winner = max(
        results.items(),
        key=lambda x: max(
            data.get("tps_stats", {}).get("mean", 0)
            for data in x[1].values()
        ),
    )
    print(f"\nWinner: {winner[0]} with {max(data.get('tps_stats', {}).get('mean', 0) for data in winner[1].values()):.2f} tok/s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Import benchmark results into dashboard")
    parser.add_argument(
        "--results-file",
        default="benchmark_results.json",
        help="Path to benchmark results JSON file",
    )
    parser.add_argument(
        "--dashboard-url",
        default="http://localhost:8000",
        help="Dashboard URL",
    )
    parser.add_argument(
        "--cli-secret",
        default="test-cli-secret",
        help="CLI secret for signature",
    )

    args = parser.parse_args()
    import_benchmark_results(
        results_file=args.results_file,
        dashboard_url=args.dashboard_url,
        cli_secret=args.cli_secret,
    )
