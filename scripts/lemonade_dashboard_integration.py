"""
Lemonade Dashboard Integration Script

This script is called by the lemonade-eval CLI to upload evaluation results
to the dashboard backend.

Usage:
    python lemonade_dashboard_integration.py \
        --dashboard-url https://dashboard.example.com \
        --api-key your-api-key \
        --yaml-path /path/to/lemonade_stats.yaml

Features:
- Upload evaluation results to dashboard
- Real-time progress reporting via WebSocket
- Retry with exponential backoff
- Offline queue for failed uploads
"""

import argparse
import asyncio
import hashlib
import hmac
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Try to import aiohttp for async HTTP
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    import requests

# Try to import websockets for progress reporting
try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False


class DashboardUploader:
    """Upload evaluation results to the dashboard."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        skip_verify: bool = False,
        max_retries: int = 3,
        timeout: int = 60,
    ):
        """
        Initialize dashboard uploader.

        Args:
            base_url: Dashboard API base URL
            api_key: API key for authentication
            skip_verify: Skip SSL verification
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.skip_verify = skip_verify
        self.max_retries = max_retries
        self.timeout = timeout
        self._session = None
        self._ws = None
        self._offline_queue: List[Dict] = []

    async def __aenter__(self):
        """Async context manager entry."""
        if HAS_AIOHTTP:
            connector = aiohttp.TCPConnector(ssl=not self.skip_verify)
            self._session = aiohttp.ClientSession(
                connector=connector,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._ws:
            await self._ws.close()
        if self._session:
            await self._session.close()

    def _generate_signature(self, payload: str) -> str:
        """Generate request signature for verification."""
        secret = self.api_key
        return hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()

    async def upload_evaluation(
        self,
        yaml_data: Dict[str, Any],
        build_name: str,
    ) -> Dict[str, Any]:
        """
        Upload evaluation results to dashboard.

        Args:
            yaml_data: Parsed YAML data
            build_name: Build name

        Returns:
            Upload result
        """
        # Prepare request payload
        payload = {
            "model_id": yaml_data.get("checkpoint", "unknown"),
            "run_type": self._determine_run_type(yaml_data),
            "build_name": build_name,
            "config": {
                "iterations": yaml_data.get("iterations"),
                "prompts": yaml_data.get("prompts"),
                "output_tokens": yaml_data.get("output_tokens"),
            },
            "device": yaml_data.get("device"),
            "backend": yaml_data.get("backend"),
            "dtype": yaml_data.get("dtype"),
            "metrics": self._extract_metrics(yaml_data),
            "status": "completed",
            "started_at": yaml_data.get("timestamp"),
        }

        # Remove None values from config
        payload["config"] = {k: v for k, v in payload["config"].items() if v is not None}

        # Upload with retry
        for attempt in range(self.max_retries):
            try:
                result = await self._post_json("/api/v1/import/evaluation", payload)
                print(f"Successfully uploaded evaluation: {build_name}")
                return result
            except Exception as e:
                if attempt == self.max_retries - 1:
                    # Queue for offline retry
                    self._offline_queue.append(payload)
                    raise
                wait_time = 2 ** attempt
                print(f"Upload failed, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)

        return {"success": False, "error": "Max retries exceeded"}

    async def _post_json(self, endpoint: str, data: Dict) -> Dict:
        """POST JSON to endpoint with retry."""
        url = f"{self.base_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if HAS_AIOHTTP:
            async with self._session.post(url, json=data, headers=headers) as resp:
                resp.raise_for_status()
                return await resp.json()
        else:
            response = requests.post(
                url,
                json=data,
                headers=headers,
                timeout=self.timeout,
                verify=not self.skip_verify,
            )
            response.raise_for_status()
            return response.json()

    def _determine_run_type(self, yaml_data: Dict[str, Any]) -> str:
        """Determine run type from YAML data."""
        if any(k.startswith("mmlu_") for k in yaml_data.keys()):
            return "accuracy-mmlu"
        elif any(k.startswith("humaneval_") for k in yaml_data.keys()):
            return "accuracy-humaneval"
        elif any(k.startswith("lm_eval_") for k in yaml_data.keys()):
            return "lm-eval"
        elif "perplexity" in yaml_data:
            return "perplexity"
        else:
            return "benchmark"

    def _extract_metrics(self, yaml_data: Dict[str, Any]) -> List[Dict]:
        """Extract metrics from YAML data."""
        metrics = []

        # Performance metrics
        perf_metrics = {
            "seconds_to_first_token": ("seconds", "performance"),
            "token_generation_tokens_per_second": ("tokens/s", "performance"),
            "max_memory_used_gbyte": ("GB", "performance"),
        }

        for name, (unit, category) in perf_metrics.items():
            if name in yaml_data and yaml_data[name] is not None:
                metrics.append({
                    "name": name,
                    "value": float(yaml_data[name]),
                    "unit": unit,
                    "category": category,
                })

        # Accuracy metrics
        for key, value in yaml_data.items():
            if key.startswith(("mmlu_", "humaneval_", "lm_eval_")):
                if isinstance(value, (int, float)):
                    metrics.append({
                        "name": key,
                        "value": float(value),
                        "unit": "%",
                        "category": "accuracy",
                    })

        return metrics


def load_yaml_file(yaml_path: str) -> Dict[str, Any]:
    """Load YAML file."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Upload lemonade-eval results to dashboard"
    )
    parser.add_argument(
        "--dashboard-url",
        type=str,
        required=True,
        help="Dashboard API URL (e.g., https://dashboard.example.com)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="API key for dashboard authentication",
    )
    parser.add_argument(
        "--yaml-path",
        type=str,
        required=True,
        help="Path to lemonade_stats.yaml file",
    )
    parser.add_argument(
        "--build-name",
        type=str,
        default=None,
        help="Build name (default: parent directory name)",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip SSL certificate verification",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Load YAML file
    print(f"Loading YAML from: {args.yaml_path}")
    try:
        yaml_data = load_yaml_file(args.yaml_path)
    except Exception as e:
        print(f"Error loading YAML: {e}")
        sys.exit(1)

    # Determine build name
    build_name = args.build_name or Path(args.yaml_path).parent.name
    print(f"Build name: {build_name}")

    # Upload to dashboard
    print(f"Uploading to dashboard: {args.dashboard_url}")

    async def upload():
        async with DashboardUploader(
            base_url=args.dashboard_url,
            api_key=args.api_key,
            skip_verify=args.skip_verify,
        ) as uploader:
            result = await uploader.upload_evaluation(
                yaml_data=yaml_data,
                build_name=build_name,
            )
            return result

    try:
        result = asyncio.run(upload())
        print(f"Upload result: {json.dumps(result, indent=2)}")

        if result.get("success"):
            print("Upload completed successfully!")
            sys.exit(0)
        else:
            print(f"Upload failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)

    except Exception as e:
        print(f"Upload error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
