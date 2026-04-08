#!/usr/bin/env python3
"""
n8n Workflow Testing Script

Tests all n8n automation workflows for the UI-UX Eval Dashboard.

Usage:
    python test_workflows.py --n8n-url http://localhost:5678 --api-key your-key
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional

try:
    import requests
except ImportError:
    print("Error: requests library required. Install with: pip install requests")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class N8NWorkflowTester:
    """Test n8n workflows for the Eval Dashboard."""

    def __init__(
        self,
        n8n_url: str,
        dashboard_url: str,
        api_key: str,
        webhook_path: str = "/webhook",
    ):
        self.n8n_url = n8n_url.rstrip("/")
        self.dashboard_url = dashboard_url.rstrip("/")
        self.api_key = api_key
        self.webhook_path = webhook_path
        self.results = []

    def test_scheduled_evaluations(self) -> Dict[str, Any]:
        """Test the scheduled evaluations workflow."""
        logger.info("Testing: Scheduled Evaluations")

        result = {
            "test": "Scheduled Evaluations",
            "status": "pending",
            "steps": [],
        }

        # Step 1: Check workflow exists
        try:
            response = requests.get(
                f"{self.n8n_url}/api/v1/workflows",
                headers={"X-N8N-API-Key": self.api_key},
                timeout=10,
            )
            workflows = response.json().get("data", [])
            scheduled_wf = next(
                (w for w in workflows if "scheduled" in w.get("name", "").lower()),
                None,
            )

            if scheduled_wf:
                result["steps"].append({
                    "step": "Workflow exists",
                    "status": "pass",
                    "details": f"Found: {scheduled_wf['name']}",
                })
            else:
                result["steps"].append({
                    "step": "Workflow exists",
                    "status": "fail",
                    "details": "Scheduled evaluations workflow not found",
                })
                result["status"] = "fail"
                return result

        except Exception as e:
            result["steps"].append({
                "step": "Workflow exists",
                "status": "fail",
                "details": str(e),
            })
            result["status"] = "fail"
            return result

        # Step 2: Test models API endpoint
        try:
            response = requests.get(
                f"{self.dashboard_url}/api/v1/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10,
            )
            if response.status_code == 200:
                result["steps"].append({
                    "step": "Models API accessible",
                    "status": "pass",
                    "details": f"Found {len(response.json().get('data', []))} models",
                })
            else:
                result["steps"].append({
                    "step": "Models API accessible",
                    "status": "fail",
                    "details": f"HTTP {response.status_code}",
                })
        except Exception as e:
            result["steps"].append({
                "step": "Models API accessible",
                "status": "fail",
                "details": str(e),
            })

        # Step 3: Test runs API endpoint
        try:
            response = requests.get(
                f"{self.dashboard_url}/api/v1/runs/stats",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10,
            )
            if response.status_code == 200:
                result["steps"].append({
                    "step": "Runs API accessible",
                    "status": "pass",
                    "details": "Stats endpoint working",
                })
            else:
                result["steps"].append({
                    "step": "Runs API accessible",
                    "status": "fail",
                    "details": f"HTTP {response.status_code}",
                })
        except Exception as e:
            result["steps"].append({
                "step": "Runs API accessible",
                "status": "fail",
                "details": str(e),
            })

        result["status"] = "pass" if all(
            s["status"] == "pass" for s in result["steps"]
        ) else "partial"
        return result

    def test_evaluation_notifications(self) -> Dict[str, Any]:
        """Test the evaluation notifications workflow."""
        logger.info("Testing: Evaluation Notifications")

        result = {
            "test": "Evaluation Notifications",
            "status": "pending",
            "steps": [],
        }

        # Test webhook endpoint
        webhook_url = f"{self.n8n_url}{self.webhook_path}/evaluation-complete"

        test_payload = {
            "run": {
                "id": f"test-{int(time.time())}",
                "status": "completed",
                "run_type": "benchmark",
                "model_id": "test-model",
                "build_name": "test-build",
                "started_at": datetime.utcnow().isoformat(),
                "completed_at": datetime.utcnow().isoformat(),
                "duration_seconds": 120,
            },
            "model": {
                "id": "test-model",
                "name": "Test Model",
                "checkpoint": "test/checkpoint",
            },
            "recipients": {
                "email": ["test@example.com"],
                "slack": ["#test-channel"],
                "teams": ["Test Team"],
            },
        }

        try:
            response = requests.post(
                webhook_url,
                json=test_payload,
                timeout=30,
            )

            if response.status_code in [200, 202]:
                result["steps"].append({
                    "step": "Webhook triggered",
                    "status": "pass",
                    "details": f"Response: {response.status_code}",
                })
            else:
                result["steps"].append({
                    "step": "Webhook triggered",
                    "status": "fail",
                    "details": f"HTTP {response.status_code}",
                })
        except Exception as e:
            result["steps"].append({
                "step": "Webhook triggered",
                "status": "fail",
                "details": str(e),
            })

        result["status"] = "pass" if all(
            s["status"] == "pass" for s in result["steps"]
        ) else "fail"
        return result

    def test_anomaly_detection(self) -> Dict[str, Any]:
        """Test the anomaly detection workflow."""
        logger.info("Testing: Anomaly Detection")

        result = {
            "test": "Anomaly Detection",
            "status": "pending",
            "steps": [],
        }

        # Step 1: Test metrics trends endpoint
        try:
            response = requests.get(
                f"{self.dashboard_url}/api/v1/metrics/trends",
                params={"model_id": "test", "metric_name": "seconds_to_first_token"},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10,
            )
            if response.status_code == 200:
                result["steps"].append({
                    "step": "Metrics trends API",
                    "status": "pass",
                    "details": "Endpoint working",
                })
            else:
                result["steps"].append({
                    "step": "Metrics trends API",
                    "status": "fail",
                    "details": f"HTTP {response.status_code}",
                })
        except Exception as e:
            result["steps"].append({
                "step": "Metrics trends API",
                "status": "fail",
                "details": str(e),
            })

        # Step 2: Test metrics aggregate endpoint
        try:
            response = requests.get(
                f"{self.dashboard_url}/api/v1/metrics/aggregate",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10,
            )
            if response.status_code == 200:
                result["steps"].append({
                    "step": "Metrics aggregate API",
                    "status": "pass",
                    "details": "Endpoint working",
                })
            else:
                result["steps"].append({
                    "step": "Metrics aggregate API",
                    "status": "fail",
                    "details": f"HTTP {response.status_code}",
                })
        except Exception as e:
            result["steps"].append({
                "step": "Metrics aggregate API",
                "status": "fail",
                "details": str(e),
            })

        result["status"] = "pass" if all(
            s["status"] == "pass" for s in result["steps"]
        ) else "fail"
        return result

    def test_report_generation(self) -> Dict[str, Any]:
        """Test the weekly/monthly report workflow."""
        logger.info("Testing: Report Generation")

        result = {
            "test": "Report Generation",
            "status": "pending",
            "steps": [],
        }

        # Test run stats (primary data source)
        try:
            response = requests.get(
                f"{self.dashboard_url}/api/v1/runs/stats",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10,
            )
            if response.status_code == 200:
                stats = response.json().get("data", {})
                result["steps"].append({
                    "step": "Run stats available",
                    "status": "pass",
                    "details": f"Total runs: {stats.get('total_runs', 0)}",
                })
            else:
                result["steps"].append({
                    "step": "Run stats available",
                    "status": "fail",
                    "details": f"HTTP {response.status_code}",
                })
        except Exception as e:
            result["steps"].append({
                "step": "Run stats available",
                "status": "fail",
                "details": str(e),
            })

        result["status"] = "pass" if all(
            s["status"] == "pass" for s in result["steps"]
        ) else "fail"
        return result

    def test_model_comparison(self) -> Dict[str, Any]:
        """Test the model comparison workflow."""
        logger.info("Testing: Model Comparison")

        result = {
            "test": "Model Comparison",
            "status": "pending",
            "steps": [],
        }

        # Test webhook endpoint
        webhook_url = f"{self.n8n_url}{self.webhook_path}/compare-models"

        test_payload = {
            "model_ids": ["model-1", "model-2"],
            "categories": ["performance", "accuracy"],
            "requested_by": "test@example.com",
        }

        try:
            response = requests.post(
                webhook_url,
                json=test_payload,
                timeout=30,
            )

            if response.status_code in [200, 202]:
                result["steps"].append({
                    "step": "Comparison webhook",
                    "status": "pass",
                    "details": f"Response: {response.status_code}",
                })
            else:
                result["steps"].append({
                    "step": "Comparison webhook",
                    "status": "fail",
                    "details": f"HTTP {response.status_code}",
                })
        except Exception as e:
            result["steps"].append({
                "step": "Comparison webhook",
                "status": "fail",
                "details": str(e),
            })

        result["status"] = "pass" if all(
            s["status"] == "pass" for s in result["steps"]
        ) else "fail"
        return result

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return summary."""
        logger.info("Starting n8n workflow tests...")

        tests = [
            self.test_scheduled_evaluations,
            self.test_evaluation_notifications,
            self.test_anomaly_detection,
            self.test_report_generation,
            self.test_model_comparison,
        ]

        for test in tests:
            try:
                result = test()
                self.results.append(result)
            except Exception as e:
                logger.error(f"Test failed with error: {e}")
                self.results.append({
                    "test": test.__name__,
                    "status": "error",
                    "error": str(e),
                })

        # Summary
        passed = sum(1 for r in self.results if r.get("status") == "pass")
        failed = sum(1 for r in self.results if r.get("status") == "fail")
        partial = sum(1 for r in self.results if r.get("status") == "partial")

        return {
            "summary": {
                "total": len(self.results),
                "passed": passed,
                "failed": failed,
                "partial": partial,
                "timestamp": datetime.utcnow().isoformat(),
            },
            "results": self.results,
        }


def main():
    parser = argparse.ArgumentParser(description="Test n8n workflows")
    parser.add_argument(
        "--n8n-url",
        type=str,
        default=os.environ.get("N8N_URL", "http://localhost:5678"),
        help="n8n base URL",
    )
    parser.add_argument(
        "--dashboard-url",
        type=str,
        default=os.environ.get("DASHBOARD_API_URL", "http://localhost:8000"),
        help="Dashboard API URL",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="API key for authentication",
    )
    parser.add_argument(
        "--output",
        type=str,
        choices=["json", "text"],
        default="text",
        help="Output format",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    tester = N8NWorkflowTester(
        n8n_url=args.n8n_url,
        dashboard_url=args.dashboard_url,
        api_key=args.api_key,
    )

    results = tester.run_all_tests()

    if args.output == "json":
        print(json.dumps(results, indent=2))
    else:
        # Text output
        print("\n" + "=" * 60)
        print("n8n Workflow Test Results")
        print("=" * 60)

        summary = results["summary"]
        print(f"\nTotal: {summary['total']} | Passed: {summary['passed']} | "
              f"Failed: {summary['failed']} | Partial: {summary['partial']}")
        print(f"Timestamp: {summary['timestamp']}")
        print("\n" + "-" * 60)

        for result in results["results"]:
            status_icon = {
                "pass": "✅",
                "fail": "❌",
                "partial": "⚠️",
                "error": "🔴",
            }.get(result.get("status", "unknown"), "❓")

            print(f"\n{status_icon} {result['test']}: {result.get('status', 'unknown').upper()}")

            for step in result.get("steps", []):
                step_icon = "✓" if step["status"] == "pass" else "✗"
                print(f"   {step_icon} {step['step']}: {step.get('details', '')}")

        print("\n" + "=" * 60)

    # Exit code
    sys.exit(0 if results["summary"]["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
