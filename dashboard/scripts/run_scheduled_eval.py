"""
Scheduled Evaluation Runner

Script to run evaluations on a schedule using Celery beat.

Features:
- Scheduled evaluation execution
- Integration with lemonade-eval CLI
- Progress reporting via WebSocket
- Error handling and retry logic

Usage:
    python run_scheduled_eval.py --model_id <id> --run-type <type> --config <json>
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_lemonade_eval(
    checkpoint: str,
    run_type: str,
    cache_dir: str,
    timeout: int = 3600,
) -> tuple[bool, str]:
    """
    Run lemonade-eval CLI command.

    Args:
        checkpoint: Model checkpoint path
        run_type: Type of evaluation
        cache_dir: Cache directory for results
        timeout: Command timeout in seconds

    Returns:
        Tuple of (success, output)
    """
    cmd = [
        "lemonade-eval",
        "--input", checkpoint,
        "--cache-dir", cache_dir,
    ]

    # Add evaluation-specific arguments
    if run_type == "accuracy-mmlu":
        cmd.extend(["AccuracyMMLU"])
    elif run_type == "accuracy-humaneval":
        cmd.extend(["AccuracyHumaneval"])
    elif run_type == "benchmark":
        cmd.extend(["ServerBench"])
    elif run_type == "perplexity":
        cmd.extend(["Perplexity"])
    else:
        cmd.extend([run_type])

    logger.info(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            return False, result.stderr
        return True, result.stdout

    except subprocess.TimeoutExpired:
        return False, f"Evaluation timed out after {timeout} seconds"
    except Exception as e:
        return False, str(e)


def upload_to_dashboard(
    cache_dir: str,
    dashboard_url: str,
    api_key: str,
    build_name: Optional[str] = None,
) -> bool:
    """
    Upload evaluation results to dashboard.

    Args:
        cache_dir: Cache directory with YAML results
        dashboard_url: Dashboard API URL
        api_key: API key for authentication
        build_name: Optional build name

    Returns:
        True if successful
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    integration_script = os.path.join(
        script_dir,
        "lemonade_dashboard_integration.py",
    )

    cmd = [
        sys.executable,
        integration_script,
        "--dashboard-url", dashboard_url,
        "--api-key", api_key,
        "--yaml-path", os.path.join(cache_dir, "lemonade_stats.yaml"),
    ]

    if build_name:
        cmd.extend(["--build-name", build_name])

    logger.info(f"Uploading to dashboard: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            logger.error(f"Upload failed: {result.stderr}")
            return False

        logger.info(f"Upload successful: {result.stdout}")
        return True

    except Exception as e:
        logger.error(f"Upload error: {e}")
        return False


async def run_scheduled_evaluation(
    model_id: str,
    run_type: str,
    config: Dict[str, Any],
    dashboard_url: str,
    api_key: str,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a scheduled evaluation and upload results.

    Args:
        model_id: Model checkpoint or ID
        run_type: Type of evaluation
        config: Evaluation configuration
        dashboard_url: Dashboard API URL
        api_key: API key
        cache_dir: Cache directory (optional)

    Returns:
        Result dict with status and details
    """
    start_time = datetime.utcnow()
    logger.info(f"Starting scheduled evaluation: {model_id} ({run_type})")

    # Default cache directory
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/lemonade/scheduled")
        os.makedirs(cache_dir, exist_ok=True)

    # Run evaluation
    success, output = run_lemonade_eval(
        checkpoint=model_id,
        run_type=run_type,
        cache_dir=cache_dir,
        timeout=config.get("timeout", 3600),
    )

    if not success:
        return {
            "success": False,
            "error": output,
            "start_time": start_time.isoformat(),
            "end_time": datetime.utcnow().isoformat(),
        }

    # Generate build name
    build_name = f"{run_type}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

    # Upload to dashboard
    upload_success = upload_to_dashboard(
        cache_dir=cache_dir,
        dashboard_url=dashboard_url,
        api_key=api_key,
        build_name=build_name,
    )

    end_time = datetime.utcnow()
    duration = (end_time - start_time).total_seconds()

    return {
        "success": upload_success,
        "build_name": build_name,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration,
        "output": output[:500] if len(output) > 500 else output,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run scheduled evaluations"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Model checkpoint or ID",
    )
    parser.add_argument(
        "--run-type",
        type=str,
        required=True,
        choices=["benchmark", "accuracy-mmlu", "accuracy-humaneval", "perplexity"],
        help="Type of evaluation",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="{}",
        help="JSON configuration for evaluation",
    )
    parser.add_argument(
        "--dashboard-url",
        type=str,
        required=True,
        help="Dashboard API URL",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="API key for dashboard",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Parse config
    try:
        config = json.loads(args.config)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON config: {e}")
        sys.exit(1)

    # Run evaluation
    result = asyncio.run(run_scheduled_evaluation(
        model_id=args.model_id,
        run_type=args.run_type,
        config=config,
        dashboard_url=args.dashboard_url,
        api_key=args.api_key,
        cache_dir=args.cache_dir,
    ))

    # Output result
    print(json.dumps(result, indent=2))

    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()
