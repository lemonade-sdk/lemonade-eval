"""
Evaluation Trends Analyzer

Script to analyze evaluation trends and detect anomalies.

Features:
- Trend analysis using linear regression
- Anomaly detection using statistical methods
- Performance degradation alerts
- Comparative analysis between periods

Usage:
    python check_trends.py --model-id <id> --metric <name> --days <n>
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


try:
    import numpy as np
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("scipy not installed, trend analysis limited")


def calculate_trend(
    metrics: List[Dict[str, Any]],
    metric_name: str,
) -> Dict[str, Any]:
    """
    Calculate trend for a metric using linear regression.

    Args:
        metrics: List of metric data points
        metric_name: Name of the metric to analyze

    Returns:
        Trend analysis result
    """
    if len(metrics) < 2:
        return {
            "trend": "insufficient_data",
            "data_points": len(metrics),
            "metric_name": metric_name,
        }

    # Extract values and timestamps
    values = []
    timestamps = []
    base_time = min(m["timestamp"] for m in metrics)

    for m in metrics:
        if m.get("value") is not None:
            values.append(float(m["value"]))
            # Convert to days from base time
            time_diff = (m["timestamp"] - base_time).total_seconds() / 86400
            timestamps.append(time_diff)

    if len(values) < 2:
        return {
            "trend": "insufficient_data",
            "data_points": len(values),
            "metric_name": metric_name,
        }

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(timestamps, values)

    # Determine trend direction
    # For performance metrics (higher is better): positive slope = improving
    # For latency metrics (lower is better): positive slope = degrading
    is_higher_better = metric_name in [
        "token_generation_tokens_per_second",
        "prefill_tokens_per_second",
    ]

    if p_value > 0.05:
        trend = "stable"
    elif slope > 0:
        trend = "improving" if is_higher_better else "degrading"
    else:
        trend = "degrading" if is_higher_better else "improving"

    # Calculate percentage change
    if values[0] != 0:
        change_percent = ((values[-1] - values[0]) / values[0]) * 100
    else:
        change_percent = 0

    return {
        "trend": trend,
        "slope": slope,
        "r_squared": r_value ** 2,
        "p_value": p_value,
        "data_points": len(values),
        "start_value": values[0],
        "end_value": values[-1],
        "change_percent": round(change_percent, 2),
        "metric_name": metric_name,
    }


def detect_anomalies(
    metrics: List[Dict[str, Any]],
    metric_name: str,
    std_threshold: float = 2.0,
) -> List[Dict[str, Any]]:
    """
    Detect anomalies using z-score method.

    Args:
        metrics: List of metric data points
        metric_name: Name of the metric
        std_threshold: Standard deviation threshold for anomaly detection

    Returns:
        List of anomalies
    """
    if len(metrics) < 3:
        return []

    values = [float(m["value"]) for m in metrics if m.get("value") is not None]

    if len(values) < 3:
        return []

    mean = np.mean(values)
    std = np.std(values)

    if std == 0:
        return []

    anomalies = []
    for m in metrics:
        if m.get("value") is not None:
            value = float(m["value"])
            z_score = abs((value - mean) / std)

            if z_score > std_threshold:
                anomalies.append({
                    "timestamp": m["timestamp"].isoformat(),
                    "value": value,
                    "z_score": round(z_score, 2),
                    "deviation": round(value - mean, 4),
                })

    return anomalies


def compare_periods(
    metrics: List[Dict[str, Any]],
    period1_days: int = 7,
    period2_days: int = 7,
) -> Dict[str, Any]:
    """
    Compare metrics between two time periods.

    Args:
        metrics: List of metric data points
        period1_days: Days in recent period
        period2_days: Days in comparison period

    Returns:
        Comparison result
    """
    if len(metrics) < 2:
        return {"comparison": "insufficient_data"}

    now = datetime.utcnow()
    period1_start = now - timedelta(days=period1_days)
    period2_start = period1_start - timedelta(days=period2_days)

    # Split into periods
    period1_values = [
        float(m["value"]) for m in metrics
        if m["timestamp"] >= period1_start and m.get("value") is not None
    ]
    period2_values = [
        float(m["value"]) for m in metrics
        if period2_start <= m["timestamp"] < period1_start and m.get("value") is not None
    ]

    if not period1_values or not period2_values:
        return {"comparison": "insufficient_data"}

    mean1 = np.mean(period1_values)
    mean2 = np.mean(period2_values)

    if mean2 != 0:
        change_percent = ((mean1 - mean2) / mean2) * 100
    else:
        change_percent = 0

    return {
        "period1_mean": round(mean1, 4),
        "period2_mean": round(mean2, 4),
        "change_percent": round(change_percent, 2),
        "period1_count": len(period1_values),
        "period2_count": len(period2_values),
        "direction": "improvement" if change_percent > 0 else "degradation",
    }


def analyze_trends(
    model_id: str,
    metric_name: str,
    days: int = 30,
    dashboard_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze trends for a specific model and metric.

    Args:
        model_id: Model ID or checkpoint
        metric_name: Metric name to analyze
        days: Number of days to analyze
        dashboard_url: Dashboard API URL (optional)
        api_key: API key (optional)

    Returns:
        Analysis result
    """
    # Fetch metrics from dashboard API if provided
    metrics = []

    if dashboard_url and api_key:
        import requests

        try:
            response = requests.get(
                f"{dashboard_url}/api/v1/metrics/trend",
                params={
                    "model_id": model_id,
                    "metric_name": metric_name,
                    "days": days,
                },
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            metrics = data.get("data", [])

            # Parse timestamps
            for m in metrics:
                if "timestamp" in m and isinstance(m["timestamp"], str):
                    m["timestamp"] = datetime.fromisoformat(m["timestamp"])

        except Exception as e:
            logger.error(f"Failed to fetch metrics from dashboard: {e}")
            return {"error": str(e)}
    else:
        # Generate sample data for testing
        logger.info("No dashboard provided, generating sample data")
        base_value = 45.0
        now = datetime.utcnow()

        for i in range(days):
            timestamp = now - timedelta(days=i)
            # Add some noise and trend
            value = base_value + (i * 0.1) + np.random.normal(0, 1)
            metrics.append({
                "timestamp": timestamp,
                "value": value,
            })

    if not metrics:
        return {"error": "No metrics found"}

    # Perform analyses
    result = {
        "model_id": model_id,
        "metric_name": metric_name,
        "days": days,
        "data_points": len(metrics),
        "trend_analysis": calculate_trend(metrics, metric_name),
        "anomalies": detect_anomalies(metrics, metric_name),
        "period_comparison": compare_periods(metrics),
        "generated_at": datetime.utcnow().isoformat(),
    }

    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze evaluation trends"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Model ID or checkpoint",
    )
    parser.add_argument(
        "--metric",
        type=str,
        required=True,
        help="Metric name to analyze",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to analyze",
    )
    parser.add_argument(
        "--dashboard-url",
        type=str,
        default=None,
        help="Dashboard API URL",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for dashboard",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="json",
        choices=["json", "text"],
        help="Output format",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    if not HAS_SCIPY:
        print("Error: scipy is required for trend analysis")
        print("Install with: pip install scipy")
        sys.exit(1)

    # Analyze trends
    result = analyze_trends(
        model_id=args.model_id,
        metric_name=args.metric,
        days=args.days,
        dashboard_url=args.dashboard_url,
        api_key=args.api_key,
    )

    # Output result
    if args.output == "json":
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"Trend Analysis for {args.model_id} - {args.metric}")
        print(f"Data points: {result.get('data_points', 0)}")

        if "trend_analysis" in result:
            ta = result["trend_analysis"]
            print(f"Trend: {ta.get('trend', 'unknown')}")
            print(f"Change: {ta.get('change_percent', 0)}%")

        if "anomalies" in result:
            print(f"Anomalies detected: {len(result['anomalies'])}")

        if "period_comparison" in result:
            pc = result["period_comparison"]
            if pc.get("comparison") != "insufficient_data":
                print(f"Period comparison: {pc.get('change_percent', 0)}% {pc.get('direction', 'unknown')}")

    sys.exit(0)


if __name__ == "__main__":
    main()
