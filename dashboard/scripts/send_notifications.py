"""
Evaluation Notification Service

Script to send evaluation completion notifications via email and webhooks.

Features:
- Email notifications (SMTP)
- Slack/Teams webhook notifications
- In-app notification queuing
- Configurable notification triggers

Usage:
    python send_notifications.py --event <type> --recipient <email> --data <json>
"""

import argparse
import json
import logging
import os
import smtplib
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class NotificationService:
    """Send notifications through various channels."""

    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: int = 587,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        webhook_urls: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize notification service.

        Args:
            smtp_host: SMTP server host
            smtp_port: SMTP server port
            smtp_user: SMTP username
            smtp_password: SMTP password
            webhook_urls: Dict of channel name to webhook URL
        """
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.webhook_urls = webhook_urls or {}

    def send_evaluation_complete(
        self,
        recipient: str,
        run_id: str,
        model_name: str,
        run_type: str,
        status: str,
        dashboard_url: str,
        message: Optional[str] = None,
    ) -> Dict[str, bool]:
        """
        Send notification when evaluation completes.

        Args:
            recipient: Email address
            run_id: Run ID
            model_name: Model name
            run_type: Evaluation type
            status: Run status (completed, failed)
            dashboard_url: Dashboard URL
            message: Optional status message

        Returns:
            Dict of channel -> success status
        """
        results = {}

        # Email notification
        if self.smtp_host:
            subject = f"Evaluation {'Completed' if status == 'completed' else 'Failed'}: {model_name}"
            body = self._format_evaluation_email(
                run_id=run_id,
                model_name=model_name,
                run_type=run_type,
                status=status,
                dashboard_url=dashboard_url,
                message=message,
            )
            results["email"] = self.send_email(recipient, subject, body)

        # Slack notification
        if "slack" in self.webhook_urls:
            results["slack"] = self.send_slack_webhook(
                self.webhook_urls["slack"],
                event_type="evaluation_complete",
                data={
                    "run_id": run_id,
                    "model_name": model_name,
                    "status": status,
                    "dashboard_url": dashboard_url,
                },
            )

        # Teams notification
        if "teams" in self.webhook_urls:
            results["teams"] = self.send_teams_webhook(
                self.webhook_urls["teams"],
                event_type="evaluation_complete",
                data={
                    "run_id": run_id,
                    "model_name": model_name,
                    "status": status,
                    "dashboard_url": dashboard_url,
                },
            )

        return results

    def send_trend_alert(
        self,
        recipient: str,
        model_name: str,
        metric_name: str,
        trend: str,
        change_percent: float,
        dashboard_url: str,
    ) -> Dict[str, bool]:
        """
        Send alert for significant metric trends.

        Args:
            recipient: Email address
            model_name: Model name
            metric_name: Metric name
            trend: Trend direction
            change_percent: Percentage change
            dashboard_url: Dashboard URL

        Returns:
            Dict of channel -> success status
        """
        results = {}

        # Email notification
        if self.smtp_host:
            subject = f"Performance Alert: {model_name} {metric_name}"
            body = self._format_trend_alert_email(
                model_name=model_name,
                metric_name=metric_name,
                trend=trend,
                change_percent=change_percent,
                dashboard_url=dashboard_url,
            )
            results["email"] = self.send_email(recipient, subject, body)

        # Slack notification
        if "slack" in self.webhook_urls:
            results["slack"] = self.send_slack_webhook(
                self.webhook_urls["slack"],
                event_type="trend_alert",
                data={
                    "model_name": model_name,
                    "metric_name": metric_name,
                    "trend": trend,
                    "change_percent": change_percent,
                },
            )

        return results

    def send_report_ready(
        self,
        recipient: str,
        report_name: str,
        download_url: str,
        expiry_days: int = 7,
    ) -> Dict[str, bool]:
        """
        Send notification when report is ready.

        Args:
            recipient: Email address
            report_name: Report name
            download_url: Download URL
            expiry_days: Link expiry in days

        Returns:
            Dict of channel -> success status
        """
        results = {}

        # Email notification
        if self.smtp_host:
            subject = f"Report Ready: {report_name}"
            body = self._format_report_email(
                report_name=report_name,
                download_url=download_url,
                expiry_days=expiry_days,
            )
            results["email"] = self.send_email(recipient, subject, body)

        return results

    def send_email(
        self,
        to_email: str,
        subject: str,
        body: str,
        html: bool = False,
    ) -> bool:
        """
        Send email notification.

        Args:
            to_email: Recipient email
            subject: Email subject
            body: Email body
            html: Whether body is HTML

        Returns:
            True if successful
        """
        if not self.smtp_host or not self.smtp_user:
            logger.info(f"Email (simulated): To {to_email}, Subject: {subject}")
            return True

        msg = MIMEMultipart("alternative")
        msg["From"] = self.smtp_user
        msg["To"] = to_email
        msg["Subject"] = subject

        # Attach body
        msg_type = "html" if html else "plain"
        msg.attach(MIMEText(body, msg_type))

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                if self.smtp_password:
                    server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            logger.info(f"Email sent to {to_email}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def send_slack_webhook(
        self,
        webhook_url: str,
        event_type: str,
        data: Dict[str, Any],
    ) -> bool:
        """
        Send Slack webhook notification.

        Args:
            webhook_url: Slack webhook URL
            event_type: Event type
            data: Event data

        Returns:
            True if successful
        """
        try:
            # Format for Slack
            payload = {
                "text": f"{event_type.replace('_', ' ').title()}",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": self._format_slack_message(event_type, data),
                        },
                    },
                ],
            }

            response = requests.post(
                webhook_url,
                json=payload,
                timeout=10,
            )
            response.raise_for_status()

            logger.info(f"Slack webhook sent: {event_type}")
            return True

        except Exception as e:
            logger.error(f"Failed to send Slack webhook: {e}")
            return False

    def send_teams_webhook(
        self,
        webhook_url: str,
        event_type: str,
        data: Dict[str, Any],
    ) -> bool:
        """
        Send Microsoft Teams webhook notification.

        Args:
            webhook_url: Teams webhook URL
            event_type: Event type
            data: Event data

        Returns:
            True if successful
        """
        try:
            # Format for Teams
            payload = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": "0076D7",
                "summary": event_type,
                "sections": [
                    {
                        "activityTitle": event_type.replace("_", " ").title(),
                        "facts": [
                            {"name": k, "value": str(v)}
                            for k, v in data.items()
                        ],
                    },
                ],
            }

            response = requests.post(
                webhook_url,
                json=payload,
                timeout=10,
            )
            response.raise_for_status()

            logger.info(f"Teams webhook sent: {event_type}")
            return True

        except Exception as e:
            logger.error(f"Failed to send Teams webhook: {e}")
            return False

    def _format_evaluation_email(
        self,
        run_id: str,
        model_name: str,
        run_type: str,
        status: str,
        dashboard_url: str,
        message: Optional[str] = None,
    ) -> str:
        """Format evaluation completion email."""
        return f"""
Evaluation Report

Model: {model_name}
Type: {run_type}
Status: {status}
Run ID: {run_id}
{f'Message: {message}' if message else ''}

View results: {dashboard_url}/runs/{run_id}

---
Lemonade Eval Dashboard
"""

    def _format_trend_alert_email(
        self,
        model_name: str,
        metric_name: str,
        trend: str,
        change_percent: float,
        dashboard_url: str,
    ) -> str:
        """Format trend alert email."""
        direction = "improvement" if "improving" in trend else "degradation"
        return f"""
Performance Trend Alert

Model: {model_name}
Metric: {metric_name}
Trend: {trend}
Change: {change_percent:.2f}%

This may indicate a {direction} in model performance.

View details: {dashboard_url}/models/{model_name}/trends

---
Lemonade Eval Dashboard
"""

    def _format_report_email(
        self,
        report_name: str,
        download_url: str,
        expiry_days: int,
    ) -> str:
        """Format report ready email."""
        return f"""
Your report is ready for download.

Report: {report_name}
Download: {download_url}

This link will expire in {expiry_days} days.

---
Lemonade Eval Dashboard
"""

    def _format_slack_message(
        self,
        event_type: str,
        data: Dict[str, Any],
    ) -> str:
        """Format message for Slack."""
        lines = [f"*{event_type.replace('_', ' ').title()}*"]
        for key, value in data.items():
            lines.append(f"{key.replace('_', ' ').title()}: {value}")
        return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Send evaluation notifications"
    )
    parser.add_argument(
        "--event",
        type=str,
        required=True,
        choices=["evaluation_complete", "trend_alert", "report_ready"],
        help="Event type",
    )
    parser.add_argument(
        "--recipient",
        type=str,
        required=True,
        help="Recipient email address",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Event data as JSON",
    )
    parser.add_argument(
        "--smtp-host",
        type=str,
        default=os.environ.get("SMTP_HOST"),
        help="SMTP server host",
    )
    parser.add_argument(
        "--smtp-user",
        type=str,
        default=os.environ.get("SMTP_USER"),
        help="SMTP username",
    )
    parser.add_argument(
        "--smtp-password",
        type=str,
        default=os.environ.get("SMTP_PASSWORD"),
        help="SMTP password",
    )
    parser.add_argument(
        "--slack-webhook",
        type=str,
        default=os.environ.get("SLACK_WEBHOOK_URL"),
        help="Slack webhook URL",
    )
    parser.add_argument(
        "--teams-webhook",
        type=str,
        default=os.environ.get("TEAMS_WEBHOOK_URL"),
        help="Teams webhook URL",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Parse data
    try:
        data = json.loads(args.data)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON data: {e}")
        sys.exit(1)

    # Initialize service
    webhook_urls = {}
    if args.slack_webhook:
        webhook_urls["slack"] = args.slack_webhook
    if args.teams_webhook:
        webhook_urls["teams"] = args.teams_webhook

    service = NotificationService(
        smtp_host=args.smtp_host,
        smtp_user=args.smtp_user,
        smtp_password=args.smtp_password,
        webhook_urls=webhook_urls,
    )

    # Send notification based on event type
    if args.event == "evaluation_complete":
        results = service.send_evaluation_complete(
            recipient=args.recipient,
            run_id=data.get("run_id", ""),
            model_name=data.get("model_name", ""),
            run_type=data.get("run_type", ""),
            status=data.get("status", "completed"),
            dashboard_url=data.get("dashboard_url", ""),
            message=data.get("message"),
        )
    elif args.event == "trend_alert":
        results = service.send_trend_alert(
            recipient=args.recipient,
            model_name=data.get("model_name", ""),
            metric_name=data.get("metric_name", ""),
            trend=data.get("trend", "stable"),
            change_percent=data.get("change_percent", 0),
            dashboard_url=data.get("dashboard_url", ""),
        )
    elif args.event == "report_ready":
        results = service.send_report_ready(
            recipient=args.recipient,
            report_name=data.get("report_name", ""),
            download_url=data.get("download_url", ""),
        )
    else:
        print(f"Unknown event type: {args.event}")
        sys.exit(1)

    # Output results
    print(json.dumps(results, indent=2))

    # Exit with error if any notification failed
    if not all(results.values()):
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
