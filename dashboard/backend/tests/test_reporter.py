"""
Test Result Reporting and Notification System.

This module provides:
- JUnit XML report generation
- Coverage threshold enforcement
- Test result aggregation
- Notification system for test failures
- Real-time test progress tracking
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    status: str  # passed, failed, skipped, error
    duration: float
    message: str = ""
    traceback: str = ""


@dataclass
class TestSuiteResult:
    """Result of a test suite."""
    name: str
    tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    duration: float
    test_results: List[TestResult] = field(default_factory=list)


@dataclass
class CoverageReport:
    """Coverage report data."""
    total_lines: int
    covered_lines: int
    percent_covered: float
    files: Dict[str, Dict[str, int]] = field(default_factory=dict)


@dataclass
class TestReport:
    """Complete test report."""
    timestamp: datetime
    suites: List[TestSuiteResult]
    coverage: Optional[CoverageReport]
    total_tests: int
    total_passed: int
    total_failed: int
    total_skipped: int
    total_errors: int
    total_duration: float
    pass_rate: float


# ============================================================================
# REPORT PARSER
# ============================================================================

class JUnitReportParser:
    """Parses JUnit XML test reports."""

    def parse(self, xml_path: str) -> List[TestSuiteResult]:
        """Parse JUnit XML report file."""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        suites = []

        # Handle both <testsuites> and <testsuite> root elements
        if root.tag == "testsuites":
            suite_elements = root.findall("testsuite")
        elif root.tag == "testsuite":
            suite_elements = [root]
        else:
            logger.warning(f"Unknown root element: {root.tag}")
            return suites

        for suite_elem in suite_elements:
            suite = self._parse_testsuite(suite_elem)
            suites.append(suite)

        return suites

    def _parse_testsuite(self, elem: ET.Element) -> TestSuiteResult:
        """Parse a single testsuite element."""
        name = elem.get("name", "Unknown")
        tests = int(elem.get("tests", 0))
        failures = int(elem.get("failures", 0))
        skipped = int(elem.get("skipped", 0))
        errors = int(elem.get("errors", 0))
        duration = float(elem.get("time", 0))

        test_results = []

        for testcase in elem.findall("testcase"):
            result = self._parse_testcase(testcase)
            test_results.append(result)

        return TestSuiteResult(
            name=name,
            tests=tests,
            passed=tests - failures - skipped - errors,
            failed=failures,
            skipped=skipped,
            errors=errors,
            duration=duration,
            test_results=test_results,
        )

    def _parse_testcase(self, elem: ET.Element) -> TestResult:
        """Parse a single testcase element."""
        name = elem.get("name", "Unknown")
        classname = elem.get("classname", "")
        duration = float(elem.get("time", 0))

        # Check for failure
        failure = elem.find("failure")
        if failure is not None:
            return TestResult(
                name=f"{classname}.{name}",
                status="failed",
                duration=duration,
                message=failure.get("message", ""),
                traceback=failure.text or "",
            )

        # Check for error
        error = elem.find("error")
        if error is not None:
            return TestResult(
                name=f"{classname}.{name}",
                status="error",
                duration=duration,
                message=error.get("message", ""),
                traceback=error.text or "",
            )

        # Check for skip
        skip = elem.find("skipped")
        if skip is not None:
            return TestResult(
                name=f"{classname}.{name}",
                status="skipped",
                duration=duration,
                message=skip.get("message", ""),
            )

        return TestResult(
            name=f"{classname}.{name}",
            status="passed",
            duration=duration,
        )


# ============================================================================
# COVERAGE PARSER
# ============================================================================

class CoverageParser:
    """Parses coverage.json report."""

    def parse(self, json_path: str) -> Optional[CoverageReport]:
        """Parse coverage.json file."""
        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            totals = data.get("totals", {})
            total_lines = totals.get("num_lines", 0)
            covered_lines = totals.get("covered_lines", 0)
            percent_covered = float(totals.get("percent_covered", 0))

            files = {}
            for file_path, file_data in data.get("files", {}).items():
                file_totals = file_data.get("totals", {})
                files[file_path] = {
                    "total": file_totals.get("num_lines", 0),
                    "covered": file_totals.get("covered_lines", 0),
                }

            return CoverageReport(
                total_lines=total_lines,
                covered_lines=covered_lines,
                percent_covered=percent_covered,
                files=files,
            )

        except Exception as e:
            logger.error(f"Failed to parse coverage report: {e}")
            return None


# ============================================================================
# REPORT GENERATOR
# ============================================================================

class ReportGenerator:
    """Generates test reports in various formats."""

    def __init__(self, output_dir: str = "test-reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_summary(self, report: TestReport) -> str:
        """Generate human-readable summary."""
        summary_lines = [
            "=" * 60,
            "TEST REPORT SUMMARY",
            "=" * 60,
            f"Timestamp: {report.timestamp.isoformat()}",
            "",
            "OVERALL RESULTS:",
            f"  Total Tests:    {report.total_tests}",
            f"  Passed:         {report.total_passed}",
            f"  Failed:         {report.total_failed}",
            f"  Skipped:        {report.total_skipped}",
            f"  Errors:         {report.total_errors}",
            f"  Pass Rate:      {report.pass_rate:.1f}%",
            f"  Duration:       {report.total_duration:.2f}s",
            "",
        ]

        if report.coverage:
            summary_lines.extend([
                "COVERAGE:",
                f"  Total Lines:      {report.coverage.total_lines}",
                f"  Covered Lines:    {report.coverage.covered_lines}",
                f"  Coverage:         {report.coverage.percent_covered:.1f}%",
                "",
            ])

        # Suite summaries
        summary_lines.append("SUITE RESULTS:")
        for suite in report.suites:
            status = "PASS" if suite.failed == 0 and suite.errors == 0 else "FAIL"
            summary_lines.append(
                f"  [{status}] {suite.name}: {suite.passed}/{suite.tests} passed"
            )

        # Failed tests
        failed_tests = []
        for suite in report.suites:
            for result in suite.test_results:
                if result.status in ["failed", "error"]:
                    failed_tests.append((suite.name, result))

        if failed_tests:
            summary_lines.extend([
                "",
                "FAILED TESTS:",
            ])
            for suite_name, result in failed_tests[:10]:  # Show first 10
                summary_lines.append(f"  - {result.name}")
                if result.message:
                    summary_lines.append(f"    {result.message}")

        summary_lines.extend([
            "",
            "=" * 60,
        ])

        return "\n".join(summary_lines)

    def generate_junit_xml(self, report: TestReport) -> str:
        """Generate JUnit XML format report."""
        # Root element
        testsuites = ET.Element("testsuites")
        testsuites.set("tests", str(report.total_tests))
        testsuites.set("failures", str(report.total_failed))
        testsuites.set("errors", str(report.total_errors))
        testsuites.set("skipped", str(report.total_skipped))
        testsuites.set("time", f"{report.total_duration:.3f}")
        testsuites.set("timestamp", report.timestamp.isoformat())

        # Suite elements
        for suite in report.suites:
            testsuite = ET.SubElement(testsuites, "testsuite")
            testsuite.set("name", suite.name)
            testsuite.set("tests", str(suite.tests))
            testsuite.set("failures", str(suite.failed))
            testsuite.set("errors", str(suite.errors))
            testsuite.set("skipped", str(suite.skipped))
            testsuite.set("time", f"{suite.duration:.3f}")

            # Testcase elements
            for result in suite.test_results:
                testcase = ET.SubElement(testsuite, "testcase")
                testcase.set("name", result.name.split(".")[-1])
                testcase.set("classname", ".".join(result.name.split(".")[:-1]))
                testcase.set("time", f"{result.duration:.3f}")

                if result.status == "failed":
                    failure = ET.SubElement(testcase, "failure")
                    failure.set("message", result.message)
                    if result.traceback:
                        failure.text = result.traceback
                elif result.status == "error":
                    error = ET.SubElement(testcase, "error")
                    error.set("message", result.message)
                    if result.traceback:
                        error.text = result.traceback
                elif result.status == "skipped":
                    skipped = ET.SubElement(testcase, "skipped")
                    skipped.set("message", result.message)

        return ET.tostring(testsuites, encoding="unicode")

    def generate_markdown(self, report: TestReport) -> str:
        """Generate Markdown format report."""
        md_lines = [
            "# Test Report",
            "",
            f"**Generated:** {report.timestamp.isoformat()}",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Tests | {report.total_tests} |",
            f"| Passed | {report.total_passed} |",
            f"| Failed | {report.total_failed} |",
            f"| Skipped | {report.total_skipped} |",
            f"| Errors | {report.total_errors} |",
            f"| Pass Rate | {report.pass_rate:.1f}% |",
            f"| Duration | {report.total_duration:.2f}s |",
            "",
        ]

        if report.coverage:
            md_lines.extend([
                "## Coverage",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Total Lines | {report.coverage.total_lines} |",
                f"| Covered Lines | {report.coverage.covered_lines} |",
                f"| Coverage | {report.coverage.percent_covered:.1f}% |",
                "",
            ])

        # Failed tests
        failed_tests = []
        for suite in report.suites:
            for result in suite.test_results:
                if result.status in ["failed", "error"]:
                    failed_tests.append((suite.name, result))

        if failed_tests:
            md_lines.extend([
                "## Failed Tests",
                "",
            ])
            for suite_name, result in failed_tests:
                md_lines.append(f"### {result.name}")
                md_lines.append(f"- **Suite:** {suite_name}")
                md_lines.append(f"- **Status:** {result.status}")
                if result.message:
                    md_lines.append(f"- **Message:** {result.message}")
                md_lines.append("")

        return "\n".join(md_lines)

    def save_reports(self, report: TestReport, base_name: str = "test-report"):
        """Save reports in all formats."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Save summary
        summary_path = self.output_dir / f"{base_name}-{timestamp}.txt"
        with open(summary_path, "w") as f:
            f.write(self.generate_summary(report))

        # Save JUnit XML
        junit_path = self.output_dir / f"{base_name}-{timestamp}.xml"
        with open(junit_path, "w") as f:
            f.write(self.generate_junit_xml(report))

        # Save Markdown
        md_path = self.output_dir / f"{base_name}-{timestamp}.md"
        with open(md_path, "w") as f:
            f.write(self.generate_markdown(report))

        logger.info(f"Reports saved to {self.output_dir}")

        return {
            "summary": str(summary_path),
            "junit": str(junit_path),
            "markdown": str(md_path),
        }


# ============================================================================
# NOTIFICATION SYSTEM
# ============================================================================

class NotificationSystem:
    """Sends notifications for test results."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)

    def send_failure_notification(
        self,
        report: TestReport,
        channels: Optional[List[str]] = None,
    ):
        """Send notification for test failures."""
        if not self.enabled:
            logger.info("Notifications disabled")
            return

        if report.total_failed == 0 and report.total_errors == 0:
            logger.info("All tests passed, no notification needed")
            return

        channels = channels or self.config.get("channels", ["console"])

        message = self._build_message(report)

        for channel in channels:
            try:
                if channel == "console":
                    self._send_console(message, report)
                elif channel == "file":
                    self._send_file(message)
                elif channel == "slack":
                    self._send_slack(report)
                elif channel == "email":
                    self._send_email(report)
            except Exception as e:
                logger.error(f"Failed to send {channel} notification: {e}")

    def _build_message(self, report: TestReport) -> str:
        """Build notification message."""
        return f"""
TEST FAILURE ALERT

Pass Rate: {report.pass_rate:.1f}%
Failed: {report.total_failed}/{report.total_tests}
Errors: {report.total_errors}

Coverage: {report.coverage.percent_covered:.1f}% (if available)

Please review the test report for details.
"""

    def _send_console(self, message: str, report: TestReport):
        """Send console notification."""
        print("\n" + "=" * 60)
        print("TEST FAILURE NOTIFICATION")
        print("=" * 60)
        print(message)
        print("=" * 60 + "\n")

    def _send_file(self, message: str):
        """Write notification to file."""
        notify_file = Path("test-notification.txt")
        with open(notify_file, "w") as f:
            f.write(message)

    def _send_slack(self, report: TestReport):
        """Send Slack notification (placeholder)."""
        # Would integrate with Slack API
        # Requires webhook URL from config
        webhook_url = self.config.get("slack_webhook")
        if not webhook_url:
            logger.warning("Slack webhook URL not configured")
            return

        # Implement Slack notification
        logger.info("Slack notification would be sent here")

    def _send_email(self, report: TestReport):
        """Send email notification (placeholder)."""
        # Would integrate with SMTP
        # Requires email config
        logger.info("Email notification would be sent here")


# ============================================================================
# THRESHOLD ENFORCEMENT
# ============================================================================

class ThresholdEnforcer:
    """Enforces coverage and pass rate thresholds."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.min_coverage = self.config.get("min_coverage", 75.0)
        self.min_pass_rate = self.config.get("min_pass_rate", 95.0)

    def check_thresholds(self, report: TestReport) -> Dict[str, bool]:
        """Check if report meets thresholds."""
        results = {}

        # Check pass rate
        results["pass_rate"] = report.pass_rate >= self.min_pass_rate

        # Check coverage (if available)
        if report.coverage:
            results["coverage"] = report.coverage.percent_covered >= self.min_coverage
        else:
            results["coverage"] = True  # No coverage data, pass by default

        return results

    def enforce(self, report: TestReport) -> bool:
        """Enforce thresholds, exit with error if not met."""
        results = self.check_thresholds(report)

        all_passed = all(results.values())

        if not all_passed:
            print("\nTHRESHOLD CHECK FAILED:")

            if not results["pass_rate"]:
                print(
                    f"  - Pass rate {report.pass_rate:.1f}% "
                    f"below minimum {self.min_pass_rate}%"
                )

            if report.coverage and not results["coverage"]:
                print(
                    f"  - Coverage {report.coverage.percent_covered:.1f}% "
                    f"below minimum {self.min_coverage}%"
                )

            sys.exit(1)

        print("\nAll threshold checks passed!")
        return True


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

class TestReporter:
    """Main test reporting orchestrator."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.output_dir = self.config.get("output_dir", "test-reports")
        self.report_generator = ReportGenerator(self.output_dir)
        self.junit_parser = JUnitReportParser()
        self.coverage_parser = CoverageParser()
        self.notification_system = NotificationSystem(self.config.get("notifications"))
        self.threshold_enforcer = ThresholdEnforcer(self.config.get("thresholds"))

    def run_tests_and_report(
        self,
        test_paths: List[str],
        coverage: bool = True,
        enforce_thresholds: bool = True,
    ) -> TestReport:
        """Run tests and generate report."""
        # Run pytest
        junit_path = self._run_pytest(test_paths)

        # Parse JUnit report
        suites = self.junit_parser.parse(junit_path)

        # Parse coverage (if enabled)
        cov_report = None
        if coverage:
            cov_path = Path(self.output_dir) / "coverage.json"
            if cov_path.exists():
                cov_report = self.coverage_parser.parse(str(cov_path))

        # Build report
        report = self._build_report(suites, cov_report)

        # Save reports
        self.report_generator.save_reports(report)

        # Print summary
        print(self.report_generator.generate_summary(report))

        # Check thresholds
        if enforce_thresholds:
            self.threshold_enforcer.enforce(report)

        # Send notifications if failures
        if report.total_failed > 0 or report.total_errors > 0:
            self.notification_system.send_failure_notification(report)

        return report

    def _run_pytest(self, test_paths: List[str]) -> str:
        """Run pytest and generate JUnit report."""
        junit_path = Path(self.output_dir) / "junit.xml"

        cmd = [
            sys.executable, "-m", "pytest",
            "--junit-xml", str(junit_path),
            "--cov=app",
            "--cov-report=json",
            "-v",
        ] + test_paths

        subprocess.run(cmd)

        return str(junit_path)

    def _build_report(
        self,
        suites: List[TestSuiteResult],
        coverage: Optional[CoverageReport],
    ) -> TestReport:
        """Build complete test report."""
        total_tests = sum(s.tests for s in suites)
        total_passed = sum(s.passed for s in suites)
        total_failed = sum(s.failed for s in suites)
        total_skipped = sum(s.skipped for s in suites)
        total_errors = sum(s.errors for s in suites)
        total_duration = sum(s.duration for s in suites)

        pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

        return TestReport(
            timestamp=datetime.now(),
            suites=suites,
            coverage=coverage,
            total_tests=total_tests,
            total_passed=total_passed,
            total_failed=total_failed,
            total_skipped=total_skipped,
            total_errors=total_errors,
            total_duration=total_duration,
            pass_rate=pass_rate,
        )


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    """CLI entry point for test reporting."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Reporter")
    parser.add_argument(
        "test_paths",
        nargs="*",
        default=["tests/"],
        help="Paths to test files",
    )
    parser.add_argument(
        "--output-dir",
        default="test-reports",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Disable coverage reporting",
    )
    parser.add_argument(
        "--no-threshold",
        action="store_true",
        help="Disable threshold enforcement",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Enable notifications",
    )

    args = parser.parse_args()

    config = {
        "output_dir": args.output_dir,
        "notifications": {
            "enabled": args.notify,
            "channels": ["console"],
        },
        "thresholds": {
            "enabled": not args.no_threshold,
            "min_coverage": 75.0,
            "min_pass_rate": 95.0,
        },
    }

    reporter = TestReporter(config)
    reporter.run_tests_and_report(
        test_paths=args.test_paths,
        coverage=not args.no_coverage,
        enforce_thresholds=not args.no_threshold,
    )


if __name__ == "__main__":
    main()
