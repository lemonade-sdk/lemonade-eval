"""
Auto-Activation Hook System for Test Discovery and Execution.

This module provides:
- Hook-based test activation for new components
- Pre-commit hook integration
- Test configuration auto-generation
- Component type detection and test strategy selection
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ComponentInfo:
    """Information about a newly created component."""
    path: str
    component_type: str  # api, service, model, websocket, cache, middleware
    name: str
    tests_required: List[str] = field(default_factory=list)
    test_strategy: str = "standard"  # standard, integration, stress


@dataclass
class TestConfig:
    """Configuration for auto-generated tests."""
    component_path: str
    test_file_path: str
    test_types: List[str]
    fixtures_required: List[str]
    mock_services: List[str]


# ============================================================================
# COMPONENT DETECTOR
# ============================================================================

class ComponentDetector:
    """Detects component type and generates test requirements."""

    # Component type patterns
    TYPE_PATTERNS = {
        "api": ["api/", "api_v1/", "routes/", "endpoints/"],
        "service": ["services/", "service/"],
        "model": ["models/", "schemas/", "db/"],
        "websocket": ["websocket", "ws/", "realtime/"],
        "cache": ["cache/", "caching/"],
        "middleware": ["middleware/", "middlewares/"],
        "integration": ["integration/", "cli/", "external/"],
    }

    # Test requirements by component type
    TEST_REQUIREMENTS = {
        "api": ["unit", "integration", "stress"],
        "service": ["unit", "integration"],
        "model": ["unit", "schema"],
        "websocket": ["unit", "integration", "stress", "connection"],
        "cache": ["unit", "integration", "stampede"],
        "middleware": ["unit", "integration", "load"],
        "integration": ["unit", "integration", "e2e"],
    }

    def detect_component_type(self, path: str) -> str:
        """Detect component type from file path."""
        path_lower = path.lower()

        for comp_type, patterns in self.TYPE_PATTERNS.items():
            if any(pattern in path_lower for pattern in patterns):
                return comp_type

        return "standard"

    def get_test_requirements(self, component_type: str) -> List[str]:
        """Get required test types for component."""
        return self.TEST_REQUIREMENTS.get(component_type, ["unit"])

    def analyze_component(self, path: str) -> ComponentInfo:
        """Analyze a component and generate requirements."""
        component_type = self.detect_component_type(path)
        name = Path(path).stem
        tests_required = self.get_test_requirements(component_type)

        # Determine test strategy based on component type
        if component_type in ["websocket", "cache"]:
            test_strategy = "stress"
        elif component_type in ["api", "middleware"]:
            test_strategy = "integration"
        else:
            test_strategy = "standard"

        return ComponentInfo(
            path=path,
            component_type=component_type,
            name=name,
            tests_required=tests_required,
            test_strategy=test_strategy,
        )


# ============================================================================
# TEST GENERATOR
# ============================================================================

class TestGenerator:
    """Generates test file templates for new components."""

    def __init__(self, test_dir: str = "tests"):
        self.test_dir = Path(test_dir)

    def generate_test_config(self, component: ComponentInfo) -> TestConfig:
        """Generate test configuration for component."""
        # Determine test file path
        if component.component_type == "api":
            test_subdir = self.test_dir
        elif component.component_type == "service":
            test_subdir = self.test_dir
        else:
            test_subdir = self.test_dir / component.component_type

        test_file_path = test_subdir / f"test_{component.name}.py"

        return TestConfig(
            component_path=component.path,
            test_file_path=str(test_file_path),
            test_types=component.tests_required,
            fixtures_required=self._get_fixtures(component.component_type),
            mock_services=self._get_mocks(component.component_type),
        )

    def _get_fixtures(self, component_type: str) -> List[str]:
        """Get required fixtures for component type."""
        fixture_map = {
            "api": ["client", "db_session", "test_user"],
            "service": ["db_session", "test_model", "test_run"],
            "model": ["db_session"],
            "websocket": ["client", "async_client"],
            "cache": ["cache_manager"],
            "middleware": ["client", "rate_limiter"],
            "integration": ["client", "db_session"],
        }
        return fixture_map.get(component_type, ["client"])

    def _get_mocks(self, component_type: str) -> List[str]:
        """Get services to mock for component type."""
        mock_map = {
            "api": ["RunService", "ModelService", "MetricService"],
            "service": [],
            "model": [],
            "websocket": ["manager"],
            "cache": ["redis"],
            "middleware": ["redis"],
            "integration": ["CLIClient"],
        }
        return mock_map.get(component_type, [])

    def generate_test_template(self, config: TestConfig) -> str:
        """Generate test file template."""
        test_types = config.test_types
        fixtures = config.fixtures_required

        # Build fixture decorator
        fixture_params = ", ".join(fixtures)

        template = f'''"""
Auto-generated tests for {config.component_path}.

Generated: {datetime.now().isoformat()}
Test Types: {", ".join(test_types)}
"""

import pytest
from typing import Any, Dict


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def test_fixture():
    """Test fixture for component."""
    # TODO: Add fixture setup
    return None


# ============================================================================
# UNIT TESTS
# ============================================================================

class TestComponentUnit:
    """Unit tests for component."""

    def test_basic_functionality(self, {fixture_params}):
        """Test basic component functionality."""
        # TODO: Implement test
        assert True

    def test_input_validation(self, {fixture_params}):
        """Test input validation."""
        # TODO: Implement test
        assert True


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestComponentIntegration:
    """Integration tests for component."""

    def test_integration_flow(self, {fixture_params}):
        """Test integration flow."""
        # TODO: Implement test
        assert True


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestComponentErrorHandling:
    """Error handling tests for component."""

    def test_error_conditions(self, {fixture_params}):
        """Test error condition handling."""
        # TODO: Implement test
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
'''
        return template


# ============================================================================
# HOOK SYSTEM
# ============================================================================

class TestHookSystem:
    """
    Hook system for automatic test activation.

    Usage:
        hooks = TestHookSystem()
        hooks.on_component_created(path)
    """

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.detector = ComponentDetector()
        self.generator = TestGenerator(str(self.project_root / "tests"))

    def on_component_created(self, path: str) -> Optional[str]:
        """
        Hook called when a new component is created.

        Args:
            path: Path to the new component file

        Returns:
            Path to generated test file, or None if not applicable
        """
        # Analyze component
        component = self.detector.analyze_component(path)

        logger.info(f"New component detected: {component.name} ({component.component_type})")
        logger.info(f"Required tests: {component.tests_required}")
        logger.info(f"Test strategy: {component.test_strategy}")

        # Generate test config
        config = self.generator.generate_test_config(component)

        # Check if test file already exists
        if Path(config.test_file_path).exists():
            logger.info(f"Test file already exists: {config.test_file_path}")
            return None

        # Generate test file
        template = self.generator.generate_test_template(config)

        # Create test directory if needed
        test_dir = Path(config.test_file_path).parent
        test_dir.mkdir(parents=True, exist_ok=True)

        # Write test file
        with open(config.test_file_path, "w") as f:
            f.write(template)

        logger.info(f"Generated test file: {config.test_file_path}")
        return config.test_file_path

    def on_component_modified(self, path: str) -> List[str]:
        """
        Hook called when a component is modified.

        Args:
            path: Path to the modified component

        Returns:
            List of test files that should be re-run
        """
        component = self.detector.analyze_component(path)

        # Find related test files
        test_files = []
        test_dir = self.project_root / "tests"

        if test_dir.exists():
            for test_file in test_dir.rglob(f"test_{component.name}.py"):
                test_files.append(str(test_file))

        logger.info(f"Component modified: {path}")
        logger.info(f"Related tests to run: {test_files}")

        return test_files

    def run_tests_for_component(self, path: str) -> bool:
        """
        Run tests related to a component.

        Args:
            path: Path to the component

        Returns:
            True if tests passed, False otherwise
        """
        import subprocess

        test_files = self.on_component_modified(path)

        if not test_files:
            logger.info(f"No tests found for component: {path}")
            return True

        # Run pytest
        cmd = [sys.executable, "-m", "pytest"] + test_files + ["-v"]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            success = result.returncode == 0

            if success:
                logger.info(f"All tests passed for: {path}")
            else:
                logger.error(f"Tests failed for: {path}")
                logger.error(result.stdout)
                logger.error(result.stderr)

            return success

        except Exception as e:
            logger.error(f"Error running tests: {e}")
            return False


# ============================================================================
# GIT HOOK INTEGRATION
# ============================================================================

class GitHookInstaller:
    """Installs git hooks for automatic test activation."""

    PRE_COMMIT_HOOK = '''#!/bin/bash
# Pre-commit hook for auto-test activation

echo "Running pre-commit test checks..."

# Get staged Python files
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep -E "\\.py$")

if [ -z "$STAGED_FILES" ]; then
    echo "No Python files staged for commit."
    exit 0
fi

# Run pytest on related tests
echo "Running tests for changed files..."
python -m pytest dashboard/backend/tests/ -v --tb=short

# Check test results
if [ $? -ne 0 ]; then
    echo "Pre-commit tests failed. Please fix issues before committing."
    exit 1
fi

echo "Pre-commit tests passed!"
exit 0
'''

    POST_MERGE_HOOK = '''#!/bin/bash
# Post-merge hook for test validation

echo "Running post-merge test validation..."

# Run full test suite
python -m pytest dashboard/backend/tests/ -v --tb=short --cov=app

echo "Post-merge test validation complete."
'''

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.git_dir = self.project_root / ".git" / "hooks"

    def install_hooks(self) -> Dict[str, str]:
        """Install git hooks."""
        self.git_dir.mkdir(parents=True, exist_ok=True)

        hooks_installed = {}

        # Install pre-commit hook
        pre_commit_path = self.git_dir / "pre-commit"
        with open(pre_commit_path, "w") as f:
            f.write(self.PRE_COMMIT_HOOK)
        os.chmod(pre_commit_path, 0o755)
        hooks_installed["pre-commit"] = str(pre_commit_path)

        # Install post-merge hook
        post_merge_path = self.git_dir / "post-merge"
        with open(post_merge_path, "w") as f:
            f.write(self.POST_MERGE_HOOK)
        os.chmod(post_merge_path, 0o755)
        hooks_installed["post-merge"] = str(post_merge_path)

        logger.info(f"Git hooks installed: {list(hooks_installed.keys())}")
        return hooks_installed


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def setup_auto_testing(project_root: str = None):
    """
    Set up auto-testing for a project.

    Args:
        project_root: Path to project root directory
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent
    else:
        project_root = Path(project_root)

    # Initialize hook system
    hooks = TestHookSystem(str(project_root))

    # Install git hooks
    git_hooks = GitHookInstaller(str(project_root))
    installed = git_hooks.install_hooks()

    print(f"Auto-testing setup complete!")
    print(f"Git hooks installed: {list(installed.keys())}")
    print(f"Test directory: {project_root / 'tests'}")

    return hooks


if __name__ == "__main__":
    # Setup auto-testing
    setup_auto_testing()
