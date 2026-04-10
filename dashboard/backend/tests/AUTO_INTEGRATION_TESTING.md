# Auto-Integration Testing Guide

## Overview

The Auto-Integration Testing system provides comprehensive multi-layered testing for the UI-UX Eval Dashboard with self-configuration and real-time coordination.

## Features

### 1. Auto-Activation Testing
- **Hook-Based Triggers**: Automatically activates when new components are created
- **Self-Configuration**: Adapts testing strategies based on component type
- **Multi-Layered Testing**: Unit, Integration, System, and User Acceptance testing
- **Real-time Coordination**: Works with Quality Assurance and Innovation agents

### 2. Intelligent Testing Strategies
- **Context-Aware Testing**: Analyzes component requirements for optimal testing approach
- **Performance Benchmarking**: Continuous performance monitoring and optimization
- **Error Pattern Recognition**: Identifies and learns from common failure patterns
- **Automated Test Generation**: Creates comprehensive test suites automatically

### 3. Quality Integration
- **Multi-Agent Coordination**: Coordinates with Quality Reviewer and Innovation agents
- **Continuous Validation**: Ongoing quality assurance throughout development lifecycle
- **Predictive Quality Analysis**: Anticipates potential issues before they occur
- **Self-Improvement**: Continuously enhances testing methodologies

## Directory Structure

```
dashboard/backend/tests/
├── conftest.py                 # Shared fixtures
├── auto_activate_hooks.py      # Auto-activation hook system
├── test_reporter.py            # Test reporting and notifications
├── integration/                # Integration tests
│   ├── conftest.py
│   └── test_import_pipeline.py
├── stress/                     # Stress and load tests
│   ├── conftest.py
│   ├── test_rate_limiting_load.py
│   ├── test_cache_stampede.py
│   └── test_websocket_stress.py
└── load/                       # Load testing utilities
    └── __init__.py
```

## Test Categories

### Unit Tests (`-m "unit"`)
Fast, isolated tests for individual components:
```bash
pytest tests/ -m "unit" -v
```

### Integration Tests (`-m "integration"`)
Tests for component interactions:
```bash
pytest tests/integration/ -m "integration" -v
```

### Stress Tests (`-m "stress"`)
Performance and load testing:
```bash
pytest tests/stress/ -v
```

## Running Tests

### Full Test Suite
```bash
cd dashboard/backend
pytest --cov=app --cov-report=html --cov-report=term-missing
```

### Test with Coverage
```bash
pytest --cov=app \
       --cov-report=html \
       --cov-report=json \
       --cov-report=term-missing \
       -v
```

### Test Specific Categories
```bash
# Unit tests only
pytest tests/ -m "unit" -v

# Integration tests only
pytest tests/integration/ -m "integration" -v

# Stress tests only
pytest tests/stress/ -v
```

## Auto-Activation Hooks

### Setup
```bash
cd dashboard/backend
python tests/auto_activate_hooks.py
```

This installs:
- Git pre-commit hooks for automatic test runs
- Post-merge hooks for validation
- Component detection and test generation

### Usage
When you create a new component:
```python
# Create new API route
touch app/api/v1/new_feature.py

# Run hook system to generate tests
python tests/auto_activate_hooks.py
```

The system will:
1. Detect component type (api, service, model, etc.)
2. Generate appropriate test templates
3. Configure required fixtures
4. Set up test strategies

## Test Reporting

### Generate Reports
```bash
python tests/test_reporter.py tests/ \
    --output-dir test-reports \
    --notify
```

### Report Formats
- **JUnit XML**: For CI/CD integration
- **HTML**: Human-readable format
- **Markdown**: Documentation format
- **Console**: Immediate feedback

### Coverage Thresholds
- Backend: Minimum 60% coverage
- Frontend: Minimum 50% coverage

## CI/CD Integration

### GitHub Actions
The workflow `auto-integration-testing.yml` provides:
- Parallel test execution
- Coverage threshold enforcement
- Artifact upload for reports
- Quality gate checks

### Jobs
1. `backend-unit-tests`: Unit tests with coverage
2. `backend-integration-tests`: Integration tests
3. `backend-stress-tests`: Performance tests
4. `frontend-unit-tests`: Vitest tests
5. `frontend-e2e-tests`: Playwright tests
6. `coverage-summary`: Combined coverage report
7. `quality-gates`: Threshold enforcement
8. `test-notification`: Result notifications

## Critical Test Coverage

### CLI to Dashboard Data Flow
Tests in `tests/integration/test_import_pipeline.py`:
- Evaluation import from CLI
- Bulk import operations
- YAML data transformation
- Metric validation

### WebSocket Real-Time Updates
Tests in `tests/stress/test_websocket_stress.py`:
- Concurrent connection handling
- Broadcast efficiency
- Message throughput
- Connection stability

### Rate Limiting Under Load
Tests in `tests/stress/test_rate_limiting_load.py`:
- High-volume request flooding
- Concurrent client simulation
- Rate limit recovery
- Endpoint-specific limits

### Cache Stampede Prevention
Tests in `tests/stress/test_cache_stampede.py`:
- Thundering herd scenarios
- Lock acquisition under load
- Double-check pattern
- Fallback behavior

## Test Configuration

### pytest.ini
```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts =
    -v
    --strict-markers
    --tb=short
    --cov=app
    --cov-report=term-missing
    --cov-report=html
    --cov-report=json
    --asyncio-mode=auto

markers =
    unit: Unit tests
    integration: Integration tests
    stress: Stress tests
    slow: Slow running tests
```

### Environment Variables
```bash
export TESTING=true
export TEST_DATABASE_URL=sqlite:///:memory:
export REDIS_URL=redis://localhost:6379/0
```

## Notifications

### Configure Notifications
```python
config = {
    "notifications": {
        "enabled": True,
        "channels": ["console", "file", "slack"],
        "slack_webhook": "https://hooks.slack.com/...",
    },
    "thresholds": {
        "min_coverage": 60.0,
        "min_pass_rate": 95.0,
    },
}
```

### Notification Channels
- **Console**: Immediate terminal output
- **File**: Written to `test-notification.txt`
- **Slack**: Webhook-based (requires configuration)
- **Email**: SMTP-based (requires configuration)

## Performance Benchmarks

### Target Metrics
| Metric | Target |
|--------|--------|
| Unit Test Duration | < 30 seconds |
| Integration Test Duration | < 2 minutes |
| Stress Test Duration | < 5 minutes |
| API Response Time (p95) | < 200ms |
| WebSocket Connection Time | < 100ms |
| Cache Hit Rate | > 80% |

## Troubleshooting

### Common Issues

#### Tests Failing Due to Database
```bash
# Ensure test database is set up
export TEST_DATABASE_URL=sqlite:///:memory:
pytest tests/ --db-reset
```

#### Redis Connection Errors
```bash
# Start Redis for integration tests
docker run -d -p 6379:6379 redis:7-alpine
```

#### Coverage Not Generated
```bash
# Ensure coverage is enabled
pytest --cov=app --cov-report=json
```

## Best Practices

### Writing Tests
1. **Use fixtures**: Leverage existing fixtures in `conftest.py`
2. **Mark tests**: Add appropriate markers (`@pytest.mark.unit`)
3. **Mock external services**: Use `unittest.mock` for isolation
4. **Test asynchronously**: Use `@pytest.mark.asyncio` for async code

### Running Tests
1. **Run locally first**: Before pushing to CI
2. **Check coverage**: Ensure new code has test coverage
3. **Update mocks**: Keep mocks in sync with real implementations
4. **Review reports**: Check test reports for patterns

## API Reference

### Test Reporter
```python
from tests.test_reporter import TestReporter

reporter = TestReporter(config={
    "output_dir": "test-reports",
    "notifications": {"enabled": True},
})

report = reporter.run_tests_and_report(
    test_paths=["tests/"],
    coverage=True,
    enforce_thresholds=True,
)
```

### Auto-Activation Hooks
```python
from tests.auto_activate_hooks import TestHookSystem

hooks = TestHookSystem()

# On component creation
test_file = hooks.on_component_created("app/api/v1/new_feature.py")

# On component modification
test_files = hooks.on_component_modified("app/api/v1/new_feature.py")

# Run tests for component
success = hooks.run_tests_for_component("app/api/v1/new_feature.py")
```

## Contributing

### Adding New Tests
1. Create test file in appropriate directory
2. Add fixtures to `conftest.py` if needed
3. Mark tests with appropriate markers
4. Update documentation

### Adding New Stress Tests
1. Use fixtures from `tests/stress/conftest.py`
2. Configure load parameters
3. Document performance benchmarks
4. Add to CI workflow

## License

This testing framework is part of the Lemonade Eval Dashboard project.
