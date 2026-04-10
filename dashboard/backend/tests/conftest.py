"""
Pytest configuration and fixtures for backend testing.

Provides:
- Test database setup and teardown
- Test client with mocked dependencies
- Factory functions for creating test data
- Async test support
"""

import os

# Set testing environment variable BEFORE any other imports
# This ensures the database module uses SQLite for tests
os.environ["TESTING"] = "true"
os.environ["TEST_DATABASE_URL"] = "sqlite:///:memory:"

import asyncio
import pytest
from typing import Generator, AsyncGenerator
from uuid import uuid4
from datetime import datetime

from sqlalchemy import create_engine, text, event, JSON
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

# Import app modules after setting up test configuration
from app.main import app
from app.database import Base, get_db
from app.models import User, Model, Run, Metric
from app.schemas import (
    ModelCreate,
    ModelUpdate,
    RunCreate,
    RunUpdate,
    MetricCreate,
)


# Test database configuration - use SQLite for isolation and speed
TEST_DATABASE_URL = "sqlite:///:memory:"

# Set testing environment variable
os.environ["TESTING"] = "true"
os.environ["TEST_DATABASE_URL"] = TEST_DATABASE_URL


@pytest.fixture(scope="session")
def test_engine():
    """
    Create a test database engine.

    Uses SQLite with in-memory mode for fast, isolated tests.
    """
    # Import models to ensure they are registered with Base
    from app.models import User, Model, Run, Metric, Tag, ModelVersion, RunTag  # noqa: F401

    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )

    # Create all tables
    Base.metadata.create_all(bind=engine)

    yield engine

    # Cleanup
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session(test_engine) -> Generator[Session, None, None]:
    """
    Create a fresh database session for each test.

    Each test gets a clean database state with automatic rollback.
    """
    connection = test_engine.connect()
    transaction = connection.begin()

    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=connection,
    )

    session = SessionLocal()

    try:
        yield session
    finally:
        # Rollback all changes to keep tests isolated
        transaction.rollback()
        session.close()
        connection.close()


@pytest.fixture
def client(db_session: Session) -> Generator[TestClient, None, None]:
    """
    Create a test client with mocked database dependency.

    Overrides the default database dependency to use test database.
    """
    from app.api.deps import get_db_session

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    def override_get_db_session():
        try:
            yield db_session
        finally:
            pass

    # Override both get_db and get_db_session to ensure all routes use test database
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_db_session] = override_get_db_session

    # Execute background tasks synchronously in tests
    with TestClient(app, raise_server_exceptions=True) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def test_user(db_session: Session) -> User:
    """Create a test user fixture."""
    user = User(
        id=str(uuid4()),
        email=f"test-{uuid4()}@example.com",
        name="Test User",
        role="editor",
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def test_model(db_session: Session, test_user: User) -> Model:
    """Create a test model fixture."""
    model = Model(
        id=str(uuid4()),
        name="Test Model",
        checkpoint="test/checkpoint-1b",
        model_type="llm",
        family="Test",
        parameters=1000000000,
        created_by=test_user.id,
    )
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)
    return model


@pytest.fixture
def test_run(db_session: Session, test_model: Model, test_user: User) -> Run:
    """Create a test run fixture."""
    run = Run(
        id=str(uuid4()),
        model_id=test_model.id,
        user_id=test_user.id,
        build_name=f"test_build_{uuid4()}",
        run_type="benchmark",
        status="completed",
        device="gpu",
        backend="llamacpp",
        dtype="float16",
        config={"iterations": 10, "prompts": 5},
    )
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)
    return run


@pytest.fixture
def test_metric(db_session: Session, test_run: Run) -> Metric:
    """Create a test metric fixture."""
    metric = Metric(
        id=str(uuid4()),
        run_id=test_run.id,
        category="performance",
        name="seconds_to_first_token",
        display_name="Seconds To First Token",
        value_numeric=0.025,
        unit="seconds",
    )
    db_session.add(metric)
    db_session.commit()
    db_session.refresh(metric)
    return metric


# Factory functions for creating test data


class ModelFactory:
    """Factory for creating test models."""

    @staticmethod
    def create(
        db: Session,
        name: str | None = None,
        checkpoint: str | None = None,
        model_type: str = "llm",
        family: str | None = None,
        parameters: int | None = None,
        **kwargs,
    ) -> Model:
        model = Model(
            id=str(uuid4()),
            name=name or f"Test Model {uuid4()}",
            checkpoint=checkpoint or f"test/checkpoint-{uuid4()}",
            model_type=model_type,
            family=family or "Test",
            parameters=parameters or 1000000000,
            **kwargs,
        )
        db.add(model)
        db.commit()
        db.refresh(model)
        return model


class RunFactory:
    """Factory for creating test runs."""

    @staticmethod
    def create(
        db: Session,
        model_id: str | None = None,
        build_name: str | None = None,
        run_type: str = "benchmark",
        status: str = "pending",
        device: str | None = None,
        backend: str | None = None,
        **kwargs,
    ) -> Run:
        if not model_id:
            # Create a model if not provided
            model = ModelFactory.create(db)
            model_id = model.id

        run = Run(
            id=str(uuid4()),
            model_id=model_id,
            build_name=build_name or f"test_build_{uuid4()}",
            run_type=run_type,
            status=status,
            device=device,
            backend=backend,
            **kwargs,
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        return run


class MetricFactory:
    """Factory for creating test metrics."""

    @staticmethod
    def create(
        db: Session,
        run_id: str | None = None,
        category: str = "performance",
        name: str = "test_metric",
        value_numeric: float | None = None,
        value_text: str | None = None,
        unit: str | None = None,
        **kwargs,
    ) -> Metric:
        if not run_id:
            run = RunFactory.create(db)
            run_id = run.id

        metric = Metric(
            id=str(uuid4()),
            run_id=run_id,
            category=category,
            name=name,
            value_numeric=value_numeric,
            value_text=value_text,
            unit=unit or "units",
            **kwargs,
        )
        db.add(metric)
        db.commit()
        db.refresh(metric)
        return metric

    @staticmethod
    def create_performance_metrics(db: Session, run_id: str) -> dict[str, Metric]:
        """Create a standard set of performance metrics for a run."""
        metrics = {}

        perf_data = {
            "seconds_to_first_token": (0.025, "seconds"),
            "prefill_tokens_per_second": (1500.0, "tokens/s"),
            "token_generation_tokens_per_second": (45.5, "tokens/s"),
            "max_memory_used_gbyte": (4.2, "GB"),
        }

        for name, (value, unit) in perf_data.items():
            metric = MetricFactory.create(
                db,
                run_id=run_id,
                category="performance",
                name=name,
                value_numeric=value,
                unit=unit,
            )
            metrics[name] = metric

        return metrics

    @staticmethod
    def create_accuracy_metrics(db: Session, run_id: str) -> dict[str, Metric]:
        """Create a standard set of accuracy metrics for a run."""
        metrics = {}

        accuracy_data = {
            "mmlu_stem": (65.4, "%"),
            "mmlu_humanities": (68.2, "%"),
            "humaneval": (42.5, "%"),
        }

        for name, (value, unit) in accuracy_data.items():
            metric = MetricFactory.create(
                db,
                run_id=run_id,
                category="accuracy",
                name=name,
                value_numeric=value,
                unit=unit,
            )
            metrics[name] = metric

        return metrics


class UserFactory:
    """Factory for creating test users."""

    @staticmethod
    def create(
        db: Session,
        email: str | None = None,
        name: str | None = None,
        role: str = "viewer",
        is_active: bool = True,
        **kwargs,
    ) -> User:
        user = User(
            id=str(uuid4()),
            email=email or f"test-{uuid4()}@example.com",
            name=name or "Test User",
            role=role,
            is_active=is_active,
            **kwargs,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user


# Async fixture support
@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# pytest-asyncio configuration
@pytest.fixture(scope="function")
async def async_client(db_session: Session) -> AsyncGenerator[TestClient, None]:
    """
    Create an async test client.

    For testing async endpoints and WebSocket connections.
    """
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    async with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()
