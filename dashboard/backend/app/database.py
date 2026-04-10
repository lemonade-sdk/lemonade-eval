"""
Database connection and session management.

Provides both synchronous and asynchronous database sessions.
"""

import os
from collections.abc import AsyncGenerator, Generator
from typing import Optional

from sqlalchemy import create_engine, event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import Session, sessionmaker, DeclarativeBase

from app.config import settings


# Check if running in test mode - must be checked before engine creation
IS_TEST_MODE = (
    os.environ.get("TESTING", "false").lower() == "true" or
    "pytest" in os.environ.get("_", "") or
    os.environ.get("TEST_DATABASE_URL", "").startswith("sqlite")
)

# Force SQLite for test mode to avoid PostgreSQL dependency in tests
if IS_TEST_MODE:
    os.environ.setdefault("TEST_DATABASE_URL", "sqlite:///:memory:")
    _engine_url = os.environ.get("TEST_DATABASE_URL", "sqlite:///:memory:")
    _is_sqlite = "sqlite" in _engine_url
else:
    _engine_url = settings.database_url
    _is_sqlite = False


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


# Synchronous engine for migrations and scripts
# Use SQLite for test mode
if _is_sqlite:
    sync_engine = create_engine(
        _engine_url,
        echo=False,
        connect_args={"check_same_thread": False},
    )
else:
    sync_engine = create_engine(
        settings.database_url,
        echo=settings.debug,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
    )

# Async engine for FastAPI application
if IS_TEST_MODE and "sqlite" in os.environ.get("TEST_DATABASE_URL", ""):
    # For test mode, we don't need async engine since SQLite is synchronous
    async_engine = None
else:
    async_engine = create_async_engine(
        settings.database_async_url,
        echo=settings.debug,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
    )

# Session factories
SyncSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=sync_engine,
    class_=Session,
)

AsyncSessionLocal = async_sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI routes to get database session.

    Yields:
        Database session that closes automatically after use.
    """
    db = SyncSessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Async dependency for FastAPI routes to get database session.

    Yields:
        Async database session that closes automatically after use.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def init_db() -> None:
    """
    Initialize the database by creating all tables.

    Note: In production, use Alembic migrations instead.
    """
    from app.models import User, Model, Run, Metric, Tag, ModelVersion, RunTag  # noqa: F401
    Base.metadata.create_all(bind=sync_engine)


async def async_init_db() -> None:
    """
    Initialize the database asynchronously.
    """
    from app.models import User, Model, Run, Metric, Tag, ModelVersion, RunTag  # noqa: F401
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
