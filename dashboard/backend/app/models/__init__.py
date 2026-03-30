"""
SQLAlchemy models for the Lemonade Eval Dashboard.

Models defined:
- User: Dashboard users with authentication
- Model: LLM/VLM models being evaluated
- ModelVersion: Different variants of a model
- Run: Evaluation runs
- Metric: Performance and accuracy metrics
- Tag: Tags for organization
- RunTag: Many-to-many relationship between runs and tags
"""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    JSON,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


# Use JSONB for PostgreSQL, but allow fallback to JSON for SQLite testing
# This enables running tests with SQLite while production uses PostgreSQL
try:
    # Check if we're running in test mode with SQLite
    import os
    _USE_SQLITE = os.environ.get('TEST_DATABASE_URL', '').startswith('sqlite')
except Exception:
    _USE_SQLITE = False

# For testing compatibility, use JSON which works with both SQLite and PostgreSQL
# PostgreSQL will still work correctly with JSON type
FlexibleJSONB = JSON


def generate_uuid() -> str:
    """Generate a UUID string."""
    return str(uuid.uuid4())


class User(Base):
    """
    User model for dashboard authentication and authorization.

    Attributes:
        id: Unique identifier
        email: User email (unique)
        name: User display name
        role: User role (admin, editor, viewer)
        hashed_password: Bcrypt-hashed password
        api_key_hash: Hashed API key for programmatic access
        api_key_prefix: Prefix of API key for identification
        is_active: Whether the user account is active
        created_at: Account creation timestamp
        updated_at: Last update timestamp
    """
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=generate_uuid,
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(50), default="viewer")
    hashed_password: Mapped[Optional[str]] = mapped_column(String(255))
    api_key_hash: Mapped[Optional[str]] = mapped_column(String(255))
    api_key_prefix: Mapped[Optional[str]] = mapped_column(String(10), index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Relationships
    models: Mapped[list["Model"]] = relationship(back_populates="creator", lazy="select")
    runs: Mapped[list["Run"]] = relationship(back_populates="user", lazy="select")
    created_tags: Mapped[list["Tag"]] = relationship(back_populates="creator", lazy="select")

    __table_args__ = (
        Index("idx_users_email", "email"),
        Index("idx_users_api_key_prefix", "api_key_prefix"),
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email})>"


class Model(Base):
    """
    Model representing an LLM/VLM model being evaluated.

    Attributes:
        id: Unique identifier
        name: Human-readable model name
        checkpoint: Model checkpoint identifier (unique)
        model_type: Type of model (llm, vlm, embedding)
        family: Model family (e.g., "Llama", "Qwen", "Phi")
        parameters: Parameter count
        max_context_length: Maximum context window size
        architecture: Model architecture
        license_type: License information
        hf_repo: HuggingFace repository
        metadata: Additional model information (JSONB)
        created_by: User who added the model
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """
    __tablename__ = "models"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=generate_uuid,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    checkpoint: Mapped[str] = mapped_column(String(500), nullable=False, unique=True, index=True)
    model_type: Mapped[str] = mapped_column(String(50), default="llm")
    family: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    parameters: Mapped[Optional[int]] = mapped_column(Integer)
    max_context_length: Mapped[Optional[int]] = mapped_column(Integer)
    architecture: Mapped[Optional[str]] = mapped_column(String(100))
    license_type: Mapped[Optional[str]] = mapped_column(String(100))
    hf_repo: Mapped[Optional[str]] = mapped_column(String(255))
    model_metadata: Mapped[dict] = mapped_column("metadata_json", FlexibleJSONB, default=dict)
    created_by: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False), ForeignKey("users.id")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Relationships
    creator: Mapped[Optional["User"]] = relationship(back_populates="models")
    versions: Mapped[list["ModelVersion"]] = relationship(
        back_populates="model",
        cascade="all, delete-orphan",
        lazy="select",
    )
    runs: Mapped[list["Run"]] = relationship(
        back_populates="model",
        cascade="all, delete-orphan",
        lazy="select",
    )

    __table_args__ = (
        Index("idx_models_name", "name"),
        Index("idx_models_family", "family"),
        Index("idx_models_checkpoint", "checkpoint"),
    )

    def __repr__(self) -> str:
        return f"<Model(id={self.id}, name={self.name}, checkpoint={self.checkpoint})>"


class ModelVersion(Base):
    """
    Model version for tracking different variants of a model.

    Attributes:
        id: Unique identifier
        model_id: Reference to parent model
        version: Version identifier
        quantization: Quantization type (int4, int8, fp16, awq)
        dtype: Data type
        backend: Backend runtime
        config: Version-specific configuration (JSONB)
        is_default: Whether this is the default version
        created_at: Creation timestamp
    """
    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=generate_uuid,
    )
    model_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("models.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    quantization: Mapped[Optional[str]] = mapped_column(String(50))
    dtype: Mapped[Optional[str]] = mapped_column(String(50))
    backend: Mapped[Optional[str]] = mapped_column(String(100))
    config: Mapped[dict] = mapped_column(FlexibleJSONB, default=dict)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    model: Mapped["Model"] = relationship(back_populates="versions")

    __table_args__ = (
        Index("idx_model_versions_model", "model_id"),
        UniqueConstraint("model_id", "version", "quantization", "dtype", "backend"),
    )

    def __repr__(self) -> str:
        return f"<ModelVersion(id={self.id}, model_id={self.model_id}, version={self.version})>"


class Run(Base):
    """
    Evaluation run representing a single model evaluation execution.

    Attributes:
        id: Unique identifier
        model_id: Reference to evaluated model
        user_id: Reference to user who ran the evaluation
        build_name: Unique build identifier
        run_type: Type of evaluation (benchmark, accuracy-mmlu, etc.)
        status: Run status (pending, running, completed, failed, cancelled)
        status_message: Status details
        device: Device type (cpu, gpu, npu, etc.)
        backend: Backend runtime (llamacpp, ort, flm)
        dtype: Data type (float32, float16, int4, int8)
        config: Run configuration (JSONB)
        started_at: Evaluation start time
        completed_at: Evaluation completion time
        duration_seconds: Total duration
        system_info: System information snapshot (JSONB)
        lemonade_version: Lemonade SDK version
        build_uid: Unique build identifier
        log_file_path: Path to log file
        error_log: Error messages
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=generate_uuid,
    )
    model_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("models.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False), ForeignKey("users.id"), index=True
    )
    build_name: Mapped[str] = mapped_column(String(255), nullable=False)
    run_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(50), default="pending", index=True)
    status_message: Mapped[Optional[str]] = mapped_column(Text)
    device: Mapped[Optional[str]] = mapped_column(String(50))
    backend: Mapped[Optional[str]] = mapped_column(String(100))
    dtype: Mapped[Optional[str]] = mapped_column(String(50))
    config: Mapped[dict] = mapped_column(FlexibleJSONB, default=dict)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    duration_seconds: Mapped[Optional[float]] = mapped_column(Numeric)
    system_info: Mapped[dict] = mapped_column(FlexibleJSONB, default=dict)
    lemonade_version: Mapped[Optional[str]] = mapped_column(String(20))
    build_uid: Mapped[Optional[str]] = mapped_column(String(100))
    log_file_path: Mapped[Optional[str]] = mapped_column(String(500))
    error_log: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Relationships
    model: Mapped["Model"] = relationship(back_populates="runs")
    user: Mapped[Optional["User"]] = relationship(back_populates="runs")
    metrics: Mapped[list["Metric"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        lazy="select",
    )
    tags: Mapped[list["Tag"]] = relationship(
        secondary="run_tags",
        back_populates="runs",
        lazy="select",
    )

    __table_args__ = (
        Index("idx_runs_model", "model_id"),
        Index("idx_runs_user", "user_id"),
        Index("idx_runs_status", "status"),
        Index("idx_runs_type", "run_type"),
        Index("idx_runs_created", "created_at"),
        Index("idx_runs_device_dtype", "device", "dtype"),
    )

    def __repr__(self) -> str:
        return f"<Run(id={self.id}, build_name={self.build_name}, status={self.status})>"


class Metric(Base):
    """
    Metric representing a single evaluation measurement.

    Attributes:
        id: Unique identifier
        run_id: Reference to parent run
        category: Metric category (performance, accuracy, efficiency)
        name: Metric name (e.g., seconds_to_first_token)
        display_name: Human-readable display name
        value_numeric: Numeric value
        value_text: Text value (for categorical results)
        unit: Measurement unit (tokens/s, %, ms, GB)
        mean_value: Mean value for aggregated metrics
        std_dev: Standard deviation
        min_value: Minimum value
        max_value: Maximum value
        iteration_values: Per-iteration raw data (JSONB array)
        metadata: Additional metadata (JSONB)
        created_at: Creation timestamp
    """
    __tablename__ = "metrics"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=generate_uuid,
    )
    run_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    category: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    display_name: Mapped[Optional[str]] = mapped_column(String(255))
    value_numeric: Mapped[Optional[float]] = mapped_column(Numeric(20, 6), index=True)
    value_text: Mapped[Optional[str]] = mapped_column(Text)
    unit: Mapped[Optional[str]] = mapped_column(String(50))
    mean_value: Mapped[Optional[float]] = mapped_column(Numeric(20, 6))
    std_dev: Mapped[Optional[float]] = mapped_column(Numeric(20, 6))
    min_value: Mapped[Optional[float]] = mapped_column(Numeric(20, 6))
    max_value: Mapped[Optional[float]] = mapped_column(Numeric(20, 6))
    iteration_values: Mapped[Optional[list]] = mapped_column(FlexibleJSONB)
    metric_metadata: Mapped[dict] = mapped_column("metadata_json", FlexibleJSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    run: Mapped["Run"] = relationship(back_populates="metrics")

    __table_args__ = (
        Index("idx_metrics_run", "run_id"),
        Index("idx_metrics_category", "category"),
        Index("idx_metrics_name", "name"),
        Index("idx_metrics_value", "value_numeric"),
        UniqueConstraint("run_id", "category", "name"),
    )

    def __repr__(self) -> str:
        return f"<Metric(id={self.id}, name={self.name}, value={self.value_numeric})>"


class Tag(Base):
    """
    Tag for flexible organization of runs.

    Attributes:
        id: Unique identifier
        name: Tag name (unique)
        color: Hex color code for UI display
        created_by: User who created the tag
    """
    __tablename__ = "tags"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=generate_uuid,
    )
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    color: Mapped[str] = mapped_column(String(7), default="#6B7280")
    created_by: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False), ForeignKey("users.id")
    )

    # Relationships
    creator: Mapped[Optional["User"]] = relationship(back_populates="created_tags")
    runs: Mapped[list["Run"]] = relationship(
        secondary="run_tags",
        back_populates="tags",
        lazy="select",
    )

    def __repr__(self) -> str:
        return f"<Tag(id={self.id}, name={self.name})>"


class RunTag(Base):
    """
    Many-to-many relationship between runs and tags.
    """
    __tablename__ = "run_tags"

    run_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("runs.id", ondelete="CASCADE"),
        primary_key=True,
    )
    tag_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("tags.id", ondelete="CASCADE"),
        primary_key=True,
    )

    def __repr__(self) -> str:
        return f"<RunTag(run_id={self.run_id}, tag_id={self.tag_id})>"
