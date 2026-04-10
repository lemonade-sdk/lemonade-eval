"""
Pydantic schemas for API request/response validation.

Schemas defined:
- User schemas
- Model schemas
- ModelVersion schemas
- Run schemas
- Metric schemas
- Tag schemas
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# =============================================================================
# Shared schemas
# =============================================================================


class PaginationMeta(BaseModel):
    """Pagination metadata."""
    page: int
    per_page: int
    total: int
    total_pages: int


class APIResponse(BaseModel):
    """Standard API response wrapper."""
    success: bool = True
    data: Any | None = None
    meta: PaginationMeta | None = None
    errors: list = Field(default_factory=list)


# =============================================================================
# User schemas
# =============================================================================


class UserBase(BaseModel):
    """Base user schema."""
    email: str = Field(..., min_length=1, max_length=255)
    name: str = Field(..., min_length=1, max_length=255)
    role: str = Field(default="viewer", max_length=50)


class UserCreate(UserBase):
    """Schema for creating a user."""
    password: str = Field(..., min_length=8)


class UserUpdate(BaseModel):
    """Schema for updating a user."""
    email: str | None = Field(None, min_length=1, max_length=255)
    name: str | None = Field(None, min_length=1, max_length=255)
    role: str | None = Field(None, max_length=50)


class UserResponse(UserBase):
    """Schema for user response."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    is_active: bool
    created_at: datetime
    updated_at: datetime


# =============================================================================
# Model schemas
# =============================================================================


class ModelBase(BaseModel):
    """Base model schema."""
    name: str = Field(..., min_length=1, max_length=255)
    checkpoint: str = Field(..., min_length=1, max_length=500)
    model_type: str = Field(default="llm", max_length=50)
    family: str | None = Field(None, max_length=100)
    parameters: int | None = None
    max_context_length: int | None = None
    architecture: str | None = Field(None, max_length=100)
    license_type: str | None = Field(None, max_length=100)
    hf_repo: str | None = Field(None, max_length=255)
    model_metadata: dict = Field(default_factory=dict, alias="metadata_json")

    model_config = ConfigDict(populate_by_name=True)


class ModelCreate(ModelBase):
    """Schema for creating a model."""
    pass


class ModelUpdate(BaseModel):
    """Schema for updating a model."""
    name: str | None = Field(None, min_length=1, max_length=255)
    model_type: str | None = Field(None, max_length=50)
    family: str | None = Field(None, max_length=100)
    parameters: int | None = None
    max_context_length: int | None = None
    architecture: str | None = Field(None, max_length=100)
    license_type: str | None = Field(None, max_length=100)
    hf_repo: str | None = Field(None, max_length=255)
    model_metadata: dict | None = Field(None, alias="metadata_json")

    model_config = ConfigDict(populate_by_name=True)


class ModelResponse(ModelBase):
    """Schema for model response."""
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: str
    created_by: str | None
    created_at: datetime
    updated_at: datetime

    # Explicitly redefine model_metadata for response
    model_metadata: dict = Field(default_factory=dict, alias="metadata_json")


class ModelListResponse(BaseModel):
    """Schema for model list response."""
    success: bool = True
    data: list[ModelResponse]
    meta: PaginationMeta | None = None


# =============================================================================
# ModelVersion schemas
# =============================================================================


class ModelVersionBase(BaseModel):
    """Base model version schema."""
    version: str = Field(..., max_length=50)
    quantization: str | None = Field(None, max_length=50)
    dtype: str | None = Field(None, max_length=50)
    backend: str | None = Field(None, max_length=100)
    config: dict = Field(default_factory=dict)
    is_default: bool = False


class ModelVersionCreate(ModelVersionBase):
    """Schema for creating a model version."""
    model_id: str


class ModelVersionResponse(ModelVersionBase):
    """Schema for model version response."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    model_id: str
    created_at: datetime


# =============================================================================
# Run schemas
# =============================================================================


class RunBase(BaseModel):
    """Base run schema."""
    model_id: str
    build_name: str = Field(..., min_length=1, max_length=255)
    run_type: str = Field(..., max_length=50)
    status: str = Field(default="pending", max_length=50)
    device: str | None = Field(None, max_length=50)
    backend: str | None = Field(None, max_length=100)
    dtype: str | None = Field(None, max_length=50)
    config: dict = Field(default_factory=dict)
    system_info: dict = Field(default_factory=dict)
    lemonade_version: str | None = Field(None, max_length=20)
    build_uid: str | None = Field(None, max_length=100)


class RunCreate(RunBase):
    """Schema for creating a run."""
    user_id: str | None = None


class RunUpdate(BaseModel):
    """Schema for updating a run."""
    status: str | None = Field(None, max_length=50)
    status_message: str | None = None
    device: str | None = Field(None, max_length=50)
    backend: str | None = Field(None, max_length=100)
    dtype: str | None = Field(None, max_length=50)
    config: dict | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float | None = None
    system_info: dict | None = None
    error_log: str | None = None


class RunResponse(RunBase):
    """Schema for run response."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str | None
    status_message: str | None
    started_at: datetime | None
    completed_at: datetime | None
    duration_seconds: float | None
    log_file_path: str | None
    error_log: str | None
    created_at: datetime
    updated_at: datetime


class RunDetailResponse(RunResponse):
    """Schema for run detail response with related data."""
    metrics: list["MetricResponse"] = Field(default_factory=list)


class RunListResponse(BaseModel):
    """Schema for run list response."""
    success: bool = True
    data: list[RunResponse]
    meta: PaginationMeta | None = None


# =============================================================================
# Metric schemas
# =============================================================================


class MetricBase(BaseModel):
    """Base metric schema."""
    category: str = Field(..., max_length=50)
    name: str = Field(..., min_length=1, max_length=255)
    display_name: str | None = Field(None, max_length=255)
    value_numeric: float | None = None
    value_text: str | None = None
    unit: str | None = Field(None, max_length=50)
    mean_value: float | None = None
    std_dev: float | None = None
    min_value: float | None = None
    max_value: float | None = None
    iteration_values: list | None = None
    metric_metadata: dict = Field(default_factory=dict, alias="metadata_json")

    model_config = ConfigDict(populate_by_name=True)


class MetricCreate(MetricBase):
    """Schema for creating a metric."""
    run_id: str


class MetricBulkCreate(BaseModel):
    """Schema for bulk creating metrics."""
    metrics: list[MetricCreate]


class MetricResponse(MetricBase):
    """Schema for metric response."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    run_id: str
    created_at: datetime


class MetricListResponse(BaseModel):
    """Schema for metric list response."""
    success: bool = True
    data: list[MetricResponse]
    meta: PaginationMeta | None = None


# =============================================================================
# Tag schemas
# =============================================================================


class TagBase(BaseModel):
    """Base tag schema."""
    name: str = Field(..., min_length=1, max_length=100)
    color: str = Field(default="#6B7280", max_length=7)


class TagCreate(TagBase):
    """Schema for creating a tag."""
    pass


class TagUpdate(BaseModel):
    """Schema for updating a tag."""
    name: str | None = Field(None, min_length=1, max_length=100)
    color: str | None = Field(None, max_length=7)


class TagResponse(TagBase):
    """Schema for tag response."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    created_by: str | None


class TagListResponse(BaseModel):
    """Schema for tag list response."""
    success: bool = True
    data: list[TagResponse]


# =============================================================================
# Import schemas
# =============================================================================


class ImportRequest(BaseModel):
    """Schema for YAML import request."""
    cache_dir: str
    skip_duplicates: bool = True
    dry_run: bool = False


class ImportJobStatus(BaseModel):
    """Schema for import job status."""
    job_id: str
    status: str  # pending, running, completed, failed
    total_files: int
    processed_files: int
    imported_runs: int
    skipped_duplicates: int
    errors: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    completed_at: datetime | None = None


# =============================================================================
# WebSocket schemas
# =============================================================================


class WSEvent(BaseModel):
    """Base WebSocket event schema."""
    event_type: str
    data: dict


class RunStatusEvent(WSEvent):
    """WebSocket event for run status updates."""
    event_type: str = "run_status"
    run_id: str
    status: str
    message: str | None = None


class MetricsStreamEvent(WSEvent):
    """WebSocket event for metrics streaming."""
    event_type: str = "metrics_stream"
    run_id: str
    metrics: list[dict]


# Rebuild forward references
RunDetailResponse.model_rebuild()
