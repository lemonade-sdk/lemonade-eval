"""
Unit tests for Model Service.

Tests cover:
- Model CRUD operations
- Pagination
- Filtering
- Search functionality
- Model versions
- Edge cases
"""

import pytest
from uuid import uuid4

from app.services.models import ModelService
from app.schemas import ModelCreate, ModelUpdate
from tests.conftest import ModelFactory


class TestModelServiceGetModels:
    """Tests for ModelService.get_models method."""

    def test_get_models_empty(self, db_session):
        """Test getting models when database is empty."""
        service = ModelService(db_session)
        models, meta = service.get_models()

        assert models == []
        assert meta.total == 0
        assert meta.page == 1
        assert meta.per_page == 20

    def test_get_models_with_data(self, db_session):
        """Test getting models with data."""
        ModelFactory.create(db_session, name="Model A")
        ModelFactory.create(db_session, name="Model B")
        ModelFactory.create(db_session, name="Model C")

        service = ModelService(db_session)
        models, meta = service.get_models()

        assert len(models) == 3
        assert meta.total == 3

    def test_get_models_pagination(self, db_session):
        """Test pagination."""
        for i in range(25):
            ModelFactory.create(db_session, name=f"Model {i}")

        service = ModelService(db_session)
        models, meta = service.get_models(page=1, per_page=10)

        assert len(models) == 10
        assert meta.total == 25
        assert meta.total_pages == 3

    def test_get_models_search_by_name(self, db_session):
        """Test search by name."""
        ModelFactory.create(db_session, name="Llama Model")
        ModelFactory.create(db_session, name="Qwen Model")
        ModelFactory.create(db_session, name="Phi Model")

        service = ModelService(db_session)
        models, meta = service.get_models(search="Llama")

        assert len(models) == 1
        assert "Llama" in models[0].name

    def test_get_models_search_by_checkpoint(self, db_session):
        """Test search by checkpoint."""
        ModelFactory.create(db_session, checkpoint="meta/llama-2b")
        ModelFactory.create(db_session, checkpoint="qwen-1b")

        service = ModelService(db_session)
        models, meta = service.get_models(search="llama")

        assert len(models) >= 1

    def test_get_models_filter_by_family(self, db_session):
        """Test filter by family."""
        ModelFactory.create(db_session, family="Llama")
        ModelFactory.create(db_session, family="Llama")
        ModelFactory.create(db_session, family="Qwen")

        service = ModelService(db_session)
        models, meta = service.get_models(family="Llama")

        assert len(models) == 2
        assert all(m.family == "Llama" for m in models)

    def test_get_models_filter_by_type(self, db_session):
        """Test filter by model type."""
        ModelFactory.create(db_session, model_type="llm")
        ModelFactory.create(db_session, model_type="vlm")
        ModelFactory.create(db_session, model_type="embedding")

        service = ModelService(db_session)
        models, meta = service.get_models(model_type="vlm")

        assert len(models) == 1
        assert models[0].model_type == "vlm"


class TestModelServiceGetModel:
    """Tests for ModelService.get_model method."""

    def test_get_model_success(self, db_session):
        """Test getting a model by ID."""
        model = ModelFactory.create(db_session)

        service = ModelService(db_session)
        result = service.get_model(model.id)

        assert result is not None
        assert result.id == model.id
        assert result.name == model.name

    def test_get_model_not_found(self, db_session):
        """Test getting non-existent model."""
        service = ModelService(db_session)
        result = service.get_model(str(uuid4()))

        assert result is None


class TestModelServiceGetModelByCheckpoint:
    """Tests for ModelService.get_model_by_checkpoint method."""

    def test_get_by_checkpoint_success(self, db_session):
        """Test getting model by checkpoint."""
        model = ModelFactory.create(db_session, checkpoint="test/unique-checkpoint")

        service = ModelService(db_session)
        result = service.get_model_by_checkpoint("test/unique-checkpoint")

        assert result is not None
        assert result.checkpoint == "test/unique-checkpoint"

    def test_get_by_checkpoint_not_found(self, db_session):
        """Test getting non-existent checkpoint."""
        service = ModelService(db_session)
        result = service.get_model_by_checkpoint("nonexistent/checkpoint")

        assert result is None


class TestModelServiceCreateModel:
    """Tests for ModelService.create_model method."""

    def test_create_model_minimal(self, db_session):
        """Test creating model with minimal data."""
        model_data = ModelCreate(
            name="Test Model",
            checkpoint="test/checkpoint",
        )

        service = ModelService(db_session)
        result = service.create_model(model_data)

        assert result.id is not None
        assert result.name == "Test Model"
        assert result.checkpoint == "test/checkpoint"
        assert result.model_type == "llm"  # default

    def test_create_model_full(self, db_session):
        """Test creating model with full data."""
        model_data = ModelCreate(
            name="Complete Model",
            checkpoint="test/complete-7b",
            model_type="llm",
            family="Test",
            parameters=7000000000,
            max_context_length=4096,
        )

        service = ModelService(db_session)
        result = service.create_model(model_data)

        assert result.parameters == 7000000000
        assert result.max_context_length == 4096

    def test_create_model_with_creator(self, db_session, test_user):
        """Test creating model with creator."""
        model_data = ModelCreate(
            name="User Model",
            checkpoint="test/user-model",
        )

        service = ModelService(db_session)
        result = service.create_model(model_data, created_by=test_user.id)

        assert result is not None


class TestModelServiceUpdateModel:
    """Tests for ModelService.update_model method."""

    def test_update_model_name(self, db_session):
        """Test updating model name."""
        model = ModelFactory.create(db_session)

        service = ModelService(db_session)
        update_data = ModelUpdate(name="Updated Name")
        result = service.update_model(model.id, update_data)

        assert result is not None
        assert result.name == "Updated Name"

    def test_update_model_metadata(self, db_session):
        """Test updating model metadata."""
        model = ModelFactory.create(db_session)

        service = ModelService(db_session)
        update_data = ModelUpdate(model_metadata={"new_key": "new_value"})
        result = service.update_model(model.id, update_data)

        assert result is not None
        assert result.model_metadata["new_key"] == "new_value"

    def test_update_model_not_found(self, db_session):
        """Test updating non-existent model."""
        service = ModelService(db_session)
        update_data = ModelUpdate(name="Updated")
        result = service.update_model(str(uuid4()), update_data)

        assert result is None

    def test_update_partial_fields(self, db_session):
        """Test that only provided fields are updated."""
        model = ModelFactory.create(db_session, name="Original", family="Test")

        service = ModelService(db_session)
        update_data = ModelUpdate(name="New Name")  # Only updating name
        result = service.update_model(model.id, update_data)

        assert result.name == "New Name"
        assert result.family == "Test"  # Unchanged


class TestModelServiceDeleteModel:
    """Tests for ModelService.delete_model method."""

    def test_delete_model_success(self, db_session):
        """Test deleting a model."""
        model = ModelFactory.create(db_session)

        service = ModelService(db_session)
        result = service.delete_model(model.id)

        assert result is True

        # Verify deleted
        assert service.get_model(model.id) is None

    def test_delete_model_not_found(self, db_session):
        """Test deleting non-existent model."""
        service = ModelService(db_session)
        result = service.delete_model(str(uuid4()))

        assert result is False


class TestModelServiceGetModelVersions:
    """Tests for ModelService.get_model_versions method."""

    def test_get_model_versions_empty(self, db_session):
        """Test getting versions when none exist."""
        model = ModelFactory.create(db_session)

        service = ModelService(db_session)
        versions = service.get_model_versions(model.id)

        assert versions == []

    def test_get_model_versions(self, db_session):
        """Test getting model versions."""
        from app.models import ModelVersion

        model = ModelFactory.create(db_session)

        # Create versions
        version = ModelVersion(
            id=str(uuid4()),
            model_id=model.id,
            version="v1.0",
            quantization="int4",
        )
        db_session.add(version)
        db_session.commit()

        service = ModelService(db_session)
        versions = service.get_model_versions(model.id)

        assert len(versions) == 1
        assert versions[0]["version"] == "v1.0"


class TestModelServiceGetModelRuns:
    """Tests for ModelService.get_model_runs method."""

    def test_get_model_runs_empty(self, db_session):
        """Test getting runs when none exist."""
        model = ModelFactory.create(db_session)

        service = ModelService(db_session)
        runs = service.get_model_runs(model.id)

        assert runs == []

    def test_get_model_runs(self, db_session):
        """Test getting runs for a model."""
        from tests.conftest import RunFactory

        model = ModelFactory.create(db_session)
        RunFactory.create(db_session, model_id=model.id)
        RunFactory.create(db_session, model_id=model.id)

        service = ModelService(db_session)
        runs = service.get_model_runs(model.id)

        assert len(runs) == 2


class TestModelServiceSearchFamilies:
    """Tests for ModelService.search_families method."""

    def test_search_families_empty(self, db_session):
        """Test getting families when none exist."""
        service = ModelService(db_session)
        families = service.search_families()

        assert families == []

    def test_search_families_unique(self, db_session):
        """Test that families are unique."""
        ModelFactory.create(db_session, family="Llama")
        ModelFactory.create(db_session, family="Llama")  # Duplicate
        ModelFactory.create(db_session, family="Qwen")
        ModelFactory.create(db_session, family="Phi")

        service = ModelService(db_session)
        families = service.search_families()

        assert len(families) == 3
        assert set(families) == {"Llama", "Qwen", "Phi"}

    def test_search_families_excludes_none(self, db_session):
        """Test that None families are excluded."""
        ModelFactory.create(db_session, family=None)
        ModelFactory.create(db_session, family="Llama")

        service = ModelService(db_session)
        families = service.search_families()

        assert "Llama" in families
        assert None not in families
