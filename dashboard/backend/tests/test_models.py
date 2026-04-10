"""
Integration tests for Model API endpoints.

Tests cover:
- List models with pagination and filtering
- Create model (success and duplicate cases)
- Get single model
- Update model
- Delete model
- Get model versions and runs
- Search families
"""

import pytest
from uuid import uuid4

from app.models import Model
from app.schemas import ModelCreate


class TestListModels:
    """Tests for GET /api/v1/models endpoint."""

    def test_list_models_empty(self, client, db_session):
        """Test listing models when database is empty."""
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"] == []
        assert data["meta"]["total"] == 0

    def test_list_models_with_data(self, client, db_session):
        """Test listing models with data in database."""
        # Create multiple models
        from tests.conftest import ModelFactory
        ModelFactory.create(db_session, name="Model A", family="Test")
        ModelFactory.create(db_session, name="Model B", family="Test")
        ModelFactory.create(db_session, name="Model C", family="Other")

        response = client.get("/api/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["meta"]["total"] == 3
        assert len(data["data"]) == 3

    def test_list_models_pagination(self, client, db_session):
        """Test model listing pagination."""
        from tests.conftest import ModelFactory

        # Create 25 models
        for i in range(25):
            ModelFactory.create(db_session, name=f"Model {i}")

        # Get first page
        response = client.get("/api/v1/models?page=1&per_page=10")
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 10
        assert data["meta"]["page"] == 1
        assert data["meta"]["per_page"] == 10
        assert data["meta"]["total"] == 25
        assert data["meta"]["total_pages"] == 3

        # Get second page
        response = client.get("/api/v1/models?page=2&per_page=10")
        data = response.json()
        assert len(data["data"]) == 10
        assert data["meta"]["page"] == 2

        # Get third page
        response = client.get("/api/v1/models?page=3&per_page=10")
        data = response.json()
        assert len(data["data"]) == 5
        assert data["meta"]["page"] == 3

    def test_list_models_search(self, client, db_session):
        """Test model listing with search filter."""
        from tests.conftest import ModelFactory

        ModelFactory.create(db_session, name="Llama Model", checkpoint="meta/llama-2b")
        ModelFactory.create(db_session, name="Qwen Model", checkpoint="qwen-1b")
        ModelFactory.create(db_session, name="Phi Model", checkpoint="phi-2")

        # Search by name
        response = client.get("/api/v1/models?search=Llama")
        data = response.json()
        assert data["meta"]["total"] == 1
        assert data["data"][0]["name"] == "Llama Model"

        # Search by checkpoint
        response = client.get("/api/v1/models?search=qwen")
        data = response.json()
        assert data["meta"]["total"] == 1

    def test_list_models_filter_by_family(self, client, db_session):
        """Test model listing with family filter."""
        from tests.conftest import ModelFactory

        ModelFactory.create(db_session, name="Llama-2b", family="Llama")
        ModelFactory.create(db_session, name="Llama-7b", family="Llama")
        ModelFactory.create(db_session, name="Qwen-1b", family="Qwen")
        ModelFactory.create(db_session, name="Phi-2", family="Phi")

        response = client.get("/api/v1/models?family=Llama")
        data = response.json()
        assert data["meta"]["total"] == 2
        assert all(m["family"] == "Llama" for m in data["data"])

    def test_list_models_filter_by_type(self, client, db_session):
        """Test model listing with model_type filter."""
        from tests.conftest import ModelFactory

        ModelFactory.create(db_session, name="LLM Model", model_type="llm")
        ModelFactory.create(db_session, name="VLM Model", model_type="vlm")
        ModelFactory.create(db_session, name="Embedding Model", model_type="embedding")

        response = client.get("/api/v1/models?model_type=vlm")
        data = response.json()
        assert data["meta"]["total"] == 1
        assert data["data"][0]["model_type"] == "vlm"


class TestCreateModel:
    """Tests for POST /api/v1/models endpoint."""

    def test_create_model_minimal(self, client, db_session):
        """Test creating a model with minimal required fields."""
        model_data = {
            "name": "Test Model",
            "checkpoint": "test/checkpoint-1b",
        }
        response = client.post("/api/v1/models", json=model_data)
        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert data["data"]["name"] == "Test Model"
        assert data["data"]["checkpoint"] == "test/checkpoint-1b"
        assert data["data"]["model_type"] == "llm"  # default

    def test_create_model_full(self, client, db_session):
        """Test creating a model with all fields."""
        model_data = {
            "name": "Complete Test Model",
            "checkpoint": "test/complete-7b",
            "model_type": "llm",
            "family": "Test",
            "parameters": 7000000000,
            "max_context_length": 4096,
            "architecture": "transformer",
            "license_type": "MIT",
            "hf_repo": "test/complete-7b",
            "metadata": {"custom_key": "custom_value"},
        }
        response = client.post("/api/v1/models", json=model_data)
        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert data["data"]["parameters"] == 7000000000
        assert data["data"]["max_context_length"] == 4096

    def test_create_model_duplicate_checkpoint(self, client, db_session):
        """Test that duplicate checkpoints are rejected."""
        from tests.conftest import ModelFactory

        # Create initial model
        ModelFactory.create(db_session, checkpoint="test/duplicate-1b")

        # Try to create duplicate
        model_data = {
            "name": "Duplicate Model",
            "checkpoint": "test/duplicate-1b",
        }
        response = client.post("/api/v1/models", json=model_data)
        assert response.status_code == 400
        data = response.json()
        assert "already exists" in data["detail"]

    def test_create_model_invalid_type(self, client, db_session):
        """Test creating model with invalid model_type."""
        model_data = {
            "name": "Invalid Model",
            "checkpoint": "test/invalid",
            "model_type": "invalid_type",
        }
        response = client.post("/api/v1/models", json=model_data)
        # Should either accept any string or validate against enum
        assert response.status_code in [201, 422]


class TestGetModel:
    """Tests for GET /api/v1/models/{model_id} endpoint."""

    def test_get_model_success(self, client, test_model):
        """Test getting a model by ID."""
        response = client.get(f"/api/v1/models/{test_model.id}")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["id"] == test_model.id
        assert data["data"]["name"] == test_model.name
        assert data["data"]["checkpoint"] == test_model.checkpoint

    def test_get_model_not_found(self, client):
        """Test getting a non-existent model."""
        fake_id = str(uuid4())
        response = client.get(f"/api/v1/models/{fake_id}")
        assert response.status_code == 404
        data = response.json()
        assert data["detail"] == "Model not found"


class TestUpdateModel:
    """Tests for PUT /api/v1/models/{model_id} endpoint."""

    def test_update_model_name(self, client, test_model):
        """Test updating model name."""
        update_data = {"name": "Updated Model Name"}
        response = client.put(f"/api/v1/models/{test_model.id}", json=update_data)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["name"] == "Updated Model Name"
        # Verify checkpoint unchanged
        assert data["data"]["checkpoint"] == test_model.checkpoint

    def test_update_model_metadata(self, client, test_model):
        """Test updating model metadata."""
        update_data = {"model_metadata": {"new_key": "new_value", "updated": True}}
        response = client.put(f"/api/v1/models/{test_model.id}", json=update_data)
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["model_metadata"]["new_key"] == "new_value"

    def test_update_model_not_found(self, client):
        """Test updating a non-existent model."""
        fake_id = str(uuid4())
        update_data = {"name": "Updated"}
        response = client.put(f"/api/v1/models/{fake_id}", json=update_data)
        assert response.status_code == 404


class TestDeleteModel:
    """Tests for DELETE /api/v1/models/{model_id} endpoint."""

    def test_delete_model_success(self, client, db_session, test_model):
        """Test deleting a model."""
        response = client.delete(f"/api/v1/models/{test_model.id}")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "deleted successfully" in data["data"]["message"]

        # Verify model is deleted
        response = client.get(f"/api/v1/models/{test_model.id}")
        assert response.status_code == 404

    def test_delete_model_cascades(self, client, db_session, test_model, test_run):
        """Test that deleting a model cascades to runs and metrics."""
        from tests.conftest import MetricFactory
        # Create metrics for the run
        MetricFactory.create(db_session, run_id=test_run.id)

        # Delete model
        response = client.delete(f"/api/v1/models/{test_model.id}")
        assert response.status_code == 200

        # Verify run is also deleted (cascade)
        response = client.get(f"/api/v1/runs/{test_run.id}")
        assert response.status_code == 404

    def test_delete_model_not_found(self, client):
        """Test deleting a non-existent model."""
        fake_id = str(uuid4())
        response = client.delete(f"/api/v1/models/{fake_id}")
        assert response.status_code == 404


class TestModelVersions:
    """Tests for GET /api/v1/models/{model_id}/versions endpoint."""

    def test_get_model_versions(self, client, test_model):
        """Test getting model versions."""
        response = client.get(f"/api/v1/models/{test_model.id}/versions")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert isinstance(data["data"], list)

    def test_get_model_versions_not_found(self, client):
        """Test getting versions for non-existent model."""
        fake_id = str(uuid4())
        response = client.get(f"/api/v1/models/{fake_id}/versions")
        assert response.status_code == 404


class TestModelRuns:
    """Tests for GET /api/v1/models/{model_id}/runs endpoint."""

    def test_get_model_runs(self, client, test_model, db_session):
        """Test getting runs for a model."""
        from tests.conftest import RunFactory

        # Create runs for this model
        RunFactory.create(db_session, model_id=test_model.id, build_name="build_1")
        RunFactory.create(db_session, model_id=test_model.id, build_name="build_2")

        response = client.get(f"/api/v1/models/{test_model.id}/runs")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) == 2

    def test_get_model_runs_not_found(self, client):
        """Test getting runs for non-existent model."""
        fake_id = str(uuid4())
        response = client.get(f"/api/v1/models/{fake_id}/runs")
        assert response.status_code == 404


class TestListFamilies:
    """Tests for GET /api/v1/models/families/list endpoint."""

    def test_list_families(self, client, db_session):
        """Test listing unique model families."""
        from tests.conftest import ModelFactory

        ModelFactory.create(db_session, family="Llama")
        ModelFactory.create(db_session, family="Llama")  # Duplicate family
        ModelFactory.create(db_session, family="Qwen")
        ModelFactory.create(db_session, family="Phi")

        response = client.get("/api/v1/models/families/list")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) == 3  # Unique families
        assert set(data["data"]) == {"Llama", "Qwen", "Phi"}

    def test_list_families_empty(self, client, db_session):
        """Test listing families when no models exist."""
        response = client.get("/api/v1/models/families/list")
        assert response.status_code == 200
        data = response.json()
        assert data["data"] == []
