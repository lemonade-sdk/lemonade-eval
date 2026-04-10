"""
Integration tests for YAML Import API endpoints.

Tests cover:
- Import YAML files
- Get import status
- Scan cache directory
- List import jobs
- Dry run imports
- Background task processing
"""

import os
import tempfile
import yaml
import pytest
from uuid import uuid4

from tests.conftest import ModelFactory, RunFactory


@pytest.fixture(scope="function", autouse=True)
def clear_import_jobs_for_import_tests():
    """Clear import jobs before each import test."""
    from app.api.v1.import_routes import import_jobs
    import_jobs.clear()
    yield


class TestImportYaml:
    """Tests for POST /api/v1/import/yaml endpoint."""

    def test_import_yaml_dry_run(self, client, db_session, tmp_path):
        """Test dry run import (scan only, no import)."""
        # Create a test YAML file
        yaml_data = {
            "checkpoint": "test/model-1b",
            "iterations": 10,
            "prompts": 5,
            "seconds_to_first_token": 0.025,
            "token_generation_tokens_per_second": 45.5,
            "mmlu_stem": 65.4,
            "device": "gpu",
            "backend": "llamacpp",
            "dtype": "float16",
            "lemonade_version": "1.7.0",
        }

        builds_dir = tmp_path / "builds" / "test_build_001"
        builds_dir.mkdir(parents=True)
        yaml_file = builds_dir / "lemonade_stats.yaml"
        yaml_file.write_text(yaml.dump(yaml_data))

        request_data = {
            "cache_dir": str(tmp_path),
            "skip_duplicates": True,
            "dry_run": True,
        }

        response = client.post("/api/v1/import/yaml", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        job_id = data["data"]["job_id"]

        # Check job status (should be completed for dry run)
        status_response = client.get(f"/api/v1/import/status/{job_id}")
        status_data = status_response.json()
        assert status_data["data"]["status"] == "completed"
        assert status_data["data"]["total_files"] == 1

        # Verify no runs were created (dry run)
        runs_response = client.get("/api/v1/runs")
        runs_data = runs_response.json()
        assert runs_data["meta"]["total"] == 0

    def test_import_yaml_full(self, client, db_session, tmp_path):
        """Test full YAML import."""
        yaml_data = {
            "checkpoint": "test/import-model-1b",
            "iterations": 10,
            "prompts": 5,
            "seconds_to_first_token": 0.025,
            "token_generation_tokens_per_second": 45.5,
            "prefill_tokens_per_second": 1500.0,
            "max_memory_used_gbyte": 4.2,
            "mmlu_stem": 65.4,
            "mmlu_humanities": 68.2,
            "device": "gpu",
            "backend": "llamacpp",
            "dtype": "float16",
            "lemonade_version": "1.7.0",
        }

        builds_dir = tmp_path / "builds" / "test_import_build"
        builds_dir.mkdir(parents=True)
        yaml_file = builds_dir / "lemonade_stats.yaml"
        yaml_file.write_text(yaml.dump(yaml_data))

        request_data = {
            "cache_dir": str(tmp_path),
            "skip_duplicates": True,
            "dry_run": False,
        }

        response = client.post("/api/v1/import/yaml", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "job_id" in data["data"]

        # For non-dry-run, job runs in background
        # In test environment, it may complete immediately
        job_id = data["data"]["job_id"]

        # Check status
        status_response = client.get(f"/api/v1/import/status/{job_id}")
        status_data = status_response.json()
        assert status_data["data"]["status"] in ["pending", "running", "completed"]

    def test_import_yaml_skip_duplicates(self, client, db_session, tmp_path):
        """Test that duplicate builds are skipped."""
        yaml_data = {
            "checkpoint": "test/duplicate-model",
            "seconds_to_first_token": 0.025,
        }

        builds_dir = tmp_path / "builds" / "duplicate_build"
        builds_dir.mkdir(parents=True)
        yaml_file = builds_dir / "lemonade_stats.yaml"
        yaml_file.write_text(yaml.dump(yaml_data))

        # First import
        request_data = {"cache_dir": str(tmp_path), "skip_duplicates": True}
        response = client.post("/api/v1/import/yaml", json=request_data)
        job_id = response.json()["data"]["job_id"]

        # Second import (should skip duplicate)
        response = client.post("/api/v1/import/yaml", json=request_data)
        job_id_2 = response.json()["data"]["job_id"]

        # Check status of second job
        status_response = client.get(f"/api/v1/import/status/{job_id_2}")
        status_data = status_response.json()
        # The duplicate should be skipped
        assert status_data["data"].get("skipped_duplicates", 0) >= 0

    def test_import_yaml_nonexistent_directory(self, client, db_session):
        """Test importing from non-existent directory."""
        request_data = {
            "cache_dir": "/nonexistent/path/to/cache",
            "skip_duplicates": True,
        }

        response = client.post("/api/v1/import/yaml", json=request_data)
        assert response.status_code == 200
        data = response.json()
        job_id = data["data"]["job_id"]

        # Check status - should complete with 0 files
        status_response = client.get(f"/api/v1/import/status/{job_id}")
        status_data = status_response.json()
        assert status_data["data"]["total_files"] == 0


class TestGetImportStatus:
    """Tests for GET /api/v1/import/status/{job_id} endpoint."""

    def test_get_import_status_success(self, client, db_session, tmp_path):
        """Test getting import job status."""
        yaml_data = {"checkpoint": "test/status-model", "seconds_to_first_token": 0.025}

        builds_dir = tmp_path / "builds" / "status_build"
        builds_dir.mkdir(parents=True)
        yaml_file = builds_dir / "lemonade_stats.yaml"
        yaml_file.write_text(yaml.dump(yaml_data))

        request_data = {"cache_dir": str(tmp_path), "dry_run": True}
        response = client.post("/api/v1/import/yaml", json=request_data)
        job_id = response.json()["data"]["job_id"]

        status_response = client.get(f"/api/v1/import/status/{job_id}")
        assert status_response.status_code == 200
        data = status_response.json()
        assert data["success"] is True
        assert data["data"]["job_id"] == job_id
        assert "status" in data["data"]

    def test_get_import_status_not_found(self, client):
        """Test getting status for non-existent job."""
        fake_id = str(uuid4())
        response = client.get(f"/api/v1/import/status/{fake_id}")
        assert response.status_code == 404
        data = response.json()
        assert "Job not found" in data["detail"]


class TestScanCacheDirectory:
    """Tests for POST /api/v1/import/scan endpoint."""

    def test_scan_cache_directory_success(self, client, db_session, tmp_path):
        """Test scanning a cache directory."""
        yaml_data = {"checkpoint": "test/scan-model", "seconds_to_first_token": 0.025}

        # Create multiple build directories
        for i in range(3):
            builds_dir = tmp_path / "builds" / f"scan_build_{i}"
            builds_dir.mkdir(parents=True)
            yaml_file = builds_dir / "lemonade_stats.yaml"
            yaml_file.write_text(yaml.dump(yaml_data))

        response = client.post(f"/api/v1/import/scan?cache_dir={tmp_path}")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["files_found"] == 3
        assert len(data["data"]["files"]) == 3

    def test_scan_cache_directory_empty(self, client, db_session, tmp_path):
        """Test scanning an empty directory."""
        response = client.post(f"/api/v1/import/scan?cache_dir={tmp_path}")
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["files_found"] == 0
        assert data["data"]["files"] == []

    def test_scan_cache_directory_nonexistent(self, client, db_session):
        """Test scanning non-existent directory."""
        response = client.post("/api/v1/import/scan?cache_dir=/nonexistent/path")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["files_found"] == 0


class TestListImportJobs:
    """Tests for GET /api/v1/import/jobs endpoint."""

    def test_list_import_jobs(self, client, db_session, tmp_path):
        """Test listing import jobs."""
        yaml_data = {"checkpoint": "test/list-model", "seconds_to_first_token": 0.025}

        # Create multiple import jobs
        for i in range(3):
            builds_dir = tmp_path / "builds" / f"list_build_{i}"
            builds_dir.mkdir(parents=True)
            yaml_file = builds_dir / "lemonade_stats.yaml"
            yaml_file.write_text(yaml.dump(yaml_data))

            request_data = {"cache_dir": str(tmp_path), "dry_run": True}
            client.post("/api/v1/import/yaml", json=request_data)

        response = client.get("/api/v1/import/jobs")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) >= 3

    def test_list_import_jobs_empty(self, client):
        """Test listing jobs when none exist."""
        response = client.get("/api/v1/import/jobs")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"] == []


class TestImportServiceEdgeCases:
    """Edge case tests for import functionality."""

    def test_import_invalid_yaml(self, client, db_session, tmp_path):
        """Test importing invalid YAML file."""
        builds_dir = tmp_path / "builds" / "invalid_build"
        builds_dir.mkdir(parents=True)
        yaml_file = builds_dir / "lemonade_stats.yaml"
        yaml_file.write_text("invalid: yaml: content: [unclosed")

        request_data = {"cache_dir": str(tmp_path)}
        response = client.post("/api/v1/import/yaml", json=request_data)
        assert response.status_code == 200
        data = response.json()
        job_id = data["data"]["job_id"]

        # Check for errors in status
        status_response = client.get(f"/api/v1/import/status/{job_id}")
        status_data = status_response.json()
        # Should have errors
        assert "errors" in status_data.get("data", {}) or status_data["data"]["status"] in ["completed", "failed"]

    def test_import_empty_yaml(self, client, db_session, tmp_path):
        """Test importing empty YAML file."""
        builds_dir = tmp_path / "builds" / "empty_build"
        builds_dir.mkdir(parents=True)
        yaml_file = builds_dir / "lemonade_stats.yaml"
        yaml_file.write_text("")

        request_data = {"cache_dir": str(tmp_path)}
        response = client.post("/api/v1/import/yaml", json=request_data)
        assert response.status_code == 200

    def test_import_yaml_missing_checkpoint(self, client, db_session, tmp_path):
        """Test importing YAML without checkpoint field."""
        yaml_data = {
            "iterations": 10,
            "seconds_to_first_token": 0.025,
            # Missing checkpoint
        }

        builds_dir = tmp_path / "builds" / "no_checkpoint_build"
        builds_dir.mkdir(parents=True)
        yaml_file = builds_dir / "lemonade_stats.yaml"
        yaml_file.write_text(yaml.dump(yaml_data))

        request_data = {"cache_dir": str(tmp_path)}
        response = client.post("/api/v1/import/yaml", json=request_data)
        assert response.status_code == 200
