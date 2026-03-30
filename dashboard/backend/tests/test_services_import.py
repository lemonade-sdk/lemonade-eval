"""
Unit tests for Import Service.

Tests cover:
- YAML file parsing
- Cache directory scanning
- Model info extraction
- Run info extraction
- Metric extraction
- Import operations
- Deduplication
- Error handling
"""

import os
import tempfile
import yaml
import pytest
from uuid import uuid4

from app.services.import_service import ImportService
from tests.conftest import ModelFactory, RunFactory


class TestImportServiceScanCacheDir:
    """Tests for ImportService.scan_cache_dir method."""

    def test_scan_cache_dir_success(self, db_session, tmp_path):
        """Test scanning a valid cache directory."""
        # Create directory structure
        builds_dir = tmp_path / "builds" / "test_build"
        builds_dir.mkdir(parents=True)
        yaml_file = builds_dir / "lemonade_stats.yaml"
        yaml_file.write_text("checkpoint: test/model")

        service = ImportService(db_session)
        discovered = service.scan_cache_dir(str(tmp_path))

        assert len(discovered) == 1
        assert discovered[0]["build_name"] == "test_build"

    def test_scan_cache_dir_multiple_files(self, db_session, tmp_path):
        """Test scanning directory with multiple files."""
        for i in range(3):
            builds_dir = tmp_path / "builds" / f"build_{i}"
            builds_dir.mkdir(parents=True)
            yaml_file = builds_dir / "lemonade_stats.yaml"
            yaml_file.write_text("checkpoint: test/model")

        service = ImportService(db_session)
        discovered = service.scan_cache_dir(str(tmp_path))

        assert len(discovered) == 3

    def test_scan_cache_dir_nonexistent(self, db_session):
        """Test scanning non-existent directory."""
        service = ImportService(db_session)
        discovered = service.scan_cache_dir("/nonexistent/path")

        assert discovered == []
        assert len(service.errors) > 0

    def test_scan_cache_dir_empty(self, db_session, tmp_path):
        """Test scanning empty directory."""
        service = ImportService(db_session)
        discovered = service.scan_cache_dir(str(tmp_path))

        assert discovered == []


class TestImportServiceParseYamlFile:
    """Tests for ImportService.parse_yaml_file method."""

    def test_parse_yaml_success(self, db_session, tmp_path):
        """Test parsing valid YAML file."""
        yaml_data = {
            "checkpoint": "test/model-1b",
            "seconds_to_first_token": 0.025,
            "token_generation_tokens_per_second": 45.5,
        }

        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml.dump(yaml_data))

        service = ImportService(db_session)
        result = service.parse_yaml_file(str(yaml_file))

        assert result is not None
        assert result["checkpoint"] == "test/model-1b"
        assert result["seconds_to_first_token"] == 0.025

    def test_parse_yaml_empty(self, db_session, tmp_path):
        """Test parsing empty YAML file."""
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")

        service = ImportService(db_session)
        result = service.parse_yaml_file(str(yaml_file))

        assert result == {}

    def test_parse_yaml_invalid(self, db_session, tmp_path):
        """Test parsing invalid YAML."""
        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text("invalid: yaml: [unclosed")

        service = ImportService(db_session)
        result = service.parse_yaml_file(str(yaml_file))

        assert result is None
        assert len(service.errors) > 0

    def test_parse_yaml_nonexistent(self, db_session):
        """Test parsing non-existent file."""
        service = ImportService(db_session)
        result = service.parse_yaml_file("/nonexistent/file.yaml")

        assert result is None


class TestImportServiceCheckDuplicate:
    """Tests for ImportService.check_duplicate method."""

    def test_check_duplicate_not_found(self, db_session):
        """Test checking for non-existent duplicate."""
        service = ImportService(db_session)
        result = service.check_duplicate("unique_build", "/cache")

        assert result is False

    def test_check_duplicate_exists(self, db_session):
        """Test checking for existing duplicate."""
        run = RunFactory.create(db_session, build_name="existing_build")

        service = ImportService(db_session)
        result = service.check_duplicate("existing_build", "/cache")

        assert result is True


class TestImportServiceExtractModelInfo:
    """Tests for ImportService.extract_model_info method."""

    def test_extract_model_info_basic(self, db_session):
        """Test extracting basic model info."""
        yaml_data = {
            "checkpoint": "meta/llama-2b",
        }

        service = ImportService(db_session)
        result = service.extract_model_info(yaml_data)

        assert result["name"] == "llama-2b"
        assert result["checkpoint"] == "meta/llama-2b"
        assert result["model_type"] == "llm"
        assert result["family"] == "Llama"

    def test_extract_model_info_qwen(self, db_session):
        """Test extracting Qwen model info."""
        yaml_data = {
            "checkpoint": "qwen-1b",
        }

        service = ImportService(db_session)
        result = service.extract_model_info(yaml_data)

        assert result["family"] == "Qwen"

    def test_extract_model_info_phi(self, db_session):
        """Test extracting Phi model info."""
        yaml_data = {
            "checkpoint": "phi-2",
        }

        service = ImportService(db_session)
        result = service.extract_model_info(yaml_data)

        assert result["family"] == "Phi"

    def test_extract_model_info_vlm(self, db_session):
        """Test extracting VLM model info."""
        yaml_data = {
            "checkpoint": "test/vlm-vision",
        }

        service = ImportService(db_session)
        result = service.extract_model_info(yaml_data)

        assert result["model_type"] == "vlm"

    def test_extract_model_info_embedding(self, db_session):
        """Test extracting embedding model info."""
        yaml_data = {
            "checkpoint": "test/embedding-model",
        }

        service = ImportService(db_session)
        result = service.extract_model_info(yaml_data)

        assert result["model_type"] == "embedding"

    def test_extract_model_info_unknown(self, db_session):
        """Test extracting unknown model type."""
        yaml_data = {
            "checkpoint": "unknown-model",
        }

        service = ImportService(db_session)
        result = service.extract_model_info(yaml_data)

        assert result["model_type"] == "llm"  # default
        assert result["family"] is None


class TestImportServiceExtractRunInfo:
    """Tests for ImportService.extract_run_info method."""

    def test_extract_run_info_benchmark(self, db_session):
        """Test extracting benchmark run info."""
        yaml_data = {
            "iterations": 10,
            "prompts": 5,
            "device": "gpu",
            "backend": "llamacpp",
        }

        service = ImportService(db_session)
        result = service.extract_run_info(yaml_data, "test_build", "/path/to/file")

        assert result["build_name"] == "test_build"
        assert result["run_type"] == "benchmark"
        assert result["device"] == "gpu"

    def test_extract_run_info_mmlu(self, db_session):
        """Test extracting MMLU run info."""
        yaml_data = {
            "mmlu_stem": 65.4,
            "mmlu_humanities": 68.2,
        }

        service = ImportService(db_session)
        result = service.extract_run_info(yaml_data, "test_build", "/path/to/file")

        assert result["run_type"] == "accuracy-mmlu"

    def test_extract_run_info_humaneval(self, db_session):
        """Test extracting HumanEval run info."""
        yaml_data = {
            "humaneval_pass": 42.5,
        }

        service = ImportService(db_session)
        result = service.extract_run_info(yaml_data, "test_build", "/path/to/file")

        assert result["run_type"] == "accuracy-humaneval"

    def test_extract_run_info_perplexity(self, db_session):
        """Test extracting perplexity run info."""
        yaml_data = {
            "perplexity": 15.2,
        }

        service = ImportService(db_session)
        result = service.extract_run_info(yaml_data, "test_build", "/path/to/file")

        assert result["run_type"] == "perplexity"

    def test_extract_run_info_config(self, db_session):
        """Test extracting config from run info."""
        yaml_data = {
            "iterations": 100,
            "prompts": 10,
            "output_tokens": 256,
        }

        service = ImportService(db_session)
        result = service.extract_run_info(yaml_data, "test_build", "/path/to/file")

        assert result["config"]["iterations"] == 100
        assert result["config"]["prompts"] == 10


class TestImportServiceExtractMetrics:
    """Tests for ImportService.extract_metrics method."""

    def test_extract_performance_metrics(self, db_session):
        """Test extracting performance metrics."""
        yaml_data = {
            "seconds_to_first_token": 0.025,
            "token_generation_tokens_per_second": 45.5,
        }

        service = ImportService(db_session)
        # Create a run for the metrics
        run = RunFactory.create(db_session)

        # Mock the run_id assignment
        metrics = service.extract_metrics(yaml_data, run.id)

        assert len(metrics) == 2
        assert metrics[0].category == "performance"

    def test_extract_accuracy_metrics(self, db_session):
        """Test extracting accuracy metrics."""
        yaml_data = {
            "mmlu_stem": 65.4,
            "mmlu_humanities": 68.2,
            "humaneval_pass": 42.5,
        }

        service = ImportService(db_session)
        run = RunFactory.create(db_session)
        metrics = service.extract_metrics(yaml_data, run.id)

        assert len(metrics) == 3
        assert all(m.category == "accuracy" for m in metrics)

    def test_extract_mixed_metrics(self, db_session):
        """Test extracting mixed metrics."""
        yaml_data = {
            "seconds_to_first_token": 0.025,
            "token_generation_tokens_per_second": 45.5,
            "mmlu_stem": 65.4,
        }

        service = ImportService(db_session)
        run = RunFactory.create(db_session)
        metrics = service.extract_metrics(yaml_data, run.id)

        assert len(metrics) == 3
        categories = set(m.category for m in metrics)
        assert categories == {"performance", "accuracy"}


class TestImportServiceImportFile:
    """Tests for ImportService.import_file method."""

    def test_import_file_success(self, db_session, tmp_path):
        """Test importing a single file."""
        yaml_data = {
            "checkpoint": "test/import-model",
            "seconds_to_first_token": 0.025,
            "token_generation_tokens_per_second": 45.5,
        }

        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml.dump(yaml_data))

        service = ImportService(db_session)
        success, message = service.import_file(
            file_path=str(yaml_file),
            build_name="test_build",
            skip_duplicates=True,
        )

        assert success is True
        assert "Imported" in message

    def test_import_file_skip_duplicate(self, db_session, tmp_path):
        """Test skipping duplicate import."""
        # Create existing run
        RunFactory.create(db_session, build_name="duplicate_build")

        yaml_data = {"checkpoint": "test/model"}
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml.dump(yaml_data))

        service = ImportService(db_session)
        success, message = service.import_file(
            file_path=str(yaml_file),
            build_name="duplicate_build",
            skip_duplicates=True,
        )

        assert success is True
        assert "Skipped duplicate" in message

    def test_import_file_invalid_yaml(self, db_session, tmp_path):
        """Test importing invalid YAML."""
        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text("invalid: [unclosed")

        service = ImportService(db_session)
        success, message = service.import_file(
            file_path=str(yaml_file),
            build_name="invalid_build",
        )

        assert success is False
        assert "Failed to parse" in message


class TestImportServiceImportDirectory:
    """Tests for ImportService.import_directory method."""

    def test_import_directory_dry_run(self, db_session, tmp_path):
        """Test dry run import."""
        for i in range(3):
            builds_dir = tmp_path / "builds" / f"build_{i}"
            builds_dir.mkdir(parents=True)
            yaml_file = builds_dir / "lemonade_stats.yaml"
            yaml_file.write_text("checkpoint: test/model")

        service = ImportService(db_session)
        result = service.import_directory(
            cache_dir=str(tmp_path),
            skip_duplicates=True,
            dry_run=True,
        )

        assert result["status"] == "dry_run"
        assert result["total_files"] == 3
        assert result["imported_runs"] == 0

    def test_import_directory_full(self, db_session, tmp_path):
        """Test full import."""
        yaml_data = {
            "checkpoint": f"test/model-{uuid4()}",
            "seconds_to_first_token": 0.025,
        }

        builds_dir = tmp_path / "builds" / "test_build"
        builds_dir.mkdir(parents=True)
        yaml_file = builds_dir / "lemonade_stats.yaml"
        yaml_file.write_text(yaml.dump(yaml_data))

        service = ImportService(db_session)
        result = service.import_directory(
            cache_dir=str(tmp_path),
            skip_duplicates=True,
            dry_run=False,
        )

        assert result["status"] == "completed"


class TestImportServiceFormatDisplayName:
    """Tests for ImportService._format_display_name method."""

    def test_format_display_name_simple(self, db_session):
        """Test formatting simple name."""
        service = ImportService(db_session)
        result = service._format_display_name("test_metric")

        assert result == "Test Metric"

    def test_format_display_name_ttft(self, db_session):
        """Test formatting TTFT."""
        service = ImportService(db_session)
        result = service._format_display_name("seconds_to_first_token")

        assert result == "Seconds To First Token"

    def test_format_display_name_tps(self, db_session):
        """Test formatting TPS."""
        service = ImportService(db_session)
        result = service._format_display_name("token_generation_tokens_per_second")

        assert "Token Generation" in result
