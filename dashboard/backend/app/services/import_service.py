"""
Import service for migrating data from YAML files to the database.

Handles:
- Scanning cache directories for YAML files
- Parsing and validating YAML structure
- Deduplication based on build_name
- Transforming and loading data into database
"""

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import yaml

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from app.models import Model, Run, Metric, User
from app.schemas import ModelCreate, RunCreate, MetricCreate


# Import the Metric model for converting MetricCreate to model instances
def _create_metric_from_schema(metric_data: MetricCreate) -> Metric:
    """Convert a MetricCreate schema to a Metric model instance."""
    return Metric(**metric_data.model_dump())


class ImportService:
    """Service class for YAML import operations."""

    # Mapping of YAML keys to metric categories
    PERFORMANCE_METRICS = {
        "seconds_to_first_token": "seconds",
        "std_dev_seconds_to_first_token": "seconds",
        "prefill_tokens_per_second": "tokens/s",
        "token_generation_tokens_per_second": "tokens/s",
        "std_dev_tokens_per_second": "tokens/s",
        "max_memory_used_gbyte": "GB",
        "max_memory_used_GB": "GB",
    }

    ACCURACY_METRICS_PREFIXES = {
        "mmlu_": "%",
        "humaneval_": "%",
        "lm_eval_": "%",
        "perplexity": "",
    }

    def __init__(self, db: Session):
        self.db = db
        self.errors: list[str] = []
        self.imported_count = 0
        self.skipped_count = 0

    def scan_cache_dir(self, cache_dir: str) -> list[dict]:
        """
        Scan a cache directory for YAML stats files.

        Args:
            cache_dir: Path to the cache directory

        Returns:
            List of discovered file info dicts
        """
        discovered = []
        cache_path = Path(cache_dir)

        if not cache_path.exists():
            self.errors.append(f"Cache directory does not exist: {cache_dir}")
            return discovered

        # Look for lemonade_stats.yaml files in build directories
        builds_dir = cache_path / "builds"
        if not builds_dir.exists():
            # Try scanning root level
            builds_dir = cache_path

        for stats_file in builds_dir.rglob("lemonade_stats.yaml"):
            discovered.append({
                "path": str(stats_file),
                "build_name": stats_file.parent.name,
                "size": stats_file.stat().st_size,
            })

        return discovered

    def parse_yaml_file(self, file_path: str) -> dict | None:
        """
        Parse a YAML file and return its contents.

        Args:
            file_path: Path to the YAML file

        Returns:
            Parsed YAML data or None on error
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return data or {}
        except Exception as e:
            self.errors.append(f"Failed to parse {file_path}: {str(e)}")
            return None

    def check_duplicate(self, build_name: str, cache_dir: str) -> bool:
        """
        Check if a run with this build_name already exists.

        Args:
            build_name: Build name to check
            cache_dir: Cache directory (for additional deduplication)

        Returns:
            True if duplicate exists
        """
        query = select(Run).where(Run.build_name == build_name)
        existing = self.db.execute(query).scalar_one_or_none()
        return existing is not None

    def extract_model_info(self, yaml_data: dict) -> dict:
        """
        Extract model information from YAML data.

        Args:
            yaml_data: Parsed YAML data

        Returns:
            Model data dict
        """
        checkpoint = yaml_data.get("checkpoint", "unknown")

        # Extract model name from checkpoint
        if "/" in checkpoint:
            name = checkpoint.split("/")[-1]
        else:
            name = checkpoint

        # Determine model type
        model_type = "llm"
        if any(kw in checkpoint.lower() for kw in ["vision", "vlm", "clip"]):
            model_type = "vlm"
        elif any(kw in checkpoint.lower() for kw in ["embedding", "embed"]):
            model_type = "embedding"

        # Extract family from checkpoint
        family = None
        if "llama" in checkpoint.lower():
            family = "Llama"
        elif "qwen" in checkpoint.lower():
            family = "Qwen"
        elif "phi" in checkpoint.lower():
            family = "Phi"
        elif "mistral" in checkpoint.lower():
            family = "Mistral"
        elif "gemma" in checkpoint.lower():
            family = "Gemma"

        return {
            "name": name,
            "checkpoint": checkpoint,
            "model_type": model_type,
            "family": family,
            "model_metadata": {
                k: v for k, v in yaml_data.items()
                if k not in self.PERFORMANCE_METRICS
                and not any(k.startswith(p) for p in self.ACCURACY_METRICS_PREFIXES)
            },
        }

    def extract_run_info(
        self, yaml_data: dict, build_name: str, file_path: str
    ) -> dict:
        """
        Extract run information from YAML data.

        Args:
            yaml_data: Parsed YAML data
            build_name: Build name
            file_path: Path to the YAML file

        Returns:
            Run data dict
        """
        # Determine run type based on metrics present
        run_type = "benchmark"
        if any(k.startswith("mmlu_") for k in yaml_data.keys()):
            run_type = "accuracy-mmlu"
        elif any(k.startswith("humaneval_") for k in yaml_data.keys()):
            run_type = "accuracy-humaneval"
        elif any(k.startswith("lm_eval_") for k in yaml_data.keys()):
            run_type = "lm-eval"
        elif "perplexity" in yaml_data:
            run_type = "perplexity"

        # Extract configuration
        config = {
            "iterations": yaml_data.get("iterations"),
            "prompts": yaml_data.get("prompts"),
            "output_tokens": yaml_data.get("output_tokens", yaml_data.get("response_tokens")),
            "prompt_tokens": yaml_data.get("prompt_tokens"),
            "warmup_iterations": yaml_data.get("warmup_iterations"),
        }

        # Remove None values
        config = {k: v for k, v in config.items() if v is not None}

        return {
            "build_name": build_name,
            "run_type": run_type,
            "status": "completed",
            "device": yaml_data.get("device"),
            "backend": yaml_data.get("backend"),
            "dtype": yaml_data.get("dtype"),
            "config": config,
            "lemonade_version": yaml_data.get("lemonade_version"),
        }

    def extract_metrics(self, yaml_data: dict, run_id: str) -> list[MetricCreate]:
        """
        Extract metrics from YAML data.

        Args:
            yaml_data: Parsed YAML data
            run_id: Associated run ID

        Returns:
            List of MetricCreate objects
        """
        metrics = []

        # Performance metrics
        for metric_name, unit in self.PERFORMANCE_METRICS.items():
            if metric_name in yaml_data:
                value = yaml_data[metric_name]
                if value is not None:
                    metrics.append(MetricCreate(
                        run_id=run_id,
                        category="performance",
                        name=metric_name,
                        display_name=self._format_display_name(metric_name),
                        value_numeric=float(value),
                        unit=unit,
                    ))

        # Accuracy metrics (MMLU, HumanEval, etc.)
        for key, value in yaml_data.items():
            for prefix, unit in self.ACCURACY_METRICS_PREFIXES.items():
                if key.startswith(prefix):
                    if isinstance(value, (int, float)):
                        metrics.append(MetricCreate(
                            run_id=run_id,
                            category="accuracy",
                            name=key,
                            display_name=self._format_display_name(key),
                            value_numeric=float(value),
                            unit=unit or "%",
                        ))
                    break

        return metrics

    def _format_display_name(self, metric_name: str) -> str:
        """Convert metric name to human-readable display name."""
        # Convert snake_case to Title Case
        return metric_name.replace("_", " ").title()

    def import_file(
        self,
        file_path: str,
        build_name: str,
        skip_duplicates: bool = True,
        cache_dir: str = "",
    ) -> tuple[bool, str]:
        """
        Import a single YAML file to the database.

        Args:
            file_path: Path to the YAML file
            build_name: Build name
            skip_duplicates: Whether to skip existing runs
            cache_dir: Cache directory for deduplication

        Returns:
            Tuple of (success, message)
        """
        # Parse YAML
        yaml_data = self.parse_yaml_file(file_path)
        if yaml_data is None:
            return False, f"Failed to parse YAML file: {file_path}"

        # Check for duplicates
        if skip_duplicates and self.check_duplicate(build_name, cache_dir):
            self.skipped_count += 1
            return True, f"Skipped duplicate: {build_name}"

        try:
            # Extract model info
            model_data = self.extract_model_info(yaml_data)

            # Find or create model
            model = self.db.execute(
                select(Model).where(Model.checkpoint == model_data["checkpoint"])
            ).scalar_one_or_none()

            if not model:
                model_create = ModelCreate(**model_data)
                model = Model(**model_create.model_dump())
                self.db.add(model)
                try:
                    self.db.commit()
                    self.db.refresh(model)
                except Exception as commit_error:
                    self.db.rollback()
                    # Model might already exist due to race condition
                    model = self.db.execute(
                        select(Model).where(Model.checkpoint == model_data["checkpoint"])
                    ).scalar_one_or_none()
                    if not model:
                        raise commit_error
            else:
                # Update model metadata if needed
                if model_data.get("metadata"):
                    model.metadata.update(model_data["metadata"])
                    try:
                        self.db.commit()
                    except Exception as update_error:
                        self.db.rollback()
                        raise update_error

            # Create run
            run_data = self.extract_run_info(yaml_data, build_name, file_path)
            run_data["model_id"] = model.id

            run = Run(**run_data)
            self.db.add(run)
            try:
                self.db.commit()
                self.db.refresh(run)
            except Exception as run_error:
                self.db.rollback()
                raise run_error

            # Create metrics
            metric_creates = self.extract_metrics(yaml_data, run.id)
            if metric_creates:
                metrics = [_create_metric_from_schema(m) for m in metric_creates]
                self.db.add_all(metrics)
                try:
                    self.db.commit()
                except Exception as metrics_error:
                    self.db.rollback()
                    raise metrics_error

            self.imported_count += 1
            return True, f"Imported: {build_name}"

        except Exception as e:
            self.db.rollback()
            self.errors.append(f"Error importing {build_name}: {str(e)}")
            return False, f"Error: {str(e)}"

    def import_directory(
        self,
        cache_dir: str,
        skip_duplicates: bool = True,
        dry_run: bool = False,
    ) -> dict:
        """
        Import all YAML files from a cache directory.

        Args:
            cache_dir: Path to the cache directory
            skip_duplicates: Whether to skip existing runs
            dry_run: If True, only scan without importing

        Returns:
            Import result summary
        """
        start_time = datetime.now(timezone.utc)

        # Scan for files
        discovered = self.scan_cache_dir(cache_dir)

        if dry_run:
            return {
                "status": "dry_run",
                "total_files": len(discovered),
                "discovered_files": discovered,
                "imported_runs": 0,
                "skipped_duplicates": 0,
                "errors": self.errors,
            }

        # Import each file
        results = []
        for file_info in discovered:
            success, message = self.import_file(
                file_path=file_info["path"],
                build_name=file_info["build_name"],
                skip_duplicates=skip_duplicates,
                cache_dir=cache_dir,
            )
            results.append({
                "build_name": file_info["build_name"],
                "success": success,
                "message": message,
            })

        end_time = datetime.now(timezone.utc)

        return {
            "status": "completed",
            "total_files": len(discovered),
            "imported_runs": self.imported_count,
            "skipped_duplicates": self.skipped_count,
            "errors": self.errors,
            "results": results,
            "duration_seconds": (end_time - start_time).total_seconds(),
        }
