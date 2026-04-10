"""
Pipeline for importing YAML evaluation results.

Features:
- Parse YAML from lemonade-eval cache
- Transform data to database schema
- Handle duplicates and conflicts
- Track import progress
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from app.models import Model, Run, Metric
from app.schemas import ModelCreate, RunCreate, MetricCreate

logger = logging.getLogger(__name__)


class ImportPipeline:
    """
    Pipeline for importing evaluation results from YAML files.

    Supports:
    - Single file import
    - Batch import from directory
    - Progress tracking
    - Error handling and recovery
    """

    # Known performance metrics
    PERFORMANCE_METRICS = {
        "seconds_to_first_token": "seconds",
        "std_dev_seconds_to_first_token": "seconds",
        "prefill_tokens_per_second": "tokens/s",
        "token_generation_tokens_per_second": "tokens/s",
        "std_dev_tokens_per_second": "tokens/s",
        "max_memory_used_gbyte": "GB",
        "max_memory_used_GB": "GB",
    }

    # Accuracy metric prefixes
    ACCURACY_PREFIXES = {
        "mmlu_": "%",
        "humaneval_": "%",
        "lm_eval_": "%",
        "perplexity": "",
    }

    def __init__(self, db_session_factory=None):
        """
        Initialize import pipeline.

        Args:
            db_session_factory: Database session factory
        """
        self.db_factory = db_session_factory
        self.errors: List[str] = []
        self.imported_count = 0
        self.skipped_count = 0
        self.failed_count = 0

    def discover_yaml_files(self, cache_dir: str) -> List[Dict[str, Any]]:
        """
        Discover YAML files in cache directory.

        Args:
            cache_dir: Path to cache directory

        Returns:
            List of file info dicts
        """
        discovered = []
        cache_path = Path(cache_dir)

        if not cache_path.exists():
            self.errors.append(f"Cache directory does not exist: {cache_dir}")
            return discovered

        # Look for lemonade_stats.yaml files
        for stats_file in cache_path.rglob("lemonade_stats.yaml"):
            discovered.append({
                "path": str(stats_file),
                "build_name": stats_file.parent.name,
                "size": stats_file.stat().st_size,
                "modified": datetime.fromtimestamp(
                    stats_file.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
            })

        logger.info(f"Discovered {len(discovered)} YAML files in {cache_dir}")
        return discovered

    def parse_yaml_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Parse a YAML file.

        Args:
            file_path: Path to YAML file

        Returns:
            Parsed data or None on error
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return data or {}
        except Exception as e:
            error_msg = f"Failed to parse {file_path}: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg)
            return None

    def extract_model_info(self, yaml_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract model information from YAML data.

        Args:
            yaml_data: Parsed YAML data

        Returns:
            Model data dict
        """
        checkpoint = yaml_data.get("checkpoint", "unknown")

        # Extract model name from checkpoint
        name = checkpoint.split("/")[-1] if "/" in checkpoint else checkpoint

        # Determine model type
        model_type = "llm"
        checkpoint_lower = checkpoint.lower()
        if any(kw in checkpoint_lower for kw in ["vision", "vlm", "clip"]):
            model_type = "vlm"
        elif any(kw in checkpoint_lower for kw in ["embedding", "embed"]):
            model_type = "embedding"

        # Determine family
        family = None
        family_keywords = {
            "Llama": "llama",
            "Qwen": "qwen",
            "Phi": "phi",
            "Mistral": "mistral",
            "Gemma": "gemma",
        }
        for fam, keyword in family_keywords.items():
            if keyword in checkpoint_lower:
                family = fam
                break

        # Extract metadata (exclude known metric keys)
        exclude_keys = set(self.PERFORMANCE_METRICS.keys())
        for prefix in self.ACCURACY_PREFIXES.keys():
            exclude_keys.update(
                k for k in yaml_data.keys() if k.startswith(prefix)
            )

        metadata = {
            k: v for k, v in yaml_data.items()
            if k not in exclude_keys and k not in ["checkpoint", "build_name"]
        }

        return {
            "name": name,
            "checkpoint": checkpoint,
            "model_type": model_type,
            "family": family,
            "metadata": metadata,
        }

    def extract_run_info(
        self,
        yaml_data: Dict[str, Any],
        build_name: str,
    ) -> Dict[str, Any]:
        """
        Extract run information from YAML data.

        Args:
            yaml_data: Parsed YAML data
            build_name: Build name

        Returns:
            Run data dict
        """
        # Determine run type
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
            "output_tokens": yaml_data.get(
                "output_tokens", yaml_data.get("response_tokens")
            ),
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

    def extract_metrics(
        self,
        yaml_data: Dict[str, Any],
        run_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Extract metrics from YAML data.

        Args:
            yaml_data: Parsed YAML data
            run_id: Associated run ID

        Returns:
            List of metric data dicts
        """
        metrics = []

        # Performance metrics
        for metric_name, unit in self.PERFORMANCE_METRICS.items():
            if metric_name in yaml_data:
                value = yaml_data[metric_name]
                if value is not None:
                    metrics.append({
                        "run_id": run_id,
                        "category": "performance",
                        "name": metric_name,
                        "display_name": self._format_display_name(metric_name),
                        "value_numeric": float(value),
                        "unit": unit,
                    })

        # Accuracy metrics
        for key, value in yaml_data.items():
            for prefix, unit in self.ACCURACY_PREFIXES.items():
                if key.startswith(prefix):
                    if isinstance(value, (int, float)):
                        metrics.append({
                            "run_id": run_id,
                            "category": "accuracy",
                            "name": key,
                            "display_name": self._format_display_name(key),
                            "value_numeric": float(value),
                            "unit": unit or "%",
                        })
                    break

        return metrics

    def _format_display_name(self, name: str) -> str:
        """Convert snake_case to Title Case."""
        return name.replace("_", " ").title()

    async def import_file(
        self,
        file_path: str,
        build_name: str,
        skip_duplicates: bool = True,
    ) -> Tuple[bool, str]:
        """
        Import a single YAML file.

        Args:
            file_path: Path to YAML file
            build_name: Build name
            skip_duplicates: Whether to skip existing runs

        Returns:
            Tuple of (success, message)
        """
        from app.database import get_db
        from sqlalchemy import select

        db = next(get_db()) if self.db_factory is None else self.db_factory()

        try:
            # Parse YAML
            yaml_data = self.parse_yaml_file(file_path)
            if yaml_data is None:
                return False, f"Failed to parse: {file_path}"

            # Check for duplicates
            if skip_duplicates:
                existing = db.execute(
                    select(Run).where(Run.build_name == build_name)
                ).scalar_one_or_none()

                if existing:
                    self.skipped_count += 1
                    return True, f"Skipped duplicate: {build_name}"

            # Extract model info
            model_data = self.extract_model_info(yaml_data)

            # Find or create model
            model = db.execute(
                select(Model).where(Model.checkpoint == model_data["checkpoint"])
            ).scalar_one_or_none()

            if not model:
                model = Model(**model_data)
                db.add(model)
                db.commit()
                db.refresh(model)
            else:
                # Update metadata if changed
                if model_data.get("metadata"):
                    model.model_metadata.update(model_data["metadata"])
                    db.commit()

            # Create run
            run_data = self.extract_run_info(yaml_data, build_name)
            run_data["model_id"] = model.id

            run = Run(**run_data)
            db.add(run)
            db.commit()
            db.refresh(run)

            # Create metrics
            metric_data_list = self.extract_metrics(yaml_data, run.id)
            if metric_data_list:
                metrics = [Metric(**m) for m in metric_data_list]
                db.add_all(metrics)
                db.commit()

            self.imported_count += 1
            return True, f"Imported: {build_name}"

        except Exception as e:
            db.rollback()
            error_msg = f"Error importing {build_name}: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg)
            return False, error_msg

        finally:
            if self.db_factory is None:
                db.close()

    async def import_directory(
        self,
        cache_dir: str,
        skip_duplicates: bool = True,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Import all YAML files from a directory.

        Args:
            cache_dir: Path to cache directory
            skip_duplicates: Whether to skip existing runs
            dry_run: If True, only scan without importing

        Returns:
            Import summary
        """
        start_time = datetime.now(timezone.utc)

        # Discover files
        discovered = self.discover_yaml_files(cache_dir)

        if dry_run:
            return {
                "status": "dry_run",
                "total_files": len(discovered),
                "discovered_files": discovered,
                "imported_runs": 0,
                "skipped_duplicates": 0,
                "errors": [],
            }

        # Import each file
        results = []
        for file_info in discovered:
            success, message = await self.import_file(
                file_path=file_info["path"],
                build_name=file_info["build_name"],
                skip_duplicates=skip_duplicates,
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
            "failed": self.failed_count,
            "errors": self.errors,
            "results": results,
            "duration_seconds": (end_time - start_time).total_seconds(),
        }


class EvaluationImporter:
    """
    High-level evaluation importer combining pipeline with services.

    Provides:
    - Model creation/update
    - Run management
    - Metric storage
    - Progress tracking
    """

    def __init__(self, db_session_factory=None):
        """
        Initialize evaluation importer.

        Args:
            db_session_factory: Database session factory
        """
        self.db_factory = db_session_factory
        self.pipeline = ImportPipeline(db_session_factory)

    async def import_from_cache(
        self,
        cache_dir: str,
        skip_duplicates: bool = True,
    ) -> Dict[str, Any]:
        """
        Import evaluations from cache directory.

        Args:
            cache_dir: Path to cache directory
            skip_duplicates: Whether to skip existing runs

        Returns:
            Import summary
        """
        return await self.pipeline.import_directory(
            cache_dir=cache_dir,
            skip_duplicates=skip_duplicates,
        )

    async def import_single_file(
        self,
        file_path: str,
        skip_duplicates: bool = True,
    ) -> Dict[str, Any]:
        """
        Import a single YAML file.

        Args:
            file_path: Path to YAML file
            skip_duplicates: Whether to skip existing runs

        Returns:
            Import result
        """
        build_name = Path(file_path).parent.name
        success, message = await self.pipeline.import_file(
            file_path=file_path,
            build_name=build_name,
            skip_duplicates=skip_duplicates,
        )

        return {
            "success": success,
            "message": message,
            "build_name": build_name,
        }

    async def import_from_data(
        self,
        yaml_data: Dict[str, Any],
        build_name: str,
        skip_duplicates: bool = True,
    ) -> Dict[str, Any]:
        """
        Import from YAML data dict (not file).

        Args:
            yaml_data: Parsed YAML data
            build_name: Build name
            skip_duplicates: Whether to skip existing runs

        Returns:
            Import result
        """
        from app.database import get_db
        from sqlalchemy import select
        import tempfile

        db = next(get_db()) if self.db_factory is None else self.db_factory()

        try:
            # Check for duplicates
            if skip_duplicates:
                existing = db.execute(
                    select(Run).where(Run.build_name == build_name)
                ).scalar_one_or_none()

                if existing:
                    return {
                        "success": True,
                        "message": f"Skipped duplicate: {build_name}",
                        "skipped": True,
                    }

            # Use pipeline methods
            model_data = self.pipeline.extract_model_info(yaml_data)
            run_data = self.pipeline.extract_run_info(yaml_data, build_name)

            # Find or create model
            model = db.execute(
                select(Model).where(Model.checkpoint == model_data["checkpoint"])
            ).scalar_one_or_none()

            if not model:
                model = Model(**model_data)
                db.add(model)
                db.commit()
                db.refresh(model)

            # Create run
            run_data["model_id"] = model.id
            run = Run(**run_data)
            db.add(run)
            db.commit()
            db.refresh(run)

            # Create metrics
            metric_data_list = self.pipeline.extract_metrics(yaml_data, run.id)
            if metric_data_list:
                metrics = [Metric(**m) for m in metric_data_list]
                db.add_all(metrics)
                db.commit()

            return {
                "success": True,
                "message": f"Imported: {build_name}",
                "run_id": run.id,
            }

        except Exception as e:
            db.rollback()
            return {
                "success": False,
                "message": str(e),
            }

        finally:
            if self.db_factory is None:
                db.close()
