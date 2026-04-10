"""
Integration module for external services and CLI.

Provides:
- Lemonade-Eval CLI client
- Import pipeline for YAML results
- WebSocket progress streaming
"""

from app.integration.cli_client import CLIClient
from app.integration.import_pipeline import ImportPipeline, EvaluationImporter

__all__ = [
    "CLIClient",
    "ImportPipeline",
    "EvaluationImporter",
]
