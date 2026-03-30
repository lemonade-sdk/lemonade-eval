"""
CLI utilities for the Lemonade Eval Dashboard Backend.

Provides commands for:
- Database initialization
- Data import
- Server management
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import init_db, sync_engine, Base
from app.services.import_service import ImportService
from app.config import settings


def init_database():
    """Initialize the database by creating all tables."""
    print("Initializing database...")
    from app.models import User, Model, Run, Metric, Tag, ModelVersion, RunTag  # noqa: F401
    Base.metadata.create_all(bind=sync_engine)
    print("Database initialized successfully!")


def import_yaml(cache_dir: str, skip_duplicates: bool = True):
    """Import YAML files from a cache directory."""
    from sqlalchemy.orm import Session

    print(f"Importing YAML files from: {cache_dir}")

    with Session(sync_engine) as db:
        service = ImportService(db)
        result = service.import_directory(
            cache_dir=cache_dir,
            skip_duplicates=skip_duplicates,
        )

    print(f"\nImport completed:")
    print(f"  Total files: {result.get('total_files', 0)}")
    print(f"  Imported: {result.get('imported_runs', 0)}")
    print(f"  Skipped: {result.get('skipped_duplicates', 0)}")

    if result.get('errors'):
        print(f"\nErrors:")
        for error in result['errors']:
            print(f"  - {error}")


def main():
    parser = argparse.ArgumentParser(
        description="Lemonade Eval Dashboard CLI utilities"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize the database")
    init_parser.set_defaults(func=lambda _: init_database())

    # Import command
    import_parser = subparsers.add_parser("import", help="Import YAML files")
    import_parser.add_argument(
        "cache_dir",
        help="Path to lemonade cache directory"
    )
    import_parser.add_argument(
        "--no-skip-duplicates",
        action="store_true",
        help="Don't skip duplicate runs"
    )
    import_parser.set_defaults(
        func=lambda args: import_yaml(
            args.cache_dir,
            skip_duplicates=not args.no_skip_duplicates
        )
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
