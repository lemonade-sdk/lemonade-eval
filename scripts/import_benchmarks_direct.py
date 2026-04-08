"""
Direct database import for benchmark results.
Creates models, runs, and metrics directly in the database.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "dashboard" / "backend"
sys.path.insert(0, str(backend_path))

# Parse arguments first to set DATABASE_URL before imports
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Import benchmark results into database")
    parser.add_argument(
        "--results-file",
        default="benchmark_results.json",
        help="Path to benchmark results JSON file",
    )
    parser.add_argument(
        "--db-url",
        default="sqlite:///test.db",
        help="Database URL",
    )
    args = parser.parse_args()
    os.environ["DATABASE_URL"] = args.db_url
else:
    args = None

# Set environment variables (TESTING must be false to use file-based DB)
os.environ["DEBUG"] = "true"
os.environ["TESTING"] = "false"
os.environ["TEST_DATABASE_URL"] = ""

from app.database import sync_engine, Base, SyncSessionLocal
from app.models import Model, Run, Metric

# Create tables if they don't exist
Base.metadata.create_all(bind=sync_engine)


def import_benchmark_results(results_file: str = "benchmark_results.json"):
    """Import benchmark results directly into the database."""

    # Load results
    with open(results_file, "r") as f:
        results = json.load(f)

    print(f"Loading results from {results_file}")
    print(f"Models: {list(results.keys())}")

    session = SyncSessionLocal()

    try:
        for model_name, model_results in results.items():
            print(f"\n{'='*60}")
            print(f"Importing: {model_name}")
            print(f"{'='*60}")

            # Find best metrics
            best_tps = 0
            best_tps_prompt = ""
            best_ttft = float('inf')
            best_ttft_prompt = ""
            tps_std_dev = 0

            for prompt_key, data in model_results.items():
                tps_stats = data.get("tps_stats", {})
                ttft_stats = data.get("ttft_stats", {})

                mean_tps = tps_stats.get("mean", 0)
                mean_ttft = ttft_stats.get("mean", 0)

                if mean_tps > best_tps:
                    best_tps = mean_tps
                    best_tps_prompt = prompt_key
                    tps_std_dev = tps_stats.get("std_dev", 0)

                if mean_ttft > 0 and mean_ttft < best_ttft:
                    best_ttft = mean_ttft
                    best_ttft_prompt = prompt_key

            print(f"  Best TPS: {best_tps:.2f} ± {tps_std_dev:.2f} tok/s ({best_tps_prompt})")
            print(f"  Best TTFT: {best_ttft:.4f}s ({best_ttft_prompt})")

            # Create or get model
            model_checkpoint = model_name
            if "/" not in model_name and ":" not in model_name:
                # GGUF model - construct checkpoint path
                model_checkpoint = f"unsloth/{model_name}:{model_name.replace('-GGUF', '').replace('.', '_')}-UD-Q4_K_XL.gguf"

            model = session.query(Model).filter_by(name=model_name).first()
            if not model:
                model = Model(
                    name=model_name,
                    checkpoint=model_checkpoint,
                    model_type="llm",
                    family="qwen",
                )
                session.add(model)
                session.flush()
                print(f"  Created model: {model_name}")
            else:
                print(f"  Model exists: {model_name}")

            # Create run
            build_name = f"{model_name.replace('/', '_').replace(':', '_')}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            run = Run(
                model_id=model.id,
                run_type="benchmark",
                build_name=build_name,
                status="completed",
                config={
                    "iterations": 5,
                    "output_tokens": 32,
                    "prompt_lengths": [64, 128, 256],
                },
                device="gpu",
                backend="llamacpp",
                dtype="int4",
                duration_seconds=best_ttft + (32 / best_tps) if best_tps > 0 else 0,
            )
            session.add(run)
            session.flush()
            print(f"  Created run: {build_name}")

            # Create metrics
            metrics_data = [
                {
                    "name": "token_generation_tokens_per_second",
                    "value": round(best_tps, 2),
                    "category": "performance",
                    "unit": "tokens/s",
                    "display_name": "Token Generation Speed",
                },
                {
                    "name": "seconds_to_first_token",
                    "value": round(best_ttft, 4),
                    "category": "performance",
                    "unit": "seconds",
                    "display_name": "Time to First Token",
                },
                {
                    "name": "std_dev_tokens_per_second",
                    "value": round(tps_std_dev, 2),
                    "category": "performance",
                    "unit": "tokens/s",
                    "display_name": "TPS Standard Deviation",
                },
            ]

            # Add per-prompt-length metrics
            for prompt_key, data in model_results.items():
                tps_stats = data.get("tps_stats", {})
                if tps_stats.get("mean", 0) > 0:
                    metrics_data.append({
                        "name": f"tps_{prompt_key}",
                        "value": round(tps_stats["mean"], 2),
                        "category": "performance",
                        "unit": "tokens/s",
                        "display_name": f"TPS ({prompt_key})",
                    })

            for metric_data in metrics_data:
                metric = Metric(
                    run_id=run.id,
                    name=metric_data["name"],
                    value_numeric=metric_data["value"],
                    category=metric_data["category"],
                    unit=metric_data.get("unit", ""),
                    display_name=metric_data.get("display_name", metric_data["name"]),
                )
                session.add(metric)

            print(f"  Created {len(metrics_data)} metrics")
            session.commit()

        # Print summary
        print(f"\n{'='*60}")
        print("IMPORT SUMMARY")
        print(f"{'='*60}")
        print("\nModel Comparison (Best TPS):")

        # Query back to show results
        runs = session.query(Run).filter_by(run_type="benchmark").all()
        for run in runs:
            tps_metric = session.query(Metric).filter(
                Metric.run_id == run.id,
                Metric.name == "token_generation_tokens_per_second"
            ).first()
            if tps_metric:
                print(f"  {run.model_id}: {tps_metric.value_numeric:.2f} tok/s")

        # Determine winner
        winner = session.query(Run, Metric).join(Metric).filter(
            Run.run_type == "benchmark",
            Metric.name == "token_generation_tokens_per_second"
        ).order_by(Metric.value_numeric.desc()).first()

        if winner:
            run, metric = winner
            print(f"\nWinner: {run.model_id} with {metric.value_numeric:.2f} tok/s")

        print(f"\nResults imported successfully!")

    except Exception as e:
        session.rollback()
        print(f"Error importing results: {e}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    import_benchmark_results(results_file=args.results_file)
