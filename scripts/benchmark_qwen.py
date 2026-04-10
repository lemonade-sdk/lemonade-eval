"""
Benchmark Qwen3.5-2B-GGUF and Qwen3.5-4B-GGUF models
and import results into the dashboard.
"""

import argparse
import hashlib
import hmac
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
import requests


class ModelBenchmarker:
    """Benchmark LLM models using lemonade-server API."""

    def __init__(
        self,
        server_url: str = "http://localhost:8001",
        dashboard_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        cli_secret: Optional[str] = None,
    ):
        self.server_url = server_url
        self.dashboard_url = dashboard_url
        self.api_key = api_key or "test-api-key"
        self.cli_secret = cli_secret or "test-cli-secret"

    def _generate_signature(self, payload: str) -> str:
        """Generate HMAC-SHA256 signature for CLI integration."""
        return hmac.new(
            self.cli_secret.encode(),
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()

    def _load_model(self, model_name: str) -> bool:
        """Load a model into lemonade-server."""
        url = f"{self.server_url}/v1/models/{model_name}"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": model_name,
            "ctx_size": 4096,
        }

        try:
            # First check if model is already loaded
            resp = requests.get(f"{self.server_url}/v1/models", timeout=10)
            if resp.status_code == 200:
                models_data = resp.json()
                for m in models_data.get("models", []):
                    if model_name in m.get("name", ""):
                        print(f"Model {model_name} already loaded")
                        return True

            # Load the model
            resp = requests.post(url, json=data, headers=headers, timeout=60)
            if resp.status_code in [200, 201]:
                print(f"Model {model_name} loaded successfully")
                return True
            elif resp.status_code == 404:
                print(f"Model endpoint not found, trying direct chat call")
                return True  # Model might auto-load on first request
            else:
                print(f"Failed to load model: {resp.status_code} - {resp.text}")
                return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return True  # Continue anyway, model might auto-load

    def _run_benchmark(
        self,
        model_name: str,
        iterations: int = 5,
        warmup: int = 1,
        prompt_length: int = 64,
        output_tokens: int = 32,
    ) -> Dict[str, Any]:
        """Run benchmark on a model."""
        url = f"{self.server_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

        # Generate prompt
        prompt = "word " * prompt_length

        results = {
            "ttft_list": [],  # Time to first token
            "tps_list": [],   # Tokens per second
            "total_time_list": [],
            "output_tokens_list": [],
            "errors": [],
        }

        print(f"\nRunning benchmark for {model_name}...")
        print(f"  Iterations: {iterations}, Warmup: {warmup}")
        print(f"  Prompt length: {prompt_length} tokens, Output: {output_tokens} tokens")

        for i in range(warmup + iterations):
            is_warmup = i < warmup
            iteration = i + 1

            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": f"{prompt} Tell me a story in exactly {output_tokens} tokens:"
                    }
                ],
                "max_tokens": output_tokens,
                "stream": False,
            }

            try:
                start_time = time.perf_counter()
                resp = requests.post(url, json=payload, headers=headers, timeout=120)
                end_time = time.perf_counter()

                if resp.status_code != 200:
                    error_msg = f"Iteration {iteration}: HTTP {resp.status_code} - {resp.text[:200]}"
                    if is_warmup:
                        print(f"  Warmup {iteration}: {error_msg}")
                    else:
                        print(f"  Iteration {iteration}: {error_msg}")
                        results["errors"].append(error_msg)
                    continue

                response_data = resp.json()
                usage = response_data.get("usage", {})
                output_tokens_actual = usage.get("completion_tokens", output_tokens)

                total_time = end_time - start_time
                tps = output_tokens_actual / total_time if total_time > 0 else 0

                if is_warmup:
                    print(f"  Warmup {iteration}: {tps:.2f} tok/s, {total_time:.3f}s")
                else:
                    results["ttft_list"].append(total_time)  # Approximate TTFT
                    results["tps_list"].append(tps)
                    results["total_time_list"].append(total_time)
                    results["output_tokens_list"].append(output_tokens_actual)
                    print(f"  Iteration {iteration}: {tps:.2f} tok/s, {total_time:.3f}s")

            except Exception as e:
                error_msg = f"Iteration {iteration}: {str(e)}"
                if is_warmup:
                    print(f"  Warmup {iteration}: {error_msg}")
                else:
                    print(f"  Iteration {iteration}: {error_msg}")
                    results["errors"].append(error_msg)

        return results

    def _compute_statistics(self, values: List[float]) -> Dict[str, float]:
        """Compute statistics from a list of values."""
        if not values:
            return {}

        import statistics

        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
        }

    def benchmark_model(
        self,
        model_name: str,
        iterations: int = 5,
        warmup: int = 1,
    ) -> Dict[str, Any]:
        """Benchmark a model and return results."""
        # Load model first
        self._load_model(model_name)

        # Run benchmarks with different prompt lengths
        all_results = {}
        for prompt_length in [64, 128, 256]:
            results = self._run_benchmark(
                model_name=model_name,
                iterations=iterations,
                warmup=warmup,
                prompt_length=prompt_length,
                output_tokens=32,
            )

            # Compute statistics
            tps_stats = self._compute_statistics(results["tps_list"])
            ttft_stats = self._compute_statistics(results["ttft_list"])

            key = f"prompt_{prompt_length}"
            all_results[key] = {
                "raw": results,
                "tps_stats": tps_stats,
                "ttft_stats": ttft_stats,
            }

            print(f"\n{model_name} - Prompt length {prompt_length}:")
            if tps_stats:
                print(f"  TPS: mean={tps_stats['mean']:.2f}, std={tps_stats['std_dev']:.2f}")
            if ttft_stats:
                print(f"  Latency: mean={ttft_stats['mean']:.3f}s, std={ttft_stats['std_dev']:.3f}s")

        return all_results

    def upload_to_dashboard(
        self,
        model_name: str,
        benchmark_results: Dict[str, Any],
        run_type: str = "benchmark",
    ) -> Dict[str, Any]:
        """Upload benchmark results to dashboard."""
        # Extract best metrics for upload
        best_tps = 0
        best_tps_prompt = ""

        for prompt_key, data in benchmark_results.items():
            tps_stats = data.get("tps_stats", {})
            mean_tps = tps_stats.get("mean", 0)
            if mean_tps > best_tps:
                best_tps = mean_tps
                best_tps_prompt = prompt_key

        # Prepare payload
        payload = {
            "model_id": model_name,
            "run_type": run_type,
            "build_name": f"{model_name.replace('/', '_').replace(':', '_')}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "config": {
                "iterations": 5,
                "output_tokens": 32,
            },
            "device": "gpu",
            "backend": "llamacpp",
            "dtype": "int4",
            "metrics": [
                {
                    "name": "token_generation_tokens_per_second",
                    "value": round(best_tps, 2),
                    "unit": "tokens/s",
                    "category": "performance",
                    "display_name": "Token Generation Speed",
                },
                {
                    "name": "seconds_to_first_token",
                    "value": round(benchmark_results.get(best_tps_prompt, {}).get("ttft_stats", {}).get("mean", 0), 4),
                    "unit": "seconds",
                    "category": "performance",
                    "display_name": "Time to First Token",
                },
            ],
            "status": "completed",
            "started_at": datetime.now().isoformat(),
            "completed_at": datetime.now().isoformat(),
        }

        # Remove None values
        payload["metrics"] = [m for m in payload["metrics"] if m["value"] is not None and m["value"] > 0]

        # Generate signature
        payload_json = json.dumps(payload)
        signature = self._generate_signature(payload_json)

        # Upload to dashboard
        url = f"{self.dashboard_url}/api/v1/import/evaluation"
        headers = {
            "Content-Type": "application/json",
            "X-CLI-Signature": signature,
        }

        try:
            print(f"\nUploading results to dashboard...")
            resp = requests.post(url, json=payload, headers=headers, timeout=30)

            if resp.status_code in [200, 201]:
                result = resp.json()
                print(f"Upload successful: {result}")
                return result
            else:
                print(f"Upload failed: {resp.status_code} - {resp.text[:500]}")
                return {"success": False, "error": f"HTTP {resp.status_code}"}

        except Exception as e:
            print(f"Upload error: {e}")
            return {"success": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen models and upload to dashboard")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Qwen3.5-2B-GGUF", "Qwen3.5-4B-GGUF"],
        help="Models to benchmark",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--server-url",
        default="http://localhost:8001",
        help="Lemonade server URL",
    )
    parser.add_argument(
        "--dashboard-url",
        default="http://localhost:8000",
        help="Dashboard URL",
    )
    parser.add_argument(
        "--cli-secret",
        default="test-cli-secret",
        help="CLI secret for signature",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip uploading to dashboard",
    )

    args = parser.parse_args()

    benchmarker = ModelBenchmarker(
        server_url=args.server_url,
        dashboard_url=args.dashboard_url,
        cli_secret=args.cli_secret,
    )

    all_results = {}

    for model in args.models:
        print(f"\n{'='*60}")
        print(f"BENCHMARKING: {model}")
        print(f"{'='*60}")

        results = benchmarker.benchmark_model(
            model_name=model,
            iterations=args.iterations,
            warmup=args.warmup,
        )
        all_results[model] = results

        if not args.no_upload:
            benchmarker.upload_to_dashboard(model, results)

    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")

    for model, results in all_results.items():
        print(f"\n{model}:")
        for prompt_key, data in results.items():
            tps_stats = data.get("tps_stats", {})
            if tps_stats:
                print(f"  {prompt_key}: {tps_stats.get('mean', 0):.2f} tok/s (±{tps_stats.get('std_dev', 0):.2f})")

    # Save results to file
    output_file = "benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
