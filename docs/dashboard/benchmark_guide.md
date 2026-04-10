# Benchmark Dashboard Usage Guide

**Last Updated:** April 8, 2026

## Overview

The Lemonade Eval Dashboard provides a comprehensive interface for benchmarking and comparing model performance. This guide covers:

1. Running benchmark tests
2. Importing results into the dashboard
3. Viewing and interpreting benchmark comparisons

---

## Prerequisites

- lemonade-server installed and running
- Python 3.12+
- Dashboard backend and frontend running
- Models to benchmark (e.g., Qwen3.5-2B-GGUF, Qwen3.5-4B-GGUF)

---

## 1. Running Benchmarks

### Benchmark Script

Use the provided benchmark script to compare models:

```bash
cd scripts
python benchmark_qwen.py --models Qwen3.5-2B-GGUF Qwen3.5-4B-GGUF
```

### Benchmark Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--models` | Required | Space-separated list of model names to benchmark |
| `--iterations` | 5 | Number of test iterations per prompt length |
| `--warmup` | 1 | Warmup runs before measurement |
| `--output-tokens` | 32 | Number of tokens to generate per run |
| `--prompt-lengths` | 64,128,256 | Prompt lengths to test |

### Example Output

```
Benchmarking Qwen3.5-2B-GGUF...
  Prompt 64 tokens: 5.32 ± 0.43 tok/s, TTFT: 5.42s
  Prompt 128 tokens: 5.20 ± 0.19 tok/s, TTFT: 5.90s
  Prompt 256 tokens: 4.43 ± 0.45 tok/s, TTFT: 6.43s

Benchmarking Qwen3.5-4B-GGUF...
  Prompt 64 tokens: 4.67 ± 1.55 tok/s, TTFT: 4.99s
  Prompt 128 tokens: 3.53 ± 0.54 tok/s, TTFT: 4.31s
  Prompt 256 tokens: 3.82 ± 0.60 tok/s, TTFT: 6.30s
```

Raw results are saved to `benchmark_results.json`.

---

## 2. Importing Results

### Import Script

Import benchmark results into the dashboard database:

```bash
cd scripts
python import_benchmarks_direct.py \
  --results-file ../benchmark_results.json \
  --db-url sqlite:///../dashboard/backend/eval.db
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--results-file` | benchmark_results.json | Path to benchmark JSON file |
| `--db-url` | sqlite:///test.db | Database connection URL |

### What Gets Imported

For each model, the script creates:
- **Model record** (if not exists)
- **Run record** with benchmark configuration
- **6 Metrics**:
  - `token_generation_tokens_per_second` (best TPS)
  - `seconds_to_first_token` (best TTFT)
  - `std_dev_tokens_per_second` (consistency)
  - `tps_prompt_64`, `tps_prompt_128`, `tps_prompt_256` (per-length TPS)

### Import Summary

After import, you'll see output like:
```
IMPORT SUMMARY
==============
Model Comparison (Best TPS):
  Qwen3.5-2B-GGUF: 5.32 tok/s
  Qwen3.5-4B-GGUF: 4.67 tok/s

Winner: Qwen3.5-2B-GGUF with 5.32 tok/s
```

---

## 3. Viewing Results in Dashboard

### Start the Dashboard

```bash
# Terminal 1: Backend
cd dashboard/backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 3001

# Terminal 2: Frontend
cd dashboard/frontend
npm run dev
```

### Navigate to Benchmarks

Open `http://localhost:5173/benchmarks` in your browser.

### Dashboard Components

#### Winner Highlight Card

Shows the top-performing model with:
- Model name
- Best TPS value
- Trophy icon

#### Summary Statistics

| Card | Description |
|------|-------------|
| Total Runs | Number of benchmark runs in database |
| Best TPS | Highest tokens/second achieved |
| Best TTFT | Lowest time-to-first-token |
| Models Compared | Number of unique models benchmarked |

#### TPS Comparison Chart

Bar chart showing token generation speed for all models:
- Higher bars = better performance
- Green highlight on best performer
- Hover for exact values

#### Detailed Results Table

| Column | Description |
|--------|-------------|
| Model | Model name |
| TPS (tok/s) | Token generation speed |
| TTFT (s) | Time to first token |
| Std Dev | Performance consistency |
| Backend | Inference backend (e.g., llamacpp) |
| Device | Hardware (cpu/gpu/npu) |
| Run ID | Unique run identifier |

Rows are sorted by TPS (best first). The winning row has a green background.

#### Model Averages

Shows aggregated statistics when multiple runs exist for the same model:
- Average TPS across runs
- Average TTFT across runs
- Number of runs

---

## Interpreting Results

### Key Metrics

**TPS (Tokens Per Second)**
- Primary throughput metric
- Higher = faster text generation
- Affected by: model size, quantization, hardware

**TTFT (Time To First Token)**
- Latency before generation starts
- Lower = more responsive
- Affected by: prompt length, model loading

**Standard Deviation**
- Performance consistency
- Lower = more predictable
- High values may indicate thermal throttling or memory pressure

### Example Analysis

From the Qwen3.5 comparison:

```
Qwen3.5-2B-GGUF: 5.32 tok/s, TTFT: 5.42s, StdDev: 0.43
Qwen3.5-4B-GGUF: 4.67 tok/s, TTFT: 4.31s, StdDev: 1.55
```

**Conclusions:**
- 2B model is **12% faster** in token generation
- 4B model has **slightly better latency** at medium prompts
- 2B model is **more consistent** (lower std dev)
- 2B model is **2.3x more parameter-efficient**

### Recommendations

| Use Case | Recommended Model |
|----------|-------------------|
| High-throughput applications | Smaller model (better TPS) |
| Low-latency interactions | Compare TTFT values |
| Resource-constrained devices | Consider parameter efficiency |
| Complex reasoning tasks | Larger model (may have better accuracy) |

---

## Troubleshooting

### No Data in Dashboard

1. Verify database has data:
   ```bash
   cd dashboard/backend
   python -c "import sqlite3; conn=sqlite3.connect('eval.db'); print(conn.execute('SELECT COUNT(*) FROM runs').fetchone()[0])"
   ```

2. Check backend is running on correct port

3. Verify frontend API URL matches backend port

### Import Errors

**"no such table: runs"**
- Run database initialization first:
  ```bash
  python -c "from app.database import Base, sync_engine; Base.metadata.create_all(bind=sync_engine)"
  ```

**"unable to open database file"**
- Use absolute path in `--db-url`
- Check file permissions

### Backend Errors

**Database connection refused**
- Ensure DATABASE_URL in `.env` points to correct file
- Check no other process is locking the database

---

## File Locations

| File | Purpose |
|------|---------|
| `scripts/benchmark_qwen.py` | Benchmark execution |
| `scripts/import_benchmarks_direct.py` | Database import |
| `benchmark_results.json` | Raw benchmark data |
| `dashboard/backend/eval.db` | SQLite database |
| `dashboard/frontend/src/pages/benchmarks/BenchmarksPage.tsx` | UI component |

---

## API Reference

### Get Benchmark Results

```bash
curl http://localhost:3001/api/v1/runs/benchmark/results
```

Response format:
```json
{
  "success": true,
  "data": [
    {
      "run": {
        "model_id": "...",
        "build_name": "Qwen3.5-2B-GGUF-20260408_001206",
        "run_type": "benchmark",
        "status": "completed",
        ...
      },
      "metrics": [
        {
          "name": "token_generation_tokens_per_second",
          "value_numeric": 5.32,
          "unit": "tokens/s",
          ...
        }
      ]
    }
  ]
}
```

---

## Related Documentation

- [Dashboard Implementation Summary](../dashboard/IMPLEMENTATION_SUMMARY.md)
- [Power Profiling Guide](../power_profiling.md)
- [MMLU Accuracy Benchmarks](../mmlu_accuracy.md)
- [HumanEval Accuracy Benchmarks](../humaneval_accuracy.md)
