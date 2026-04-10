# System Information and Benchmark Matrix

## Hardware Configuration

| Component | Specification |
|-----------|---------------|
| Platform | Windows 11 (10.0.26200) |
| Processor | AMD Ryzen AI (Family 26, Model 36) |
| CPU Cores | 24 cores |
| RAM | 79.62 GB |
| Python Version | 3.12.11 |

## Software Stack

| Component | Version/Details |
|-----------|-----------------|
| lemonade-server | 10.0.0 |
| Backend | llamacpp (vulkan) |
| Quantization | Q4_K_XL (UD) |
| GPU Backend | Vulkan |

## Model Benchmark Matrix

### Token Generation Speed (tokens/second)

| Model | 64-token prompt | 128-token prompt | 256-token prompt | Best |
|-------|-----------------|------------------|------------------|------|
| Qwen3.5-2B-GGUF | 5.32 ± 0.43 | 5.20 ± 0.19 | 4.43 ± 0.45 | **5.32** |
| Qwen3.5-4B-GGUF | 4.67 ± 1.55 | 3.53 ± 0.54 | 3.82 ± 0.60 | 4.67 |
| **Advantage** | **+12.3%** | **+31.8%** | **+14.0%** | **2B wins** |

### Latency (Time to First Token in seconds)

| Model | 64-token prompt | 128-token prompt | 256-token prompt | Best |
|-------|-----------------|------------------|------------------|------|
| Qwen3.5-2B-GGUF | 5.42 ± 0.56 | 5.90 ± 0.24 | 6.43 ± 0.69 | 5.42s |
| Qwen3.5-4B-GGUF | 4.99 ± 1.32 | 4.31 ± 0.34 | 6.30 ± 0.74 | **4.31s** |
| **Advantage** | 4B faster | **4B faster** | Similar | **4B wins** |

### Performance per Billion Parameters

| Model | Parameters | Best TPS | TPS per Billion |
|-------|------------|----------|-----------------|
| Qwen3.5-2B-GGUF | ~2B | 5.32 | **2.66 tok/s/B** |
| Qwen3.5-4B-GGUF | ~4B | 4.67 | 1.17 tok/s/B |
| **Efficiency** | | | **2B is 2.3x more efficient** |

## Performance Charts

### Token Generation Speed Comparison
```
Tokens/second (higher is better)

Qwen3.5-2B-GGUF  ████████████████████████████████████████  5.32
Qwen3.5-4B-GGUF  ████████████████████████████████          4.67
                 0        2        4        6
```

### Performance by Prompt Length
```
TPS by Prompt Length

64 tokens:
  2B ████████████████████████████████████████  5.32
  4B ████████████████████████████████          4.67

128 tokens:
  2B ██████████████████████████████████████    5.20
  4B ██████████████████████████                3.53

256 tokens:
  2B ████████████████████████████████          4.43
  4B ██████████████████████████████            3.82
```

## Model Characteristics

| Attribute | Qwen3.5-2B-GGUF | Qwen3.5-4B-GGUF |
|-----------|-----------------|-----------------|
| Model Size | 1.34 GB | 2.91 GB |
| Quantization | Q4_K_XL | Q4_K_XL |
| Family | Qwen | Qwen |
| Type | LLM (Vision-capable) | LLM (Vision-capable) |
| Best Use Case | High-throughput | Better reasoning |

## Database Records

| Table | Records Created | IDs |
|-------|-----------------|-----|
| models | 2 | 3d32510e..., cc182c7f... |
| runs | 2 | Benchmark runs |
| metrics | 12 | 6 per model |

## Benchmark Methodology

### Test Parameters
| Parameter | Value |
|-----------|-------|
| Iterations | 5 |
| Warmup Runs | 1 |
| Output Tokens | 32 |
| Prompt Lengths | 64, 128, 256 tokens |
| Backend | llamacpp (GPU/Vulkan) |

### Metrics Collected
- **TPS (Tokens Per Second):** Primary throughput metric
- **TTFT (Time To First Token):** Latency metric
- **Standard Deviation:** Consistency measure
- **Min/Max Values:** Performance bounds

## Files Created

| File | Purpose |
|------|---------|
| `scripts/benchmark_qwen.py` | Benchmark execution script |
| `scripts/import_benchmarks.py` | Dashboard API import |
| `scripts/import_benchmarks_direct.py` | Direct DB import |
| `benchmark_results.json` | Raw benchmark data |
| `BENCHMARK_COMPARISON_REPORT.md` | Detailed report |
| `SYSTEM_INFO_AND_MATRIX.md` | This document |

## Conclusions

1. **Speed Winner:** Qwen3.5-2B-GGUF delivers 12% better throughput
2. **Latency Winner:** Qwen3.5-4B-GGUF has marginally better latency at medium prompts
3. **Efficiency Winner:** Qwen3.5-2B-GGUF is 2.3x more parameter-efficient
4. **Recommendation:** Use 2B for high-throughput, 4B for complex reasoning tasks

---

*Generated: April 7, 2026*
*Lemonade Eval Dashboard Benchmarking System*
