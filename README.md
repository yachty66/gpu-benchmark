# GPU Benchmark by [United Compute](https://www.unitedcompute.ai)

A simple CLI tool to benchmark your GPU's performance with Stable Diffusion and compare results on our global leaderboard.

![United Compute Logo](https://www.unitedcompute.ai/logo.png)

## Installation

```bash
pip install gpu-benchmark
```

## Usage

Run the benchmark (takes 5 minutes after the pipeline is loaded):

```bash
gpu-benchmark
```

The tool will:
1. Load a Stable Diffusion pipeline
2. Generate images for 5 minutes
3. Count image generations and track GPU temperature
4. Upload results to the [United Compute Benchmark Leaderboard](https://www.unitedcompute.ai/gpu-benchmark)

## What it measures

- **Images Generated**: Number of Stable Diffusion images generated in 5 minutes
- **Max Heat**: Maximum GPU temperature reached (°C)
- **Avg Heat**: Average GPU temperature during the benchmark (°C)
- **Country**: Your location (detected automatically)

## Requirements

- CUDA-compatible NVIDIA GPU
- Python 3.8+
- Internet connection (for results submission - although you can run the test offline too)

## Links

- [Official Website](https://www.unitedcompute.ai)
- [GPU Benchmark Leaderboard](https://www.unitedcompute.ai/gpu-benchmark)