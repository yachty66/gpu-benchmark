# GPU Benchmark

A simple CLI tool to benchmark your GPU's performance with Stable Diffusion and compare it on our leaderboard.

![GPU Benchmark Leaderboard](https://path-to-screenshot.png)

## Installation

```bash
pip install gpu-benchmark
```

## Usage

Run the benchmark (takes ˜5 minutes):

```bash
gpu-benchmark
```

The tool will:
1. Load a Stable Diffusion pipeline
2. Generate images for 5 minutes
3. Counts image generations and track GPU temperature
4. Upload results to the leaderboard at [United Compute Benchmark](https://www.unitedcompute.ai/gpu-benchmark)

## What it measures

- **Images Generated**: Number of Stable Diffusion images generated in 10 seconds
- **Max Heat**: Maximum GPU temperature reached (°C)
- **Avg Heat**: Average GPU temperature during the benchmark (°C)
- **Country**: Your location (detected automatically)

## Requirements

- CUDA-compatible NVIDIA GPU
- Python 3.8+
- Internet connection (for results submission - which is not needed, you can run the test without the internet as well, but it will not be displayed online)

## License

Coming soon