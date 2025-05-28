# GPU Benchmark by [United Compute](https://www.unitedcompute.ai)

A simple CLI tool to benchmark your GPU's performance with Stable Diffusion and compare results in our global benchmark results.

![United Compute Logo](https://www.unitedcompute.ai/logo.png)

## Installation

For Nvidia GPUs (CUDA)
```bash
pip install gpu-benchmark
```

For AMD GPUs (ROCm, currently Linux only)
```bash
pip install --extra-index-url https://download.pytorch.org/whl/rocm6.3 gpu-benchmark.[rocm]
```

## Usage

Run the benchmark (takes 5 minutes after the pipeline is loaded):

```bash
gpu-benchmark
```

### Optional Arguments

If you're running on a cloud provider, specify it with the `--provider` flag:

```bash
gpu-benchmark --provider runpod
```

You can specify the model to use for the benchmark with the `--model` flag. By default, the Stable Diffusion 1.5 model is used.
Example for running a different model:

```bash
gpu-benchmark --model qwen3-0-6b
```

For multi-GPU systems, you can select a specific GPU like this:

1. Using the `--gpu` flag:

```bash
gpu-benchmark --gpu 1  # Uses GPU index 1
```

The tool will:

1. Load a Stable Diffusion pipeline
2. Generate images for 5 minutes
3. Count image generations and track GPU temperature
4. Upload results to the [United Compute Benchmark Results](https://www.unitedcompute.ai/gpu-benchmark)

## What it measures

- **Benchmark Score**: Number of iterations or images generated in 5 minutes (model-dependent)
- **GPU Model**: The specific model of your GPU (e.g., NVIDIA GeForce RTX 4090)
- **Max Heat**: Maximum GPU temperature reached (°C)
- **Avg Heat**: Average GPU temperature during the benchmark (°C)
- **Country**: Your location (detected automatically)
- **GPU Power**: Power consumption in watts (W)
- **GPU Memory**: Total GPU memory in gigabytes (GB)
- **Platform**: Operating system information
- **Acceleration**: CUDA/ROCM version
- **PyTorch Version**: PyTorch library version

## Requirements

- CUDA-compatible NVIDIA GPU or ROCm-compatible AMD GPU (Linux Only)
- Python 3.8+

## Links

- [Official Website](https://www.unitedcompute.ai)
- [GPU Benchmark Results](https://www.unitedcompute.ai/gpu-benchmark)
