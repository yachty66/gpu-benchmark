# src/gpu_benchmark/main.py
from .benchmarks import stable_diffusion
from .benchmark import load_pipeline, run_benchmark
from .database import upload_benchmark_results
import argparse
import torch 

# Import benchmark runners dynamically or add specific imports here later
# For now, let's assume functions like run_stable_diffusion_benchmark, run_llm_benchmark
# will be available from src.gpu_benchmark.benchmarks
from .benchmarks import stable_diffusion # This will be created
from .utils import get_clean_platform # This will be created, assuming get_clean_platform moves to utils

def main():
    """Entry point for the GPU benchmark command-line tool."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="GPU Benchmark by United Compute")
    parser.add_argument("--provider", type=str, help="Cloud provider (e.g., RunPod, AWS, GCP) or Private", default="Private")
    parser.add_argument("--gpu", type=int, help="GPU device index to use (defaults to CUDA_VISIBLE_DEVICES or 0)", default=None)
    parser.add_argument("--model", type=str, help="Model to benchmark (e.g., stable-diffusion, llm)", default="stable-diffusion", choices=["stable-diffusion", "llm"])
    args = parser.parse_args()
    
    # If GPU device is specified, set it
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    
    # Convert provider to lowercase
    provider = args.provider.lower()
    
    # Simple start message
    print(f"GPU Benchmark starting for model: {args.model}...")
    print("This benchmark will run for 5 minutes")
    
    # Fixed duration
    duration = 300  # 300 seconds
    
    results = None
    if args.model == "stable-diffusion":
        print("Loading Stable Diffusion pipeline...")
        # This will be moved to stable_diffusion.py
        pipe = stable_diffusion.load_sd_pipeline() 
        print("Pipeline loaded successfully!")
        
        print("Running Stable Diffusion benchmark...")
        results = stable_diffusion.run_sd_benchmark(pipe=pipe, duration=duration)
    elif args.model == "llm":
        print("Loading LLM model...")
        # This will be a new function in llm.py
        model = stable_diffusion.load_llm_model() 
        print("LLM Model loaded successfully!")

        print("Running LLM benchmark...")
        results = stable_diffusion.run_llm_benchmark(model=model, duration=duration)
    else:
        print(f"Error: Model {args.model} not supported.")
        return

    # Only proceed if the benchmark completed successfully (not canceled)
    if results and results.get("completed", False):
        # Print the key metrics immediately after benchmark completion
        print("\n" + "="*50)
        print("BENCHMARK RESULTS:")
        print(f"Images Generated: {results['images_generated']}")
        print(f"Max GPU Temperature: {results['max_temp']}°C")
        print(f"Avg GPU Temperature: {results['avg_temp']:.1f}°C")
        if results.get('gpu_power_watts'):
            print(f"GPU Power: {results['gpu_power_watts']}W")
        if results.get('gpu_memory_total'):
            print(f"GPU Memory: {results['gpu_memory_total']}GB")
        if results.get('platform'):
            print(f"Platform: {results['platform']}")
        if results.get('acceleration'):
            print(f"Acceleration: {results['acceleration']}")
        if results.get('torch_version'):
            print(f"PyTorch Version: {results['torch_version']}")
        print(f"Provider: {provider}")
        print("="*50)
        
        print("\nSubmitting to benchmark results...")
        # Upload results to Supabase with the provider information (lowercase)
        # The upload_benchmark_results function might need to be generalized
        # or each benchmark might return a common set of keys.
        upload_benchmark_results(
            image_count=results.get('images_generated'), # This key might be SD specific
            # Consider making returned keys more generic, e.g., 'items_processed'
            max_temp=results['max_temp'],
            avg_temp=results['avg_temp'],
            gpu_power_watts=results.get('gpu_power_watts'),
            gpu_memory_total=results.get('gpu_memory_total'),
            platform=results.get('platform'), # platform can be fetched by a utility
            acceleration=results.get('acceleration'), # acceleration can be fetched by a utility
            torch_version=results.get('torch_version'), # torch_version can be fetched by a utility
            cloud_provider=provider,  # Use the lowercase provider
            model_name=args.model # Add model name to results
        )
        
        print("Benchmark completed")
    elif results and results.get("error"):
        print(f"\nBenchmark failed: {results.get('error')}")
    elif results is None and args.model != "stable-diffusion" and args.model != "llm": # Model not supported
        pass # Error already printed
    else:
        print("\nBenchmark was canceled or did not complete. Results not submitted.")

if __name__ == "__main__":
    main()