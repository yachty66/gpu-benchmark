# src/gpu_benchmark/main.py
from .benchmarks import stable_diffusion_1_5, llm
from .benchmark import load_pipeline, run_benchmark
from .database import upload_benchmark_results
import argparse
import torch 

# Import benchmark runners dynamically or add specific imports here later
# For now, let's assume functions like run_stable_diffusion_benchmark, run_llm_benchmark
# will be available from src.gpu_benchmark.benchmarks
# from .benchmarks import stable_diffusion # This will be created
# from .utils import get_clean_platform # This will be created, assuming get_clean_platform moves to utils

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
    duration = 10  # 300 seconds
    
    results = None
    if args.model == "stable-diffusion":
        print("Loading Stable Diffusion pipeline...")
        # This will be moved to stable_diffusion.py
        pipe = stable_diffusion_1_5.load_pipeline() 
        print("Pipeline loaded successfully!")
        
        print("Running Stable Diffusion benchmark...")
        results = stable_diffusion_1_5.run_benchmark(pipe=pipe, duration=duration)
    elif args.model == "llm":
        print("Loading LLM model...")
        # This will be a new function in llm.py
        model_payload_dict = llm.load_llm_model() 
        
        print("Running LLM benchmark...")
        results = llm.run_llm_benchmark(model_payload=model_payload_dict, duration=duration)
    else:
        print(f"Error: Model {args.model} not supported.")
        return

    # Only proceed if the benchmark completed successfully (not canceled)
    if results and results.get("completed", False):
        # The detailed print block is removed.
        # Variables are still prepared for the upload_benchmark_results call.
        primary_metric_val = None
        max_temp_val = None
        avg_temp_val = None
        gpu_memory_val = None

        if args.model == "stable-diffusion":
            primary_metric_val = results.get('images_generated')
            max_temp_val = results.get('max_temp')
            avg_temp_val = results.get('avg_temp')
            gpu_memory_val = results.get('gpu_memory_total')
        elif args.model == "llm":
            primary_metric_val = results.get('tokens_processed')
            max_temp_val = results.get('max_temp_c')
            avg_temp_val = results.get('avg_temp_c')
            gpu_memory_val = results.get('gpu_memory_total_gb')
        
        # The upload_benchmark_results function will print the success message and ID.
        upload_benchmark_results(
            model_name=args.model,
            primary_metric_value=primary_metric_val,
            max_temp=max_temp_val,
            avg_temp=avg_temp_val,
            cloud_provider=provider,
            gpu_power_watts=results.get('gpu_power_watts'),
            gpu_memory_total=gpu_memory_val, 
            platform=results.get('platform'),
            acceleration=results.get('acceleration'),
            torch_version=results.get('torch_version')
        )
        
        print("Benchmark completed") # Final confirmation message
    elif results and results.get("error"):
        print(f"\nBenchmark failed: {results.get('error')}")
    elif results is None and args.model != "stable-diffusion" and args.model != "llm": # Model not supported
        pass # Error already printed
    else:
        print("\nBenchmark was canceled or did not complete. Results not submitted.")

if __name__ == "__main__":
    main()