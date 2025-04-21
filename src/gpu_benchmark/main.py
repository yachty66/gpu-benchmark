# src/gpu_benchmark/main.py
from .benchmark import load_pipeline, run_benchmark
from .database import upload_benchmark_results

def main():
    """Entry point for the GPU benchmark command-line tool."""
    # Simple start message
    print("GPU Benchmark starting...")
    print("This benchmark will run for 5 minutes")
    
    # Fixed duration
    duration = 300  # 300 seconds
    
    print("Loading Stable Diffusion pipeline...")
    pipe = load_pipeline()
    print("Pipeline loaded successfully!")
    
    print("Running benchmark...")
    # Run the benchmark with the loaded pipeline
    results = run_benchmark(pipe=pipe, duration=duration)
    
    # Only proceed if the benchmark completed successfully (not canceled)
    if results.get("completed", False):
        # Print the key metrics immediately after benchmark completion
        print("\n" + "="*50)
        print("BENCHMARK RESULTS:")
        print(f"Images Generated: {results['images_generated']}")
        print(f"Max GPU Temperature: {results['max_temp']}°C")
        print(f"Avg GPU Temperature: {results['avg_temp']:.1f}°C")
        print("="*50)
        
        print("\nSubmitting to benchmark results...")
        # Upload results to Supabase
        upload_benchmark_results(
            image_count=results['images_generated'],
            max_temp=results['max_temp'],
            avg_temp=results['avg_temp'],
            gpu_power_watts=results.get('gpu_power_watts'),
            gpu_memory_total=results.get('gpu_memory_total'),
            platform=results.get('platform'),
            acceleration=results.get('acceleration'),
            torch_version=results.get('torch_version')
        )
        
        print("Benchmark completed")
    else:
        print("\nBenchmark was canceled or failed. Results not submitted.")

if __name__ == "__main__":
    main()