# src/gpu_benchmark/main.py
import argparse
from benchmark import load_pipeline, run_benchmark
from database import upload_benchmark_results
import datetime

def main():
    """Entry point for the GPU benchmark command-line tool."""
    # Simple start message
    print("GPU Benchmark starting...")
    print("This benchmark will run for 10 seconds")
    
    # Fixed duration
    duration = 10  # 10 seconds
    
    print("Loading Stable Diffusion pipeline...")
    pipe = load_pipeline()
    print("Pipeline loaded successfully!")
    
    print("Running benchmark...")
    # Run the benchmark with the loaded pipeline
    results = run_benchmark(pipe=pipe, duration=duration)
    
    # Print the key metrics immediately after benchmark completion
    print("\n" + "="*50)
    print("BENCHMARK RESULTS:")
    print(f"Images Generated: {results['images_generated']}")
    print(f"Max GPU Temperature: {results['max_temp']}°C")
    print(f"Avg GPU Temperature: {results['avg_temp']:.1f}°C")
    print("="*50)
    
    print("\nSubmitting to leaderboard...")
    # Upload results to Supabase
    upload_benchmark_results(
        image_count=results['images_generated'],
        max_temp=results['max_temp'],
        avg_temp=results['avg_temp']
    )
    
    print("Benchmark completed")

if __name__ == "__main__":
    main()