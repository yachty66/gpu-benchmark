# src/gpu_benchmark/main.py
import argparse
from benchmark import load_pipeline, run_benchmark
from database import upload_benchmark_results

def main():
    """Entry point for the GPU benchmark command-line tool."""
    # Hardcoded duration
    duration = 10  # 10 seconds fixed duration
    
    print(f"Starting GPU benchmark from United Compute")
    
    # Load the pipeline
    print("Loading Stable Diffusion pipeline...")
    pipe = load_pipeline()
    print("Pipeline loaded successfully!")
    
    # Run the benchmark with the loaded pipeline
    results = run_benchmark(pipe=pipe, duration=duration)
    
    print("\nBenchmark completed!")
    print(f"Images generated: {results['images_generated']}")
    print(f"Maximum GPU temperature: {results['max_temp']}°C")
    print(f"Average GPU temperature: {results['avg_temp']}°C")

    # Upload results to Supabase
    upload_benchmark_results(
        image_count=results['images_generated'],
        max_temp=results['max_temp'],
        avg_temp=results['avg_temp']
    )

if __name__ == "__main__":
    main()