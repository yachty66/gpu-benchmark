# src/gpu_benchmark/main.py
import argparse
from benchmark import load_pipeline, run_benchmark
from database import upload_benchmark_results
import datetime

def main():
    """Entry point for the GPU benchmark command-line tool."""
    # Simple log message at the beginning
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] GPU Benchmark starting...")
    print(f"This benchmark will run for 10 seconds")
    print("-" * 50)
    
    # Fixed duration
    duration = 10  # 10 seconds
    
    print("Loading Stable Diffusion pipeline...")
    pipe = load_pipeline()
    print("Pipeline loaded successfully!")
    
    print("\nRunning benchmark:")
    # Run the benchmark with the loaded pipeline
    results = run_benchmark(pipe=pipe, duration=duration)
    
    print("\nUploading results to Supabase...")
    # Upload results to Supabase
    upload_benchmark_results(
        image_count=results['images_generated'],
        max_temp=results['max_temp'],
        avg_temp=results['avg_temp']
    )
    
    # Simple log message at the end
    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("-" * 50)
    print(f"[{end_time}] GPU Benchmark completed")
    print(f"Images generated: {results['images_generated']}")
    print(f"Max temperature: {results['max_temp']}°C")
    print(f"Avg temperature: {results['avg_temp']}°C")

if __name__ == "__main__":
    main()