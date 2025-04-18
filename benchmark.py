from diffusers import StableDiffusionPipeline
import torch
import time
from tqdm import tqdm
import pynvml
import datetime
import os

def run_benchmark(duration=300):
    # Initialize the model
    print("Loading Stable Diffusion model...")
    model_id = "sd-legacy/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    print("Model loaded successfully")
    
    # Initialize NVIDIA Management Library for temperature monitoring
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    
    # Set up timing variables
    benchmark_duration = duration  # seconds
    image_count = 0
    total_gpu_time = 0
    temp_readings = []
    
    # Get current timestamp and create log filename with absolute path
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_dir = os.getcwd()
    log_filename = os.path.join(current_dir, f"benchmark_{timestamp}.txt")
    
    # Print the log file path to verify
    print(f"Will save log to: {log_filename}")
    
    # Prompt for generation
    prompt = "a photo of an astronaut riding a horse on mars"
    
    # Try to create the log file with error handling
    try:
        with open(log_filename, "w") as log:
            log.write(f"GPU Benchmark - {timestamp}\n")
            log.write(f"Device: {torch.cuda.get_device_name(0)}\n")
            log.write(f"Benchmark duration: {benchmark_duration} seconds\n")
            log.write(f"Prompt: {prompt}\n")
            log.write("-" * 50 + "\n\n")
            log.write("DETAILED LOG:\n")
        print(f"Successfully created log file")
    except Exception as e:
        print(f"Error creating log file: {e}")
        # Try creating in home directory as fallback
        log_filename = os.path.expanduser(f"~/benchmark_{timestamp}.txt")
        print(f"Trying alternate location: {log_filename}")
        with open(log_filename, "w") as log:
            log.write(f"GPU Benchmark - {timestamp}\n")
    
    # Start the benchmark
    print(f"Starting benchmark for {benchmark_duration} seconds...")
    start_time = time.time()
    end_time = start_time + benchmark_duration
    
    # Function to safely append to log file
    def append_to_log(message):
        try:
            with open(log_filename, "a") as log:
                log.write(message)
        except Exception as e:
            print(f"Error writing to log: {e}")
    
    # Run until time is up
    with tqdm() as pbar:
        while time.time() < end_time:
            # Get GPU temperature and add to list
            current_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            temp_readings.append(current_temp)
            
            # CUDA timing events
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            # Synchronize before generation
            torch.cuda.synchronize()
            
            # Record start time
            start_event.record()
            
            # Generate image (but don't save it)
            image = pipe(prompt).images[0]
            
            # Record end time
            end_event.record()
            torch.cuda.synchronize()
            
            # Calculate GPU time
            gpu_time_ms = start_event.elapsed_time(end_event)
            total_gpu_time += gpu_time_ms
            
            # Log this iteration
            append_to_log(f"Image {image_count}: Time={time.time()-start_time:.2f}s, Temp={current_temp}°C, GenTime={gpu_time_ms:.2f}ms\n")
            
            # Update counter and progress
            image_count += 1
            pbar.update(1)
            pbar.set_description(f"Generated: {image_count} imgs | Current temp: {current_temp}°C")
    
    # Get final temperature reading
    final_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    temp_readings.append(final_temp)
    
    # Calculate results
    elapsed = time.time() - start_time
    avg_time_ms = total_gpu_time / image_count if image_count > 0 else 0
    avg_temp = sum(temp_readings) / len(temp_readings)
    max_temp = max(temp_readings)
    
    # Create summary
    summary = "\n" + "="*50 + "\n"
    summary += "BENCHMARK SUMMARY:\n"
    summary += f"Benchmark completed in {elapsed:.2f} seconds\n"
    summary += f"Images generated: {image_count}\n"
    summary += f"Images per second: {image_count/elapsed:.2f}\n"
    summary += f"Average GPU time per image: {avg_time_ms:.2f} ms\n"
    summary += f"Total GPU processing time: {total_gpu_time/1000:.2f} seconds\n"
    summary += f"GPU utilization: {(total_gpu_time/1000)/elapsed*100:.1f}%\n"
    summary += f"\nTemperature Statistics:\n"
    summary += f"  Starting temperature: {temp_readings[0]}°C\n"
    summary += f"  Ending temperature: {final_temp}°C\n"
    summary += f"  Average temperature: {avg_temp:.1f}°C\n"
    summary += f"  Maximum temperature: {max_temp}°C\n"
    summary += f"  Temperature increase: {final_temp - temp_readings[0]}°C\n"
    summary += "="*50
    
    # Print summary to console
    print(summary)
    
    # Add summary to log file
    append_to_log(summary)
    
    # Clean up
    pynvml.nvmlShutdown()
    
    print(f"Log saved to {log_filename}")
    
    # Double check if file exists
    if os.path.exists(log_filename):
        print(f"Confirmed: log file exists at {log_filename}")
        print(f"File size: {os.path.getsize(log_filename)} bytes")
    else:
        print(f"Warning: Could not find log file at {log_filename}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='GPU Benchmark using Stable Diffusion')
    parser.add_argument('--duration', type=int, default=300,
                        help='Benchmark duration in seconds (default: 300)')
    
    args = parser.parse_args()
    
    print(f"Starting GPU benchmark for {args.duration} seconds")
    run_benchmark(duration=args.duration)