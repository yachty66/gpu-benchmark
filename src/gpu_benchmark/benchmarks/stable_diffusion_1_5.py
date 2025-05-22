# src/gpu_benchmark/benchmark.py
import torch
import time
from tqdm import tqdm
import pynvml
from diffusers import StableDiffusionPipeline
import platform
import os

def get_clean_platform():
    os_platform = platform.system()
    if os_platform == "Linux":
        try:
            with open("/etc/os-release") as f:
                for line in f:
                    if line.startswith("PRETTY_NAME="):
                        return line.strip().split("=")[1].strip('"')
        except Exception:
            pass
        return f"Linux {platform.release()}"
    elif os_platform == "Windows":
        return f"Windows {platform.release()}"
    elif os_platform == "Darwin":
        return f"macOS {platform.mac_ver()[0]}"
    else:
        return os_platform

def load_pipeline():
    """Load the Stable Diffusion pipeline and return it."""    
    model_id = "yachty66/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    pipe = pipe.to("cuda")
    return pipe

def get_nvml_device_handle():
    """Get the correct NVML device handle for the GPU being used."""
    pynvml.nvmlInit()
    
    # Check CUDA_VISIBLE_DEVICES first
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_visible_devices is not None:
        try:
            # When CUDA_VISIBLE_DEVICES is set, the first (and only) visible GPU
            # becomes index 0 to CUDA, but we need to use the original index for NVML
            original_gpu_index = int(cuda_visible_devices.split(',')[0])
            handle = pynvml.nvmlDeviceGetHandleByIndex(original_gpu_index)
            return handle
        except (ValueError, IndexError):
            print(f"Warning: Could not parse CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
    
    # Fallback to current CUDA device
    cuda_idx = torch.cuda.current_device()
    return pynvml.nvmlDeviceGetHandleByIndex(cuda_idx)

def run_benchmark(pipe, duration):
    """Run the GPU benchmark for the specified duration in seconds."""
    # Get the correct NVML handle for the GPU being used
    handle = get_nvml_device_handle()
    
    # Setup variables
    image_count = 0
    total_gpu_time = 0
    temp_readings = []
    power_readings = []
    
    # Start benchmark
    start_time = time.time()
    end_time = start_time + duration
    prompt = "a photo of an astronaut riding a horse on mars"
    
    try:
        # Disable progress bar for the pipeline
        pipe.set_progress_bar_config(disable=True)
        
        # Create a progress bar for the entire benchmark
        with tqdm(total=100, desc="Benchmark progress", unit="%") as pbar:
            # Calculate update amount per check
            last_update_time = start_time
            last_update_percent = 0
            
            # Run until time is up
            while time.time() < end_time:
                # Get GPU temperature
                current_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                temp_readings.append(current_temp)
                
                # CUDA timing events
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                
                # Record start time and generate image
                start_event.record()
                image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
                end_event.record()
                torch.cuda.synchronize()
                
                # Calculate timing
                gpu_time_ms = start_event.elapsed_time(end_event)
                total_gpu_time += gpu_time_ms
                
                # Update counter
                image_count += 1
                
                # Sample power usage
                try:
                    current_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                    power_readings.append(current_power)
                except:
                    pass
                
                # Update progress bar
                current_time = time.time()
                current_percent = min(100, int((current_time - start_time) / duration * 100))
                if current_percent > last_update_percent:
                    pbar.update(current_percent - last_update_percent)
                    pbar.set_postfix({
                        'Images': image_count, 
                        'Temp': f"{current_temp}°C"
                    })
                    last_update_percent = current_percent
        
        # Final temperature reading
        final_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        temp_readings.append(final_temp)
        
        # Calculate results
        elapsed = time.time() - start_time
        avg_time_ms = total_gpu_time / image_count if image_count > 0 else 0
        avg_temp = sum(temp_readings) / len(temp_readings)
        max_temp = max(temp_readings)
        
        # Get GPU power info
        try:
            power_usage = round(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0, 2)  # mW to W with 2 decimal places
        except:
            power_usage = None
        
        # Get GPU memory info
        try:
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory_total = round(meminfo.total / (1024 * 1024 * 1024), 2)  # bytes to GB
        except:
            gpu_memory_total = None
        
        # Get platform info
        platform_info = get_clean_platform()
        
        # Get CUDA version (acceleration)
        cuda_version = f"CUDA {torch.version.cuda}" if torch.cuda.is_available() else "N/A"
        
        # Get torch version
        torch_version = torch.__version__
        
        # Clean up
        pynvml.nvmlShutdown()

        # Calculate average power
        avg_power = round(sum(power_readings) / len(power_readings), 2) if power_readings else None

        # Return benchmark results with completed flag
        return {
            "completed": True,  # Flag indicating the benchmark completed successfully
            "result": image_count,
            "max_temp": max_temp,
            "avg_temp": avg_temp,
            "elapsed_time": elapsed,
            "avg_time_ms": avg_time_ms,
            "gpu_utilization": (total_gpu_time/1000)/elapsed*100,
            "gpu_power_watts": avg_power,
            "gpu_memory_total": gpu_memory_total,
            "platform": platform_info,
            "acceleration": cuda_version,
            "torch_version": torch_version
        }
    
    except KeyboardInterrupt:
        # Clean up and return partial results with completed flag set to False
        pynvml.nvmlShutdown()
        return {
            "completed": False,  # Flag indicating the benchmark was canceled
            "result": image_count,
            "max_temp": max(temp_readings) if temp_readings else 0,
            "avg_temp": sum(temp_readings)/len(temp_readings) if temp_readings else 0
        }
    except Exception as e:
        # Handle any other errors, clean up, and return error info
        pynvml.nvmlShutdown()
        print(f"Error during benchmark: {e}")
        return {
            "completed": False,  # Flag indicating the benchmark failed
            "error": str(e),
            "result": image_count
        }