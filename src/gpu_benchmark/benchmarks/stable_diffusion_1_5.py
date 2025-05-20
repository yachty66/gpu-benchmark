# src/gpu_benchmark/benchmarks/stable_diffusion.py
import torch
import time
from tqdm import tqdm
import pynvml
from diffusers import StableDiffusionPipeline
import os
from ..utils import get_nvml_device_handle, get_system_info # Adjusted import


def load_sd_pipeline():
    """Load the Stable Diffusion pipeline and return it."""    
    model_id = "yachty66/stable-diffusion-v1-5" 
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    pipe = pipe.to("cuda")
    return pipe



def run_sd_benchmark(pipe, duration):
    """Run the Stable Diffusion benchmark for the specified duration in seconds."""
    if pipe is None:
        return {
            "completed": False,
            "error": "Pipeline not loaded. Cannot run benchmark."
        }

    handle = get_nvml_device_handle()
    if handle is None and torch.cuda.is_available(): # Only critical if CUDA is expected
        print("Warning: NVML handle could not be obtained. Some GPU stats (temp, power) might be unavailable.")
        # Allow benchmark to continue, but stats will be limited
    
    image_count = 0
    total_gpu_time = 0
    temp_readings = []
    power_readings = []
    
    start_time = time.time()
    end_time = start_time + duration
    prompt = "a photo of an astronaut riding a horse on mars"
    
    try:
        pipe.set_progress_bar_config(disable=True)
        
        with tqdm(total=100, desc="SD Benchmark", unit="%") as pbar:
            last_update_percent = 0
            
            while time.time() < end_time:
                current_temp = None
                if handle:
                    try:
                        current_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        temp_readings.append(current_temp)
                    except pynvml.NVMLError as e:
                        if "Uninitialized" in str(e) and not torch.cuda.is_available(): # NVML might not be usable without CUDA
                             pass # Expected if no CUDA
                        else:
                            print(f"NVML warning (temperature): {e}")


                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                # No need to synchronize here, record does it
                
                start_event.record()
                _ = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
                end_event.record()
                torch.cuda.synchronize() # Wait for operations to complete before getting time
                
                gpu_time_ms = start_event.elapsed_time(end_event)
                total_gpu_time += gpu_time_ms
                image_count += 1
                
                if handle:
                    try:
                        current_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                        power_readings.append(current_power)
                    except pynvml.NVMLError as e:
                        if "Uninitialized" in str(e) and not torch.cuda.is_available():
                             pass
                        elif "Not Supported" in str(e): # Some GPUs might not support power reading
                            pass # Silently ignore if not supported.
                        else:
                            print(f"NVML warning (power): {e}")


                current_time_progress = time.time()
                current_percent = min(100, int((current_time_progress - start_time) / duration * 100))
                if current_percent > last_update_percent:
                    pbar.update(current_percent - last_update_percent)
                    postfix_dict = {'Images': image_count}
                    if current_temp is not None:
                         postfix_dict['Temp'] = f"{current_temp}°C"
                    pbar.set_postfix(postfix_dict)
                    last_update_percent = current_percent
        
        if handle:
            try:
                final_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                temp_readings.append(final_temp)
            except pynvml.NVMLError as e:
                if "Uninitialized" in str(e) and not torch.cuda.is_available():
                    pass
                else:
                    print(f"NVML warning (final temp): {e}")


        elapsed = time.time() - start_time
        avg_time_ms = total_gpu_time / image_count if image_count > 0 else 0
        avg_temp = sum(temp_readings) / len(temp_readings) if temp_readings else 0
        max_temp = max(temp_readings) if temp_readings else 0
        
        gpu_power_watts = None
        gpu_memory_total = None

        if handle:
            try: # This was originaly power_usage, but avg_power seems more appropriate if collected over time
                avg_power = round(sum(power_readings) / len(power_readings), 2) if power_readings else None
                gpu_power_watts = avg_power
            except pynvml.NVMLError as e:
                 if "Uninitialized" in str(e) and not torch.cuda.is_available(): pass
                 else: print(f"NVML warning (avg power): {e}")
            except Exception: # Catch division by zero if power_readings is empty
                pass

            try:
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory_total = round(meminfo.total / (1024 * 1024 * 1024), 2)  # bytes to GB
            except pynvml.NVMLError as e:
                if "Uninitialized" in str(e) and not torch.cuda.is_available(): pass
                else: print(f"NVML warning (memory info): {e}")
        
        system_info = get_system_info()

        if handle: pynvml.nvmlShutdown()

        return {
            "completed": True,
            "images_generated": image_count, # Specific to SD
            "items_processed": image_count, # Generic key
            "max_temp": max_temp,
            "avg_temp": avg_temp,
            "elapsed_time": elapsed,
            "avg_time_ms": avg_time_ms, # Time per image
            "avg_item_processing_time_ms": avg_time_ms, # Generic key
            "gpu_utilization": (total_gpu_time/1000)/elapsed*100 if elapsed > 0 else 0,
            "gpu_power_watts": gpu_power_watts,
            "gpu_memory_total": gpu_memory_total,
            **system_info # platform, acceleration, torch_version
        }
    
    except KeyboardInterrupt:
        if handle: pynvml.nvmlShutdown()
        return {
            "completed": False,
            "reason": "canceled",
            "images_generated": image_count,
            "items_processed": image_count,
            "max_temp": max(temp_readings) if temp_readings else 0,
            "avg_temp": sum(temp_readings)/len(temp_readings) if temp_readings else 0
        }
    except Exception as e:
        if handle: pynvml.nvmlShutdown()
        print(f"Error during Stable Diffusion benchmark: {e}")
        import traceback
        traceback.print_exc()
        return {
            "completed": False,
            "error": str(e),
            "items_processed": image_count,
        }