# src/gpu_benchmark/benchmarks/llm.py
import torch
import time
from tqdm import tqdm
import pynvml
import os
import platform
from transformers import AutoModelForCausalLM, AutoTokenizer

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

def get_nvml_device_handle():
    """Get the correct NVML device handle for the GPU being used."""
    pynvml.nvmlInit()
    
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_visible_devices is not None:
        try:
            original_gpu_index = int(cuda_visible_devices.split(',')[0])
            handle = pynvml.nvmlDeviceGetHandleByIndex(original_gpu_index)
            return handle
        except (ValueError, IndexError):
            print(f"Warning: Could not parse CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
    
    cuda_idx = torch.cuda.current_device()
    return pynvml.nvmlDeviceGetHandleByIndex(cuda_idx)

def load_llm_model():
    """Load the LLM and tokenizer."""
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    print(f"Loading LLM model: {model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto", # Automatically uses CUDA if available
            trust_remote_code=True
        )
        # Ensure tokenizer has a pad token if it doesn't (some models might not)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            model.config.pad_token_id = model.config.eos_token_id

        print("LLM model and tokenizer loaded successfully!")
        return {"model": model, "tokenizer": tokenizer}
    except Exception as e:
        print(f"Error loading LLM model {model_id}: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def run_llm_benchmark(model_payload, duration):
    """Run the LLM benchmark for the specified duration in seconds."""
    if model_payload.get("error"):
        return {"completed": False, "error": f"LLM model failed to load: {model_payload.get('error')}"}
    
    model = model_payload.get("model")
    tokenizer = model_payload.get("tokenizer")

    if not model or not tokenizer:
         return {"completed": False, "error": "LLM model or tokenizer not provided correctly."}

    handle = get_nvml_device_handle() # This should handle pynvml.nvmlInit()
    if handle is None and torch.cuda.is_available():
        print("Warning: NVML handle could not be obtained for LLM benchmark. GPU stats might be unavailable.")

    tokens_processed = 0
    total_gpu_time_ms = 0 # Total GPU kernel execution time in milliseconds
    generation_steps = 0
    
    temp_readings = []
    power_readings = []

    prompt_text = "Hello, world! This is a benchmark test. " # A simple repetitive prompt
    max_new_tokens_per_step = 50  # Number of new tokens to generate in each inner loop

    start_time_wall = time.time()
    end_time_wall = start_time_wall + duration
    
    print(f"Running LLM benchmark for {duration} seconds...")

    try:
        model.eval() # Set model to evaluation mode

        with tqdm(total=100, desc="Benchmark progress", unit="%") as pbar:
            last_update_percent = 0
            
            while time.time() < end_time_wall:
                current_temp = None
                if handle:
                    try:
                        current_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        temp_readings.append(current_temp)
                    except pynvml.NVMLError as e:
                        if "Uninitialized" not in str(e) or torch.cuda.is_available(): # Log if not expected
                            print(f"NVML warning (temperature): {e}")


                input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)

                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                with torch.no_grad(): # Ensure no gradients are computed
                    generated_outputs = model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens_per_step,
                        do_sample=True, # Using typical generation parameters
                        temperature=0.7,
                        top_k=50,
                        top_p=0.95,
                        pad_token_id=tokenizer.pad_token_id
                    )
                end_event.record()
                torch.cuda.synchronize() # Wait for operations to complete before getting time
                
                gpu_time_this_step_ms = start_event.elapsed_time(end_event)
                total_gpu_time_ms += gpu_time_this_step_ms
                generation_steps += 1
                
                num_generated_this_step = generated_outputs.shape[1] - input_ids.shape[1]
                tokens_processed += num_generated_this_step

                if handle:
                    try:
                        current_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                        power_readings.append(current_power)
                    except pynvml.NVMLError as e:
                        if "Uninitialized" not in str(e) and "Not Supported" not in str(e) or torch.cuda.is_available():
                             print(f"NVML warning (power): {e}")
                
                current_time_progress_wall = time.time()
                current_percent = min(100, int((current_time_progress_wall - start_time_wall) / duration * 100))
                if current_percent > last_update_percent:
                    pbar.update(current_percent - last_update_percent)
                    postfix_dict = {'Tokens': tokens_processed}
                    if current_temp is not None:
                         postfix_dict['Temp'] = f"{current_temp}°C"
                    pbar.set_postfix(postfix_dict)
                    last_update_percent = current_percent
        
        if handle:
            try:
                final_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                temp_readings.append(final_temp)
            except pynvml.NVMLError as e:
                 if "Uninitialized" not in str(e) or torch.cuda.is_available():
                    print(f"NVML warning (final temp): {e}")

        elapsed_wall_time_sec = time.time() - start_time_wall
        
        avg_temp = sum(temp_readings) / len(temp_readings) if temp_readings else 0
        max_temp = max(temp_readings) if temp_readings else 0
        
        avg_tokens_per_second_wall = tokens_processed / elapsed_wall_time_sec if elapsed_wall_time_sec > 0 else 0
        avg_gpu_time_ms_per_step = total_gpu_time_ms / generation_steps if generation_steps > 0 else 0

        gpu_power_watts = None
        gpu_memory_total_gb = None

        if handle:
            try:
                avg_power = round(sum(power_readings) / len(power_readings), 2) if power_readings else None
                gpu_power_watts = avg_power
            except Exception: pass # Catch division by zero if power_readings is empty

            try:
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory_total_gb = round(meminfo.total / (1024**3), 2)  # bytes to GB
            except pynvml.NVMLError as e:
                if "Uninitialized" not in str(e) or torch.cuda.is_available():
                    print(f"NVML warning (memory info): {e}")
        
        platform_info = get_clean_platform()
        acceleration_info = f"CUDA {torch.version.cuda}" if torch.cuda.is_available() else "N/A"
        torch_version_info = torch.__version__

        if handle: pynvml.nvmlShutdown()

        return {
            "completed": True,
            "tokens_processed": tokens_processed,
            "items_processed": tokens_processed, # Generic key
            "elapsed_time_sec": elapsed_wall_time_sec,
            "total_gpu_time_ms": total_gpu_time_ms,
            "generation_steps": generation_steps,
            "avg_tokens_per_second_wall": avg_tokens_per_second_wall,
            "avg_gpu_time_ms_per_step": avg_gpu_time_ms_per_step,
            "max_temp_c": max_temp,
            "avg_temp_c": avg_temp,
            "gpu_power_watts": gpu_power_watts,
            "gpu_memory_total_gb": gpu_memory_total_gb,
            "platform": platform_info,
            "acceleration": acceleration_info,
            "torch_version": torch_version_info,
        }
    
    except KeyboardInterrupt:
        if handle: pynvml.nvmlShutdown()
        elapsed_wall_time_sec = time.time() - start_time_wall
        return {
            "completed": False,
            "reason": "canceled",
            "tokens_processed": tokens_processed,
            "items_processed": tokens_processed,
            "elapsed_time_sec": elapsed_wall_time_sec,
            "max_temp_c": max(temp_readings) if temp_readings else 0,
            "avg_temp_c": sum(temp_readings)/len(temp_readings) if temp_readings else 0
        }
    except Exception as e:
        if handle: pynvml.nvmlShutdown()
        print(f"Error during LLM benchmark: {e}")
        import traceback
        traceback.print_exc()
        return {
            "completed": False,
            "error": str(e),
            "tokens_processed": tokens_processed,
            "items_processed": tokens_processed,
        }
