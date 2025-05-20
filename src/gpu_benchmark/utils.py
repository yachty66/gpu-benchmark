# src/gpu_benchmark/utils.py
import platform
import os
import torch
import pynvml

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
        # On macOS, platform.mac_ver()[0] gives the OS version like "14.4.1"
        # platform.platform() gives more details like "macOS-14.4.1-arm64-arm-64bit"
        # We'll stick to the version number for consistency with other OSes.
        return f"macOS {platform.mac_ver()[0]}"
    else:
        return os_platform

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
        except pynvml.NVMLError as e:
            print(f"NVML error when using CUDA_VISIBLE_DEVICES: {e}. Falling back.")

    # Fallback to current CUDA device if CUDA_VISIBLE_DEVICES is not set or fails
    try:
        if torch.cuda.is_available():
            cuda_idx = torch.cuda.current_device()
            handle = pynvml.nvmlDeviceGetHandleByIndex(cuda_idx)
            return handle
        else:
            print("Error: CUDA is not available. Cannot get NVML device handle.")
            return None
    except pynvml.NVMLError as e:
        print(f"NVML error getting handle by current CUDA device: {e}")
        # Attempt to get the first GPU as a last resort if CUDA thinks one is current
        # but NVML can't get it by that index. This is a bit of a guess.
        try:
            print("Attempting to get handle for GPU index 0 as a last resort.")
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return handle
        except pynvml.NVMLError as e_fallback:
            print(f"NVML error on fallback to index 0: {e_fallback}")
            return None
    except Exception as e:
        print(f"An unexpected error occurred in get_nvml_device_handle: {e}")
        return None

def get_system_info():
    """Gathers system information like platform, CUDA version, and PyTorch version."""
    platform_info = get_clean_platform()
    cuda_version = f"CUDA {torch.version.cuda}" if torch.cuda.is_available() else "N/A"
    torch_version = torch.__version__
    return {
        "platform": platform_info,
        "acceleration": cuda_version,
        "torch_version": torch_version
    }
