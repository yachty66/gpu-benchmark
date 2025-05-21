# src/gpu_benchmark/database.py
import requests
import datetime
import torch

# Hardcoded Supabase credentials (anon key is designed to be public)
SUPABASE_URL = "https://jftqjabhnesfphpkoilc.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImpmdHFqYWJobmVzZnBocGtvaWxjIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDQ5NzI4NzIsImV4cCI6MjA2MDU0ODg3Mn0.S0ZdRIauUyMhdVJtYFNquvnlW3dV1wxERy7YrurZyag"

def country_code_to_flag(country_code):
    """Convert country code to flag emoji."""
    if len(country_code) != 2 or not country_code.isalpha():
        return "🏳️"  # White flag for unknown
    
    # Convert each letter to regional indicator symbol
    # A-Z: 0x41-0x5A -> regional indicators: 0x1F1E6-0x1F1FF
    return ''.join(chr(ord(c.upper()) - ord('A') + ord('🇦')) for c in country_code)

def get_country_flag():
    """Get country flag emoji based on IP."""
    try:
        country_response = requests.get("https://ipinfo.io/json")
        country_code = country_response.json().get("country", "Unknown")
        return country_code_to_flag(country_code)
    except Exception as e:
        print(f"Error getting country info: {e}")
        return "🏳️"  # White flag for unknown

def upload_benchmark_results(model_name: str, primary_metric_value: int, max_temp: float, avg_temp: float, cloud_provider: str = "Private", **kwargs):
    """Upload benchmark results to Supabase database.
    
    Args:
        model_name: Name of the model ("stable-diffusion", "llm") to determine the target table.
        primary_metric_value: Value for the primary metric (e.g., images generated or tokens processed),
                              which will be stored in the 'result' column.
        max_temp: Maximum GPU temperature recorded.
        avg_temp: Average GPU temperature recorded.
        cloud_provider: Cloud provider name (default: "Private").
        **kwargs: Additional fields to upload (e.g., gpu_power_watts, gpu_memory_total).
        
    Returns:
        tuple: (success, message, record_id)
    """
    
    table_name = ""
    metric_column_name = "result" # Generic column name for the primary metric

    if model_name == "stable-diffusion":
        table_name = "benchmark"
    elif model_name == "llm":
        # For "llm" type, we target "deepseek-r1-1-5b" table as previously specified
        table_name = "deepseek-r1-1-5b" 
    else:
        err_msg = f"Unsupported model_name '{model_name}' for database upload."
        print(f"❌ {err_msg}")
        return False, err_msg, None

    # Get country flag
    flag_emoji = get_country_flag()
    
    # Prepare benchmark results
    benchmark_data = {
        "created_at": datetime.datetime.now().isoformat(),
        "gpu_type": torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "N/A",
        metric_column_name: primary_metric_value, # Using "result" as the column name
        "max_heat": int(max_temp) if max_temp is not None else None,
        "avg_heat": int(avg_temp) if avg_temp is not None else None,
        "country": flag_emoji,
        "provider": cloud_provider
    }
    
    # Add additional fields if provided.
    additional_fields_expected = [
        "gpu_power_watts", "gpu_memory_total", "platform", 
        "acceleration", "torch_version"
    ]
    
    for field in additional_fields_expected:
        if field in kwargs and kwargs[field] is not None:
            benchmark_data[field] = kwargs[field]
    
    # Upload to Supabase using REST API
    try:
        api_url = f"{SUPABASE_URL}/rest/v1/{table_name}" # Dynamic table name
        
        response = requests.post(
            api_url,
            json=benchmark_data,
            headers={
                "Content-Type": "application/json",
                "apikey": SUPABASE_ANON_KEY,
                "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
                "Prefer": "return=representation"  # Changed from minimal to get data back
            }
        )
        
        # Check if successful
        if response.status_code in (200, 201):
            # Parse the response to get the ID
            try:
                record_data = response.json()
                if isinstance(record_data, list) and len(record_data) > 0:
                    record_id = record_data[0].get('id')
                    print(f"✅ Results uploaded successfully to benchmark results!")
                    print(f"Your ID at www.unitedcompute.ai/gpu-benchmark: {record_id}")
                    return True, "Upload successful", record_id
                else:
                    return True, "Upload successful, but couldn't retrieve ID", None
            except Exception as e:
                return True, f"Upload successful, but error parsing response: {e}", None
        else:
            error_message = f"Error: {response.text}"
            print(f"❌ {error_message}")
            return False, error_message, None
            
    except Exception as e:
        error_message = f"Error uploading submitting to benchmark results: {e}"
        print(f"❌ {error_message}")
        print("\nTroubleshooting tips:")
        print("1. Check your network connection")
        return False, error_message, None