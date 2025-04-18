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

def upload_benchmark_results(image_count, max_temp, avg_temp):
    """Upload benchmark results to Supabase database.
    
    Args:
        image_count: Number of images generated during benchmark
        max_temp: Maximum GPU temperature recorded
        avg_temp: Average GPU temperature recorded
        
    Returns:
        tuple: (success, message)
    """
    # Get country flag
    flag_emoji = get_country_flag()
    
    # Prepare benchmark results
    benchmark_results = {
        "created_at": datetime.datetime.now().isoformat(),
        "gpu_type": torch.cuda.get_device_name(0),
        "number_images_generated": image_count,
        "max_heat": int(max_temp),
        "avg_heat": int(avg_temp),
        "country": flag_emoji
    }
    # Upload to Supabase using REST API
    try:
        # Direct REST API endpoint for the table
        api_url = f"{SUPABASE_URL}/rest/v1/benchmark"
        
        # Make the request with auth headers
        response = requests.post(
            api_url,
            json=benchmark_results,
            headers={
                "Content-Type": "application/json",
                "apikey": SUPABASE_ANON_KEY,
                "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
                "Prefer": "return=minimal"
            }
        )
        
        # Check if successful
        if response.status_code in (200, 201):
            return True
        else:
            error_message = f"Error: {response.text}"
            print(f"❌ {error_message}")
            return False, error_message
            
    except Exception as e:
        error_message = f"Error uploading to Supabase: {e}"
        print(f"❌ {error_message}")
        print("\nTroubleshooting tips:")
        print("1. Check your network connection")
        print("2. Verify Supabase service is running")
        return False, error_message