# Install the Supabase client if needed
# !pip install supabase requests

from supabase import create_client
import datetime
import platform
import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Supabase credentials from environment variables
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

# Convert country code to flag emoji (Unicode regional indicator symbols)
def country_code_to_flag(country_code):
    if len(country_code) != 2 or not country_code.isalpha():
        return "🏳️"  # White flag for unknown
    
    # Convert each letter to regional indicator symbol
    # A-Z: 0x41-0x5A -> regional indicators: 0x1F1E6-0x1F1FF
    return ''.join(chr(ord(c.upper()) - ord('A') + ord('🇦')) for c in country_code)

# Get country information
try:
    country_response = requests.get("https://ipinfo.io/json")
    country_code = country_response.json().get("country", "Unknown")
    
    # Convert country code to flag emoji
    flag_emoji = country_code_to_flag(country_code)
    
except Exception as e:
    print(f"Error getting country info: {e}")
    flag_emoji = "🏳️"  # White flag for unknown

# Prepare benchmark results for Supabase using data from the previous run
#todo provide this values from other file
benchmark_results = {
    "created_at": datetime.datetime.now().isoformat(),
    "gpu_type": torch.cuda.get_device_name(0),
    "number_images_generated": image_count,
    "max_heat": int(max_temp),
    "avg_heat": int(avg_temp),
    "country": flag_emoji  # Store the flag emoji directly instead of country code
}

# Print the data being uploaded
print("Uploading the following data to Supabase:")
for key, value in benchmark_results.items():
    print(f"  {key}: {value}")

# Upload to Supabase
try:
    print("\nConnecting to Supabase...")
    # Initialize Supabase client
    supabase = create_client(supabase_url, supabase_key)
    
    print("Uploading results...")
    # Insert benchmark results into the 'benchmark' table
    response = supabase.table('benchmark').insert(benchmark_results).execute()
    
    # Check for successful upload
    if hasattr(response, 'data') and response.data:
        print(f"✅ Results successfully uploaded to Supabase! {flag_emoji}")
        print(f"Inserted record ID: {response.data[0]['id'] if response.data and len(response.data) > 0 else 'unknown'}")
    else:
        print("❌ Error uploading to Supabase.")
        if hasattr(response, 'error'):
            print(f"Error details: {response.error}")
        
except Exception as e:
    print(f"❌ Error uploading to Supabase: {e}")
    print("\nTroubleshooting tips")