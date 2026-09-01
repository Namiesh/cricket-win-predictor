import os
import sys
import requests
from dotenv import load_dotenv, find_dotenv

def test_sportmonks():
    # Load environment variables from .env file
    env_file = find_dotenv()
    if env_file:
        load_dotenv(env_file, override=True)
    else:
        load_dotenv(override=True)
        
    token = os.getenv("SPORTMONKS_API_TOKEN")
    token_detected = bool(token and token.strip())
    
    print("========================================")
    print("SPORTMONKS API CONNECTION TEST")
    print("========================================")
    print(f"Token detected: {'YES' if token_detected else 'NO'}")
    
    if not token_detected:
        print("API response status: N/A")
        print("Connection status: FAILED")
        print("\nSPORTMONKS_API_TOKEN not found or empty in .env file.")
        return

    # Sportmonks Cricket API 2.0 simple endpoint
    url = "https://cricket.sportmonks.com/api/v2.0/continents"
    params = {"api_token": token.strip()}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        status_code = response.status_code
        print(f"API response status: {status_code}")
        
        if status_code == 200:
            print("Connection status: SUCCESS")
            print("\nAPI response received successfully.")
        else:
            print("Connection status: FAILED")
            print("\nHTTP error details:")
            if status_code == 401:
                print("401: Invalid or incorrect API token.")
            elif status_code == 403:
                print("403: API access or plan permission problem.")
            elif status_code == 429:
                print("429: Rate limit exceeded.")
            elif status_code >= 500:
                print(f"{status_code}: Sportmonks server error.")
            else:
                print(f"{status_code}: Unexpected HTTP response status code.")
                
    except requests.exceptions.Timeout:
        print("API response status: TIMEOUT")
        print("Connection status: FAILED")
        print("\nRequest timed out after 10 seconds.")
    except requests.exceptions.RequestException as e:
        print("API response status: ERROR")
        print("Connection status: FAILED")
        print(f"\nNetwork error occurred: {type(e).__name__}")

if __name__ == "__main__":
    test_sportmonks()
