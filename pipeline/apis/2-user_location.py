#!/usr/bin/env python3
"""
Script to print the location of a specific GitHub user
"""
import requests
import sys
from datetime import datetime


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: ./2-user_location.py <github_api_url>")
        sys.exit(1)

    url = sys.argv[1]

    try:
        response = requests.get(url)

        # Check if rate limit exceeded (403)
        if response.status_code == 403:
            # Get the reset time from headers
            reset_timestamp = response.headers.get('X-Ratelimit-Reset')

            if reset_timestamp:
                # Convert to integer timestamp
                reset_time = int(reset_timestamp)
                # Get current time
                current_time = int(datetime.now().timestamp())
                # Calculate minutes until reset
                minutes_until_reset = (reset_time - current_time) // 60

                print(f"Reset in {minutes_until_reset} min")
            else:
                print("Reset in 0 min")

        # Check if user not found (404)
        elif response.status_code == 404:
            print("Not found")

        # Success (200)
        elif response.status_code == 200:
            data = response.json()
            location = data.get('location')

            if location:
                print(location)
            else:
                print("Not found")

        else:
            print("Not found")

    except Exception as e:
        print("Not found")
