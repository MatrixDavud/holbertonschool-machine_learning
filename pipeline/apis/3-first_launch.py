#!/usr/bin/env python3
"""
Script to display the first upcoming SpaceX launch
"""
import requests
from datetime import datetime


if __name__ == '__main__':
    # Get all upcoming launches
    launches_url = "https://api.spacexdata.com/v4/launches/upcoming"
    response = requests.get(launches_url)
    launches = response.json()

    if not launches:
        print("No upcoming launches")
    else:
        # Sort by date_unix to get the first launch
        first_launch = min(launches, key=lambda x: x['date_unix'])

        # Get launch details
        launch_name = first_launch['name']
        launch_date = first_launch['date_local']
        rocket_id = first_launch['rocket']
        launchpad_id = first_launch['launchpad']

        # Get rocket name
        rocket_url = f"https://api.spacexdata.com/v4/rockets/{rocket_id}"
        rocket_response = requests.get(rocket_url)
        rocket_data = rocket_response.json()
        rocket_name = rocket_data['name']

        # Get launchpad details
        launchpad_url = (
            f"https://api.spacexdata.com/v4/launchpads/{launchpad_id}"
        )
        launchpad_response = requests.get(launchpad_url)
        launchpad_data = launchpad_response.json()
        launchpad_name = launchpad_data['name']
        launchpad_locality = launchpad_data['locality']

        # Print formatted output
        print(
            f"{launch_name} ({launch_date}) {rocket_name} - "
            f"{launchpad_name} ({launchpad_locality})"
        )
