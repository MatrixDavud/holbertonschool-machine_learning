#!/usr/bin/env python3
"""
Script to display the number of launches per rocket
"""
import requests


if __name__ == '__main__':
    # Get all launches
    launches_url = "https://api.spacexdata.com/v4/launches"
    response = requests.get(launches_url)
    launches = response.json()

    # Dictionary to store rocket_id: count
    rocket_launch_count = {}

    # Count launches per rocket_id
    for launch in launches:
        rocket_id = launch['rocket']
        if rocket_id in rocket_launch_count:
            rocket_launch_count[rocket_id] += 1
        else:
            rocket_launch_count[rocket_id] = 1

    # Dictionary to store rocket_name: count
    rocket_name_count = {}

    # Get rocket names
    for rocket_id, count in rocket_launch_count.items():
        rocket_url = f"https://api.spacexdata.com/v4/rockets/{rocket_id}"
        rocket_response = requests.get(rocket_url)
        rocket_data = rocket_response.json()
        rocket_name = rocket_data['name']
        rocket_name_count[rocket_name] = count

    # Sort by count (descending), then by name (ascending)
    sorted_rockets = sorted(
        rocket_name_count.items(),
        key=lambda x: (-x[1], x[0])
    )

    # Print results
    for rocket_name, count in sorted_rockets:
        print(f"{rocket_name}: {count}")
