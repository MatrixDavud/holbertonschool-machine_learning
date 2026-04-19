#!/usr/bin/env python3
"""
Module to fetch available ships from SWAPI based on passenger count
"""
import requests


def availableShips(passengerCount):
    """
    Returns the list of ships that can hold a given number of passengers.

    Args:
        passengerCount (int): Minimum number of passengers the ship must hold

    Returns:
        list: Names of ships that can accommodate the passenger count
    """
    ships = []
    url = "https://swapi-api.hbtn.io/api/starships/"

    while url:
        response = requests.get(url)
        data = response.json()

        for ship in data['results']:
            passengers = ship.get('passengers', '0')

            # Clean passengers value (remove commas, handle 'unknown')
            passengers = passengers.replace(',', '').replace('unknown', '0')

            # Try to convert to integer
            try:
                passenger_capacity = int(passengers)

                # Check if ship can hold required number of passengers
                if passenger_capacity >= passengerCount:
                    ships.append(ship['name'])
            except ValueError:
                # Skip ships with invalid passenger data
                continue

        # Handle pagination
        url = data.get('next')

    return ships
