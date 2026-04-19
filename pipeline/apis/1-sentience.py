#!/usr/bin/env python3
"""
Module to fetch home planets of sentient species from SWAPI
"""
import requests


def sentientPlanets():
    """
    Returns the list of names of home planets of all sentient species.

    Sentient species are those with 'sentient' in their classification
    or designation attributes.

    Returns:
        list: Names of home planets of sentient species
    """
    planets = []
    url = "https://swapi-api.hbtn.io/api/species/"

    while url:
        response = requests.get(url)
        data = response.json()

        for species in data['results']:
            # Check if species is sentient
            classification = species.get('classification', '').lower()
            designation = species.get('designation', '').lower()

            if 'sentient' in classification or 'sentient' in designation:
                # Get homeworld
                homeworld_url = species.get('homeworld')

                if homeworld_url:
                    # Fetch planet name
                    planet_response = requests.get(homeworld_url)
                    planet_data = planet_response.json()
                    planet_name = planet_data.get('name')

                    if planet_name and planet_name not in planets:
                        planets.append(planet_name)
                else:
                    # If no homeworld URL, add 'unknown'
                    if 'unknown' not in planets:
                        planets.append('unknown')

        # Handle pagination
        url = data.get('next')

    return planets
