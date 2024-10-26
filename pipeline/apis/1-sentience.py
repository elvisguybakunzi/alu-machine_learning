#!/usr/bin/env python3
'''Fetches and returns the names of home planets'''
import requests


def sentientPlanets():
    """Fetches and returns the names of home planets
    of all sentient species."""
    species_url = 'https://swapi-api.alx-tools.com/api/species/'
    sentient_planets = set()

    while species_url:
        response = requests.get(species_url)
        if response.status_code != 200:
            print("Failed to retrieve species data")
            return []

        data = response.json()
        species_list = data['results']

        # Process each species in the current page
        for species in species_list:
            # Check if species is sentient
            if (species.get('designation') == 'sentient' or
                    species.get('classification') == 'sentient'):
                homeworld_url = species.get('homeworld')

                # Only proceed if a homeworld is specified
                if homeworld_url:
                    homeworld_response = requests.get(homeworld_url)
                    if homeworld_response.status_code == 200:
                        homeworld_data = homeworld_response.json()
                        planet_name = homeworld_data.get('name', 'unknown')
                        sentient_planets.add(planet_name)
                    else:
                        sentient_planets.add('unknown')

        # Move to the next page if available
        species_url = data.get('next')

    return sorted(sentient_planets)
