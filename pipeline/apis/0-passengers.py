#!/usr/bin/env python3
'''Returns the list of ships'''
import requests


def availableShips(passengerCount):
    """Returns the list of ships that can hold at least
    the given number of passengers."""
    url = 'https://swapi-api.alx-tools.com/api/starships'
    ships = []

    while url:
        response = requests.get(url)
        if response.status_code != 200:
            return []  # If the request fails, return an empty list

        data = response.json()

        for ship in data['results']:
            # Some starships have 'unknown' or empty passengers field,
            # so we skip those
            try:
                # Remove commas from passenger numbers
                passengers = ship['passengers'].replace(',', '')

                if passengers.isdigit() and int(passengers) >= passengerCount:
                    ships.append(ship['name'])
            except ValueError:
                continue

        # Handle pagination by checking if there's a 'next' page
        url = data['next']

    return ships
