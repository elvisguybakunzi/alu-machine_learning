#!/usr/bin/env python3
'''Fetch and print the location of a GitHub user'''
import sys
import requests
import time


def get_user_location(user_url):
    """Fetch and print the location of a GitHub user or handle errors."""
    try:
        # Make the request to the provided user URL
        response = requests.get(user_url)

        # Handle rate limiting (403 status code)
        if response.status_code == 403:
            reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
            # Calculate minutes until rate limit resets
            reset_in_minutes = (reset_time - int(time.time())) // 60
            print("Reset in {} min".format(reset_in_minutes))
            return

        # Handle user not found (404 status code)
        if response.status_code == 404:
            print("Not found")
            return

        # Handle successful request (200 status code)
        if response.status_code == 200:
            user_data = response.json()
            location = user_data.get('location', 'Location not provided')
            print(location)
        else:
            print("Unexpected error occurred")

    except requests.RequestException as e:
        print("Request failed: {}".format(e))
