#!/usr/bin/env python3
"""Script that provides stats about Nginx logs stored in MongoDB"""

from pymongo import MongoClient


def print_nginx_stats():
    """
    Connect to MongoDB and print stats about nginx logs
    """
    # Connect to MongoDB
    client = MongoClient('mongodb://127.0.0.1:27017')
    
    # Get the logs database and nginx collection
    collection = client.logs.nginx
    
    # Get total number of logs
    total_logs = collection.count_documents({})
    print("{} logs".format(total_logs))
    
    # Print methods section
    print("Methods:")
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for method in methods:
        count = collection.count_documents({"method": method})
        print("\tmethod {}: {}".format(method, count))
    
    # Get number of status check logs (method=GET, path=/status)
    status_checks = collection.count_documents({
        "method": "GET",
        "path": "/status"
    })
    print("{} status check".format(status_checks))


if __name__ == "__main__":
    print_nginx_stats()