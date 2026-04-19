#!/usr/bin/env python3
"""
This module provides a function to retrieve all school documents
that include a specific topic in their 'topics' field.
"""

def schools_by_topic(mongo_collection, topic):
    """
    Returns a list of school documents that contain the specified topic.

    Args:
        mongo_collection: The pymongo collection object.
        topic (str): The topic to search for in the 'topics' field.

    Returns:
        A list of matching documents. Returns an empty list if no match is found.
    """
    return list(mongo_collection.find({ "topics": topic }))
