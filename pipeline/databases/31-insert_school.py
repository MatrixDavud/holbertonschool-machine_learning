#!/usr/bin/env python3
"""
This module provides a function to insert a new document into a MongoDB collection using keyword arguments.
"""

def insert_school(mongo_collection, **kwargs):
    """
    Inserts a new document into the specified MongoDB collection.

    Args:
        mongo_collection: The pymongo collection object.
        **kwargs: Arbitrary keyword arguments representing the document fields and values.

    Returns:
        The _id of the newly inserted document.
    """
    return mongo_collection.insert_one(kwargs).inserted_id
