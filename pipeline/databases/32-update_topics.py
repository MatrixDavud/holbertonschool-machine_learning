#!/usr/bin/env python3
"""
This module provides a function to update the topics of school documents in a MongoDB collection.
"""

def update_topics(mongo_collection, name, topics):
    """
    Updates the 'topics' field of all documents in the collection where the 'name' matches.

    Args:
        mongo_collection: The pymongo collection object.
        name (str): The name of the school to update.
        topics (list of str): The list of topics to set for the matching school documents.

    Returns:
        The result of the update operation.
    """
    return mongo_collection.update_many(
        { "name": name },
        { "$set": { "topics": topics } }
    )
