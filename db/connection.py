"""
Database connection module
"""

import os
from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient
import gridfs
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB connection string from environment variables
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb+srv://Cluster80101:VXJYYkR6bFpL@cluster80101.oa4vk.mongodb.net/als_data?retryWrites=true&w=majority")
DB_NAME = os.environ.get("MONGODB_DB_NAME", "als_knowledge")

# Print diagnostics
print(f"Using MongoDB URI: {MONGODB_URI[:30]}...")
print(f"Using database: {DB_NAME}")

# Initialize MongoDB clients
client = MongoClient(MONGODB_URI)
async_client = AsyncIOMotorClient(MONGODB_URI)

def get_database():
    """
    Get the MongoDB database instance
    
    Returns:
        MongoDB database instance
    """
    return client[DB_NAME]

def get_async_database():
    """
    Get the asynchronous MongoDB database instance
    
    Returns:
        Async MongoDB database instance
    """
    return async_client[DB_NAME]

def get_gridfs():
    """
    Get a GridFS instance for the database
    
    Returns:
        GridFS instance
    """
    return gridfs.GridFS(get_database())
