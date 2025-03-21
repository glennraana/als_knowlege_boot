"""
Database connection module
"""

import os
import streamlit as st
from dotenv import load_dotenv

# Forsøk å importere MongoDB-pakker
try:
    from pymongo import MongoClient
    from motor.motor_asyncio import AsyncIOMotorClient
    import gridfs
except ImportError as e:
    st.error(f"Kunne ikke importere MongoDB-pakker: {e}")
    raise

# Load environment variables
load_dotenv()

# MongoDB connection string from environment variables or Streamlit secrets
def get_mongodb_uri():
    # Prøv først Streamlit secrets
    if hasattr(st, 'secrets') and 'MONGODB_URI' in st.secrets:
        return st.secrets['MONGODB_URI']
    # Ellers fra miljøvariabler
    elif 'MONGODB_URI' in os.environ:
        return os.environ['MONGODB_URI']
    else:
        st.error("MONGODB_URI ble ikke funnet i secrets eller miljøvariabler")
        return None

def get_db_name():
    # Prøv først Streamlit secrets
    if hasattr(st, 'secrets') and 'MONGODB_DB_NAME' in st.secrets:
        return st.secrets['MONGODB_DB_NAME']
    # Ellers fra miljøvariabler
    elif 'MONGODB_DB_NAME' in os.environ:
        return os.environ.get('MONGODB_DB_NAME', 'als_knowledge')
    else:
        return 'als_knowledge'

# MongoDB connection string and DB name
MONGODB_URI = get_mongodb_uri()
DB_NAME = get_db_name()

# Diagnostics (masked for sikkerhet)
if MONGODB_URI:
    masked_uri = MONGODB_URI.split('@')[0][:15] + "..." if '@' in MONGODB_URI else "..."
    print(f"Using MongoDB URI: {masked_uri}")
    print(f"Using database: {DB_NAME}")

# Initialize MongoDB clients
client = None
async_client = None

if MONGODB_URI:
    try:
        client = MongoClient(MONGODB_URI)
        async_client = AsyncIOMotorClient(MONGODB_URI)
    except Exception as e:
        st.error(f"Kunne ikke koble til MongoDB: {e}")

def get_database():
    """
    Get the MongoDB database instance
    
    Returns:
        MongoDB database instance
    """
    if not client:
        st.error("MongoDB client er ikke initialisert")
        return None
        
    return client[DB_NAME]

def get_async_database():
    """
    Get the asynchronous MongoDB database instance
    
    Returns:
        Async MongoDB database instance
    """
    if not async_client:
        st.error("Async MongoDB client er ikke initialisert")
        return None
        
    return async_client[DB_NAME]

def get_gridfs():
    """
    Get a GridFS instance for the database
    
    Returns:
        GridFS instance
    """
    if not client:
        st.error("MongoDB client er ikke initialisert")
        return None
        
    return gridfs.GridFS(get_database())
