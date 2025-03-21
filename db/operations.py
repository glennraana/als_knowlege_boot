import os
from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient
from bson.binary import Binary
from typing import List, Dict, Any, Optional
import gridfs
from .models import Contribution, ContentTemplate
import json
import numpy as np
from bson.objectid import ObjectId
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

# Get database
db = client[DB_NAME]
async_db = async_client[DB_NAME]

# Collections
contributions_collection = db["contributions"]
embeddings_collection = db["embeddings"]

# GridFS for storing files
fs = gridfs.GridFS(db)

# Flag to track if Atlas Search is available
HAS_VECTOR_SEARCH = False

def get_file_by_id(file_id):
    """
    Retrieve a file from GridFS by its ID.
    
    Args:
        file_id: The ID of the file to retrieve (string or ObjectId)
        
    Returns:
        The binary content of the file or None if not found
    """
    try:
        # Convert string ID to ObjectId if needed
        if isinstance(file_id, str):
            file_id = ObjectId(file_id)
            
        # Get the file from GridFS
        if fs.exists(file_id):
            grid_out = fs.get(file_id)
            return grid_out.read()
        else:
            print(f"File with ID {file_id} not found in GridFS")
            return None
    except Exception as e:
        print(f"Error retrieving file from GridFS: {e}")
        return None

def save_contribution(contribution: Contribution) -> str:
    """
    Save a contribution to MongoDB
    Returns the ID of the saved document
    """
    contribution_dict = contribution.to_dict()
    
    # Handle file content separately using GridFS
    file_id = None
    if contribution.file_content:
        file_id = fs.put(
            contribution.file_content,
            filename=contribution.file_name,
            content_type=contribution.file_type
        )
        # Store reference to the file instead of the file itself
        contribution_dict["file_id"] = file_id
        del contribution_dict["file_content"]
    
    # Insert contribution
    result = contributions_collection.insert_one(contribution_dict)
    return str(result.inserted_id)

def get_contribution(contribution_id: str) -> Optional[Contribution]:
    """
    Retrieve a contribution by ID
    """
    result = contributions_collection.find_one({"id": contribution_id})
    
    if not result:
        return None
    
    # If there's a file reference, retrieve the file content
    if "file_id" in result:
        file_id = result["file_id"]
        grid_out = fs.get(file_id)
        
        # Add file content back to the result
        result["file_content"] = grid_out.read()
        
        # Remove the file_id field as it's not part of our model
        del result["file_id"]
    
    return Contribution.from_dict(result)

def get_all_contributions() -> List[Contribution]:
    """
    Retrieve all contributions
    """
    results = list(contributions_collection.find())
    contributions = []
    
    for result in results:
        # Handle file retrieval for each document
        if "file_id" in result:
            file_id = result["file_id"]
            try:
                grid_out = fs.get(file_id)
                result["file_content"] = grid_out.read()
            except:
                # If file not found, set content to None
                result["file_content"] = None
            
            # Remove the file_id field
            del result["file_id"]
        
        contributions.append(Contribution.from_dict(result))
    
    return contributions

def get_contributions_by_category(category: str) -> List[Contribution]:
    """
    Retrieve contributions filtered by category
    
    Args:
        category: The category to filter by
        
    Returns:
        List of contributions in the specified category
    """
    results = list(contributions_collection.find({"category": category}))
    contributions = []
    
    for result in results:
        # Handle file retrieval for each document
        if "file_id" in result:
            file_id = result["file_id"]
            try:
                grid_out = fs.get(file_id)
                result["file_content"] = grid_out.read()
            except:
                # If file not found, set content to None
                result["file_content"] = None
            
            # Remove the file_id field
            del result["file_id"]
        
        contributions.append(Contribution.from_dict(result))
    
    return contributions

def get_contributions_by_content_type(content_type: str) -> List[Contribution]:
    """
    Retrieve contributions filtered by content type
    
    Args:
        content_type: The content type to filter by
        
    Returns:
        List of contributions with the specified content type
    """
    results = list(contributions_collection.find({"content_type": content_type}))
    contributions = []
    
    for result in results:
        # Handle file retrieval
        if "file_id" in result:
            file_id = result["file_id"]
            try:
                grid_out = fs.get(file_id)
                result["file_content"] = grid_out.read()
            except:
                result["file_content"] = None
            del result["file_id"]
        
        contributions.append(Contribution.from_dict(result))
    
    return contributions

def search_contributions(query: Dict[str, Any]) -> List[Contribution]:
    """
    Search for contributions based on a MongoDB query
    
    Args:
        query: MongoDB query dict
        
    Returns:
        List of matching contributions
    """
    results = list(contributions_collection.find(query))
    contributions = []
    
    for result in results:
        # Handle file retrieval
        if "file_id" in result:
            file_id = result["file_id"]
            try:
                grid_out = fs.get(file_id)
                result["file_content"] = grid_out.read()
            except:
                result["file_content"] = None
            del result["file_id"]
        
        contributions.append(Contribution.from_dict(result))
    
    return contributions

def get_available_categories() -> List[str]:
    """
    Get list of categories that have at least one contribution
    
    Returns:
        List of category names
    """
    return contributions_collection.distinct("category")

def get_available_content_types() -> List[str]:
    """
    Get list of content types that have at least one contribution
    
    Returns:
        List of content type names
    """
    return contributions_collection.distinct("content_type")

def get_available_tags() -> List[str]:
    """
    Get all tags used across contributions
    
    Returns:
        List of tag names
    """
    # Find all documents with tags array
    documents = contributions_collection.find({"tags": {"$exists": True}})
    all_tags = set()
    
    # Collect all tags
    for doc in documents:
        if doc.get("tags"):
            all_tags.update(doc["tags"])
    
    return sorted(list(all_tags))

def save_content_template(template: ContentTemplate) -> str:
    """
    Save a content template to the database
    
    Args:
        template: The template to save
        
    Returns:
        ID of the saved template
    """
    # Add a separate collection for templates
    templates_collection = db["content_templates"]
    
    # Convert to dictionary
    template_dict = template.dict()
    
    # Check if template with this name already exists
    existing = templates_collection.find_one({"template_name": template.template_name})
    if existing:
        # Update existing template
        templates_collection.update_one(
            {"template_name": template.template_name},
            {"$set": template_dict}
        )
        return str(existing["_id"])
    
    # Insert new template
    result = templates_collection.insert_one(template_dict)
    return str(result.inserted_id)

def get_content_templates() -> List[ContentTemplate]:
    """
    Get all available content templates
    
    Returns:
        List of content templates
    """
    from .models import ContentTemplate
    templates_collection = db["content_templates"]
    
    results = list(templates_collection.find())
    templates = []
    
    for result in results:
        # Remove MongoDB _id field
        if "_id" in result:
            del result["_id"]
        
        templates.append(ContentTemplate(**result))
    
    return templates

def delete_contribution(contribution_id: str) -> bool:
    """
    Delete a contribution by ID
    Returns True if deletion was successful
    """
    result = contributions_collection.find_one({"id": contribution_id})
    
    if not result:
        return False
    
    # Delete associated file if exists
    if "file_id" in result:
        fs.delete(result["file_id"])
    
    # Delete contribution
    contributions_collection.delete_one({"id": contribution_id})
    return True

def save_vector_embedding(text: str, embedding: List[float], metadata: Dict[str, Any]) -> str:
    """
    Save a vector embedding to the embeddings collection
    
    Args:
        text: The text that was embedded
        embedding: The vector embedding
        metadata: Additional metadata for the document
        
    Returns:
        The ID of the inserted document
    """
    # Check existing embedding dimensions in the collection
    expected_dim = 1536  # Default for OpenAI embeddings
    
    try:
        # Get a sample document to check the dimension
        sample_doc = embeddings_collection.find_one({})
        if sample_doc and "embedding" in sample_doc:
            expected_dim = len(sample_doc["embedding"])
            print(f"Using existing embedding dimension from DB: {expected_dim}")
    except Exception as e:
        print(f"Error getting sample embedding, using default: {e}")
    
    # Adjust embedding dimension if needed
    embedding_dim = len(embedding)
    if embedding_dim != expected_dim:
        print(f"Adjusting embedding dimension from {embedding_dim} to {expected_dim}")
        if embedding_dim < expected_dim:
            # Pad with zeros
            embedding = embedding + [0.0] * (expected_dim - embedding_dim)
        else:
            # Truncate
            embedding = embedding[:expected_dim]
    
    document = {
        "text": text,
        "embedding": embedding,
        "metadata": metadata
    }
    
    result = embeddings_collection.insert_one(document)
    return str(result.inserted_id)

def get_vector_embeddings(limit: int = 1000) -> List[Dict[str, Any]]:
    """
    Retrieve vector embeddings
    """
    return list(embeddings_collection.find().limit(limit))

def search_similar_embeddings(query_embedding: List[float], n_results: int = 5):
    """
    Search for similar embeddings in the database using vector search
    
    Args:
        query_embedding: The embedding to search for
        n_results: Number of results to return (default: 5)
        
    Returns:
        List of documents with similar embeddings
    """
    global HAS_VECTOR_SEARCH
    
    # Get the actual embedding dimension
    embedding_dim = len(query_embedding)
    print(f"Using embedding dimension: {embedding_dim}")
    
    # Get expected dimension from database
    expected_dim = 1536  # Default
    try:
        sample_doc = embeddings_collection.find_one({})
        if sample_doc and "embedding" in sample_doc:
            expected_dim = len(sample_doc["embedding"])
            print(f"Database embedding dimension: {expected_dim}")
    except Exception as e:
        print(f"Error getting sample embedding: {e}, using default dimension: {expected_dim}")
    
    # Adjust embedding dimension if needed
    if embedding_dim != expected_dim:
        print(f"Adjusting embedding dimension from {embedding_dim} to {expected_dim}")
        if embedding_dim < expected_dim:
            # Pad with zeros
            query_embedding = query_embedding + [0.0] * (expected_dim - embedding_dim)
        else:
            # Truncate
            query_embedding = query_embedding[:expected_dim]
        print(f"Adjusted embedding dimension: {len(query_embedding)}")
    
    try:
        if HAS_VECTOR_SEARCH:
            try:
                # Using MongoDB Atlas Vector Search
                pipeline = [
                    {
                        "$search": {
                            "index": "vector_index",
                            "knnBeta": {
                                "vector": query_embedding,
                                "path": "embedding",
                                "k": n_results * 10
                            }
                        }
                    },
                    {
                        "$project": {
                            "_id": 1,
                            "metadata": 1,
                            "text": 1,
                            "embedding": 1,
                            "score": { "$meta": "searchScore" }
                        }
                    },
                    { "$limit": n_results }
                ]
                
                print("Using MongoDB Atlas Vector Search")
                results = list(embeddings_collection.aggregate(pipeline))
                
                # Sort by score
                results.sort(key=lambda x: x.get("score", 0), reverse=True)
                
                # If no results, fall back to regular search
                if not results:
                    print("No results from Vector Search, falling back to regular search")
                    HAS_VECTOR_SEARCH = False
                    return search_similar_embeddings(query_embedding, n_results)
                    
                return results
                
            except Exception as e:
                print(f"Error using vector search: {e}, falling back to regular search")
                HAS_VECTOR_SEARCH = False
                # Fall back to regular search
                return search_similar_embeddings(query_embedding, n_results)
        else:
            # Fallback method - return all documents sorted by dot product similarity
            print("Using fallback vector search method")
            
            all_documents = list(embeddings_collection.find({}))
            
            # Calculate dot product for each document
            for doc in all_documents:
                embedding = doc.get("embedding", [])
                if embedding and len(embedding) == len(query_embedding):
                    # Calculate dot product similarity
                    similarity = sum(a * b for a, b in zip(embedding, query_embedding))
                    doc["score"] = similarity
                else:
                    doc["score"] = 0
            
            # Sort by score (highest first)
            all_documents.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            # Return top n results
            return all_documents[:n_results]
            
    except Exception as e:
        print(f"Error in search_similar_embeddings: {e}")
        # Return empty list in case of error
        return []

def create_vector_search_index():
    """
    Guide for creating a vector search index in MongoDB Atlas.
    This function will check if we can use vector search but won't try to create it programmatically.
    The index should be created manually in the MongoDB Atlas dashboard.
    """
    global HAS_VECTOR_SEARCH
    
    try:
        # Get a sample document to determine the embedding dimension
        sample_doc = embeddings_collection.find_one({})
        if sample_doc and "embedding" in sample_doc:
            embedding_dim = len(sample_doc["embedding"])
            print(f"Detected embedding dimension: {embedding_dim}")
        else:
            # Default to 384 if we can't determine
            embedding_dim = 384
            print(f"Using default embedding dimension: {embedding_dim}")

        # Check if index exists by trying a simple vector search
        # This is a test to see if vector search capabilities are available
        # Use a non-zero embedding for testing to avoid cosine similarity issues
        test_embedding = [0.01] * embedding_dim  # Non-zero embedding with correct dimension
        
        # Try a basic vector search query
        test_pipeline = [
            {
                "$search": {
                    "index": "vector_index",
                    "knnBeta": {
                        "vector": test_embedding,
                        "path": "embedding",
                        "k": 10
                    }
                }
            },
            { "$limit": 1 }
        ]
        
        # Try to execute a test query
        try:
            # Just test if we can execute the query
            list(embeddings_collection.aggregate(test_pipeline))
            print("Vector search is working correctly!")
            HAS_VECTOR_SEARCH = True
        except Exception as query_err:
            if "index 'vector_index' does not exist" in str(query_err):
                # Index doesn't exist, show instructions
                print("\n=== VECTOR SEARCH SETUP REQUIRED ===")
                print("Your M10 cluster is ready, but you need to create a vector search index.")
                print("Please follow these steps in the MongoDB Atlas dashboard:")
                print("1. Go to your Atlas cluster")
                print("2. Select 'Search' from the left menu")
                print("3. Click 'Create Search Index'")
                print("4. Choose JSON editor and use this configuration:")
                print("""
                {
                  "mappings": {
                    "dynamic": true,
                    "fields": {
                      "embedding": {
                        "dimensions": %d,
                        "similarity": "cosine",
                        "type": "knnVector"
                      }
                    }
                  }
                }
                """ % embedding_dim)
                print("5. Name the index 'vector_index'")
                print("6. Create the index on the 'embeddings' collection")
                print("7. Click 'Create Search Index'")
                print("=====================================\n")
                HAS_VECTOR_SEARCH = False
            else:
                print(f"Error testing vector search: {query_err}")
                HAS_VECTOR_SEARCH = False
        
    except Exception as e:
        print(f"Error checking vector search capability: {e}")
        print("You'll need to create the index manually in MongoDB Atlas dashboard.")
        HAS_VECTOR_SEARCH = False
        
        # Make sure we at least have a regular index on metadata.source
        try:
            embeddings_collection.create_index("metadata.source")
            print("Created fallback index on metadata.source")
        except Exception as index_err:
            print(f"Error creating fallback index: {index_err}")
            
    return HAS_VECTOR_SEARCH

# Initialize the vector search index
try:
    create_vector_search_index()
except Exception as e:
    print(f"Error initializing vector search: {e}")
    HAS_VECTOR_SEARCH = False
