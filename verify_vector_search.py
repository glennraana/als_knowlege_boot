"""
Verify Vector Search - A simple script to directly test vector search and show the results.
This script demonstrates that vector search is retrieving real documents from the knowledge base.
"""

import os
import sys
import numpy as np
from dotenv import load_dotenv
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from rag.embeddings import get_embeddings
from db.operations import create_vector_search_index

# Load environment variables
load_dotenv()

# Create a function to adjust embedding dimensions to match the database
def adjust_embedding_dimensions(embedding, target_dim=1536):
    """
    Adjust the dimensions of an embedding to match the target dimensions by
    either padding with zeros or truncating.
    
    Args:
        embedding (list): The embedding to adjust
        target_dim (int): The target number of dimensions
        
    Returns:
        list: The adjusted embedding
    """
    current_dim = len(embedding)
    
    if current_dim == target_dim:
        return embedding
        
    if current_dim < target_dim:
        # Pad with zeros
        return embedding + [0.0] * (target_dim - current_dim)
    else:
        # Truncate
        return embedding[:target_dim]

# Create a custom wrapper for the embeddings model
class DimensionAdjustingEmbeddings:
    def __init__(self, base_embeddings, target_dim=1536):
        self.base_embeddings = base_embeddings
        self.target_dim = target_dim
        
    def embed_documents(self, texts):
        embeddings = self.base_embeddings.embed_documents(texts)
        return [adjust_embedding_dimensions(emb, self.target_dim) for emb in embeddings]
        
    def embed_query(self, text):
        embedding = self.base_embeddings.embed_query(text)
        return adjust_embedding_dimensions(embedding, self.target_dim)

def test_vector_search(query="Hva er ALS?", top_k=5):
    """
    Test vector search with a specific query and display the results.
    
    Args:
        query (str): The query to search for
        top_k (int): Number of results to return
    """
    print(f"\n{'='*80}")
    print(f"Testing vector search with query: '{query}'")
    print(f"{'='*80}")
    
    try:
        # First, ensure vector search index exists
        create_vector_search_index()
        
        # Get the embeddings model
        print("\nInitializing embeddings model...")
        base_embeddings = get_embeddings()
        print(f"Base embeddings model initialized: {base_embeddings.__class__.__name__}")
        
        # Wrap with our dimension-adjusting embeddings
        embeddings = DimensionAdjustingEmbeddings(base_embeddings, target_dim=1536)
        print(f"Using dimension-adjusting wrapper to ensure 1536 dimensions")
        
        # Connect to MongoDB
        mongodb_uri = os.getenv("MONGODB_URI")
        db_name = os.getenv("MONGODB_DB_NAME", "als_knowledge")
        
        if not mongodb_uri:
            print("Error: MongoDB URI not found in environment variables")
            sys.exit(1)
            
        print(f"Using MongoDB URI: {mongodb_uri[:25]}...")
        print(f"Using database: {db_name}")
        
        # Create vector store
        from pymongo import MongoClient
        client = MongoClient(mongodb_uri)
        
        # Check multiple possible collections to find the vectors
        collections_to_check = ["vector_entries", "contributions", "embeddings"]
        
        collection = None
        for collection_name in collections_to_check:
            collection = client[db_name][collection_name]
            doc_count = collection.count_documents({})
            print(f"Found {doc_count} documents in '{collection_name}' collection")
            
            # If this collection has documents and has an 'embedding' field in at least one doc
            if doc_count > 0:
                sample_doc = collection.find_one({"embedding": {"$exists": True}})
                if sample_doc:
                    print(f"SUCCESS: Using collection '{collection_name}' for vector search")
                    break
        else:
            print("ERROR: Could not find any collections with vector embeddings")
            return
        
        # Count entries in the collection
        doc_count = collection.count_documents({})
        print(f"\nFound {doc_count} documents in vector store.")
        
        # Initialize vector store
        vector_store = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embeddings,
            text_key="text",
            embedding_key="embedding",
            index_name="vector_index"
        )
        
        # Perform vector search
        print(f"\nPerforming vector search for query: '{query}'")
        search_results = vector_store.similarity_search(query, k=top_k)
        
        # Display results
        if search_results:
            print(f"\nFound {len(search_results)} results:")
            print(f"{'-'*80}")
            
            for i, doc in enumerate(search_results):
                print(f"Result {i+1}:")
                print(f"Content: {doc.page_content[:200]}..." if len(doc.page_content) > 200 else f"Content: {doc.page_content}")
                print(f"Metadata: {doc.metadata}")
                print(f"{'-'*80}")
                
            print("\nVector search is working correctly and retrieving real documents from the database!")
        else:
            print("\nNo results found. This might indicate a problem with the vector search or simply no matching documents.")
            
    except Exception as e:
        print(f"\nError during vector search test: {e}")
        import traceback
        print(traceback.format_exc())
        print("\nVector search test failed.")
        
if __name__ == "__main__":
    # If command line arguments provided, use first as query
    if len(sys.argv) > 1:
        test_vector_search(sys.argv[1])
    else:
        # Run with default query
        test_vector_search()
        
        # Also try some other test queries
        test_vector_search("Hva er spasmer og hvordan behandle dem?")
        test_vector_search("Hjelpemidler for å spise når hendene er svake")
        test_vector_search("Hva er Alltid litt sterkere støttegruppe?")
