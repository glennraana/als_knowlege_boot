"""
Test script to verify that the vector search functionality is working correctly.
This script will:
1. Initialize the embeddings model
2. Get a test embedding
3. Perform a vector search
4. Display the results
"""

import os
import sys
import traceback
import json
from dotenv import load_dotenv
from rag.embeddings import get_embeddings
from db.operations import search_similar_embeddings, create_vector_search_index

# Load environment variables
load_dotenv()

def test_vector_search():
    """Test vector search functionality"""
    print("\n==== Testing Vector Search Functionality ====")
    
    success = True
    
    # Step 1: Initialize the embeddings model
    try:
        print("Initializing embeddings model...")
        embeddings = get_embeddings()
        print(f"✓ Embeddings model initialized successfully (Type: {type(embeddings).__name__})")
    except Exception as e:
        print(f"✗ Error initializing embeddings model: {e}")
        traceback.print_exc(file=sys.stdout)
        success = False
        # Try to continue with fallback embeddings
        try:
            from langchain_community.embeddings import FakeEmbeddings
            print("Using fallback embeddings for testing...")
            embeddings = FakeEmbeddings(size=1536)  # Match the expected dimension
        except Exception:
            print("Could not initialize fallback embeddings. Aborting test.")
            return
    
    # Step 2: Get a test embedding for a query
    try:
        print("\nGenerating test embedding...")
        test_query = "Hjelp med mobilitet og bevegelse"
        query_embedding = embeddings.embed_query(test_query)
        print(f"✓ Test embedding generated for: '{test_query}'")
        print(f"  - Embedding dimension: {len(query_embedding)}")
        print(f"  - First 5 values: {query_embedding[:5]}")
        
        # Check if the embedding contains non-zero values
        if all(val == 0 for val in query_embedding):
            print("⚠ Warning: All embedding values are zero, which may cause issues with cosine similarity")
            # Create a test embedding with small non-zero values
            query_embedding = [0.001] * len(query_embedding)
            print("  - Using non-zero test embedding instead")
    except Exception as e:
        print(f"✗ Error generating test embedding: {e}")
        traceback.print_exc(file=sys.stdout)
        success = False
        # Use a dummy embedding for testing
        query_embedding = [0.001] * 1536  # Small non-zero values
        print("  - Using dummy embedding for testing...")
    
    # Step 3: Check if vector search index is available
    try:
        print("\nChecking vector search index...")
        # Force re-creation of the vector search index
        import db.operations
        db.operations.HAS_VECTOR_SEARCH = None  # Reset the flag
        has_vector_search = create_vector_search_index()
        print(f"✓ Vector search index status: {'Available' if has_vector_search else 'Not available - will use fallback'}")
    except Exception as e:
        print(f"✗ Error checking vector search index: {e}")
        traceback.print_exc(file=sys.stdout)
        success = False
        has_vector_search = False
    
    # Step 4: Test the vector search function
    try:
        print("\nTesting vector search functionality...")
        
        # Verify the dimension of our query embedding and adjust if necessary
        expected_dim = 1536  # Default for OpenAI embeddings
        embedding_dim = len(query_embedding)
        
        if embedding_dim != expected_dim:
            print(f"⚠ Query embedding dimension ({embedding_dim}) does not match expected dimension ({expected_dim})")
            print("  - Adjusting embedding dimension...")
            
            if embedding_dim < expected_dim:
                # Pad with zeros
                query_embedding = query_embedding + [0.0] * (expected_dim - embedding_dim)
            else:
                # Truncate
                query_embedding = query_embedding[:expected_dim]
            
            print(f"  - Adjusted embedding dimension: {len(query_embedding)}")
        
        # Perform the search
        print("\nExecuting search query...")
        results = search_similar_embeddings(query_embedding, n_results=5)
        
        # Check the results
        if not results:
            print("⚠ No results found. This could be normal if the database is empty.")
        else:
            print(f"✓ Found {len(results)} similar documents:")
            for i, result in enumerate(results, 1):
                print(f"\nResult {i}:")
                print(f"  - Text: {result.get('text', '')[:100]}...")
                print(f"  - Score: {result.get('score', 'N/A')}")
                print(f"  - Metadata: {json.dumps(result.get('metadata', {}), indent=2, ensure_ascii=False)[:200]}...")
    except Exception as e:
        print(f"✗ Error during vector search: {e}")
        traceback.print_exc(file=sys.stdout)
        success = False
    
    # Summary
    print("\n==== Test Summary ====")
    if success:
        print("✓ All tests completed successfully")
    else:
        print("⚠ Some tests encountered errors. See details above.")
    
    return success

def test_gradual_dimension_changes():
    """Test how the system handles embedding dimension changes over time"""
    print("\n==== Testing Dimension Handling ====")
    
    try:
        # Create test embeddings with different dimensions
        dimensions = [384, 768, 1536]
        embeddings_model = get_embeddings()
        
        for dim in dimensions:
            # Create a test embedding of the specific dimension
            if isinstance(embeddings_model.embed_query("test"), list):
                base_embedding = embeddings_model.embed_query("test for dimension " + str(dim))
                
                # Adjust to the target dimension
                if len(base_embedding) < dim:
                    test_embedding = base_embedding + [0.0] * (dim - len(base_embedding))
                else:
                    test_embedding = base_embedding[:dim]
                
                print(f"Testing with dimension {dim}: {len(test_embedding)} values")
                
                # Try to search with this embedding
                try:
                    results = search_similar_embeddings(test_embedding, n_results=3)
                    print(f"✓ Search successful with dimension {dim}")
                    print(f"  - Results: {len(results)}")
                except Exception as e:
                    print(f"✗ Search failed with dimension {dim}: {e}")
    
    except Exception as e:
        print(f"✗ Error during dimension testing: {e}")
        traceback.print_exc(file=sys.stdout)

if __name__ == "__main__":
    print("Running vector search tests...")
    success = test_vector_search()
    
    if success:
        print("\nRunning dimension handling tests...")
        test_gradual_dimension_changes()
