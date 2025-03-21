"""
Test script for MongoDB operations in the ALS Knowledge application.
This script tests the basic MongoDB operations to ensure the integration is working correctly.
"""

import os
import sys
from datetime import datetime
from pprint import pprint
import uuid

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project components
from db.operations import (
    save_contribution, 
    get_contribution, 
    get_all_contributions, 
    save_vector_embedding, 
    search_similar_embeddings,
    create_vector_search_index
)
from db.models import Contribution

def test_contribution_operations():
    """Test saving and retrieving contributions"""
    print("\n--- Testing Contribution Operations ---")
    
    # Create a test contribution
    test_id = f"test-{uuid.uuid4()}"
    test_contribution = Contribution(
        problem="Test problem for MongoDB integration",
        aids_used="Test aids used for MongoDB testing",
        medicine_info="Test medicine info",
        contributor_name="Test User",
        file_type=None,
        file_name=None,
        file_content=None
    )
    
    # Save the contribution
    try:
        contribution_id = save_contribution(test_contribution)
        print(f"✓ Contribution saved with ID: {contribution_id}")
    except Exception as e:
        print(f"✗ Error saving contribution: {e}")
        return False
    
    # Retrieve all contributions
    try:
        contributions = get_all_contributions()
        print(f"✓ Retrieved {len(contributions)} contributions")
        # Print the last contribution
        if contributions:
            print("Latest contribution:")
            latest = contributions[-1]
            print(f"  Problem: {latest.problem}")
            print(f"  Aids: {latest.aids_used}")
            print(f"  Contributor: {latest.contributor_name}")
    except Exception as e:
        print(f"✗ Error retrieving contributions: {e}")
        return False
    
    return True

def test_vector_operations():
    """Test vector embedding operations"""
    print("\n--- Testing Vector Operations ---")
    
    # Create test vector search index
    try:
        create_vector_search_index()
        print("✓ Vector search index created or confirmed")
    except Exception as e:
        print(f"✗ Error creating vector search index: {e}")
        return False
    
    # Save a test embedding
    test_text = "This is a test document about ALS symptoms and mobility aids"
    test_embedding = [0.1] * 1536  # Mock embedding with 1536 dimensions (OpenAI dimension size)
    test_metadata = {
        "source": "test",
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        doc_id = save_vector_embedding(test_text, test_embedding, test_metadata)
        print(f"✓ Vector embedding saved with ID: {doc_id}")
    except Exception as e:
        print(f"✗ Error saving vector embedding: {e}")
        return False
    
    # Search for similar embeddings
    try:
        query_embedding = [0.1] * 1536  # Same embedding for testing
        results = search_similar_embeddings(query_embedding, n_results=2)
        print(f"✓ Found {len(results)} similar embeddings")
        if results:
            print("First result:")
            print(f"  Text: {results[0].get('text', '')[:50]}...")
            print(f"  Score: {results[0].get('score', 0)}")
    except Exception as e:
        print(f"✗ Error searching for similar embeddings: {e}")
        return False
    
    return True

def run_tests():
    """Run all tests"""
    print("=== Testing MongoDB Integration ===")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Run contribution tests
    contribution_success = test_contribution_operations()
    
    # Run vector tests
    vector_success = test_vector_operations()
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Contribution operations: {'Passed' if contribution_success else 'Failed'}")
    print(f"Vector operations: {'Passed' if vector_success else 'Failed'}")
    
    if contribution_success and vector_success:
        print("\n✓ All tests passed! MongoDB integration is working correctly.")
    else:
        print("\n✗ Some tests failed. Check the logs above for errors.")

if __name__ == "__main__":
    run_tests()
