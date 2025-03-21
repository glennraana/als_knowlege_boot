"""
This script tests if a specific query can be found in the vector store.
It searches for a contribution about nutrition mentioned by the user.
"""

import os
from dotenv import load_dotenv
from rag.embeddings import get_embeddings
from rag.retriever import get_retriever
from rag.vectorstore import MongoDBVectorStore
from langchain.schema import Document

# Load environment variables
load_dotenv()

def main():
    # Get embeddings model
    print("Loading embedding model...")
    embedding_model = get_embeddings()
    
    # Create vector store
    print("Creating vector store...")
    vector_store = MongoDBVectorStore(embedding_model)
    
    # Test queries
    test_queries = [
        "jeg får i meg for lite næring",
        "næring",
        "ernæring",
        "vekttap",
        "appetitt",
        "problemer med næring",
        "problemer med å spise",
        "spise",
        "vanskelig å spise",
        "vanskeligheter med å spise",
        "gått ned i vekt",
    ]
    
    # Test each query
    for query in test_queries:
        print(f"\n\nTesting query: '{query}'")
        try:
            # Get similar documents directly from vector store
            docs = vector_store.similarity_search(query, k=5)
            
            if docs:
                print(f"Found {len(docs)} documents for query '{query}':")
                for i, doc in enumerate(docs):
                    print(f"\nDocument {i+1}:")
                    print(f"Content: {doc.page_content}")
                    if hasattr(doc, 'metadata') and doc.metadata:
                        print(f"Metadata: {doc.metadata}")
            else:
                print(f"No documents found for query '{query}'")
        except Exception as e:
            print(f"Error searching for query '{query}': {e}")
    
    # Test using our retriever (which should fix any dimension issues)
    print("\n\nTesting with retriever:")
    retriever = get_retriever(embedding_model, k=5)
    for query in test_queries:
        print(f"\nTesting retriever query: '{query}'")
        try:
            docs = retriever.invoke(query)
            if docs:
                print(f"Found {len(docs)} documents for query '{query}':")
                for i, doc in enumerate(docs):
                    print(f"\nDocument {i+1}:")
                    print(f"Content: {doc.page_content}")
                    if hasattr(doc, 'metadata') and doc.metadata:
                        print(f"Metadata: {doc.metadata}")
            else:
                print(f"No documents found for query '{query}'")
        except Exception as e:
            print(f"Error searching with retriever for query '{query}': {e}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
