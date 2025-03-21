import os
from typing import List, Dict, Any, Optional, Callable
import pymongo
from db.connection import get_database, get_gridfs, get_mongodb_uri

# Definer nødvendige attributter her i stedet for å importere fra operations
db = get_database()
embeddings_collection = db["embeddings"]

class MongoDBVectorStore:
    """Custom MongoDB vector store implementation"""
    
    def __init__(self, embedding_function: Callable):
        self.embedding_function = embedding_function
    
    def add_texts(
        self, 
        texts: List[str], 
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Add texts to the vector store
        
        Args:
            texts: List of text strings to embed and store
            metadatas: Metadata for each text (optional)
            
        Returns:
            List of document IDs
        """
        # Process metadatas
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Ensure metadatas and texts have the same length
        if len(metadatas) != len(texts):
            raise ValueError("Number of metadatas must match number of texts")
        
        try:
            # Embed texts one by one using our embedding function
            ids = []
            for text, metadata in zip(texts, metadatas):
                try:
                    # Generate embedding for the text
                    embedding = self.embedding_function(text)
                    
                    # Save to MongoDB
                    doc_id = embeddings_collection.insert_one({"text": text, "embedding": embedding, "metadata": metadata}).inserted_id
                    ids.append(str(doc_id))
                except Exception as e:
                    print(f"Error saving vector embedding: {e}")
                    # Continue with the next item instead of failing completely
            
            return ids
        except Exception as e:
            print(f"Error in add_texts: {e}")
            return []
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4,
        **kwargs: Any,
    ) -> List[Dict]:
        """
        Search for documents similar to the query string
        
        Args:
            query: Query string
            k: Number of documents to return
            
        Returns:
            List of documents with similarity scores
        """
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_function(query)
            
            # Search MongoDB for similar embeddings
            results = embeddings_collection.find({"embedding": {"$near": query_embedding}}).limit(k)
            
            # Convert to Document objects
            documents = []
            for result in results:
                documents.append({
                    "page_content": result["text"],
                    "metadata": result["metadata"]
                })
            
            return documents
        except Exception as e:
            print(f"Error in similarity_search: {e}")
            return []

def get_vectorstore(embedding_function: Callable):
    """
    Initialize and return a MongoDB vector store instance.
    
    Args:
        embedding_function: Function that generates embeddings for text
        
    Returns:
        A configured vector store instance
    """
    # Ensure the vector search index exists
    try:
        embeddings_collection.create_index([("embedding", pymongo.GEOSPHERE)])
    except Exception as e:
        print(f"Warning: Could not create vector search index: {e}")
    
    return MongoDBVectorStore(embedding_function)

def add_texts_to_vectorstore(
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    embedding_function: Callable
):
    """
    Add texts and their embeddings to the vector store
    
    Args:
        texts: List of text strings to embed and store
        metadatas: Metadata for each text
        embedding_function: Function that generates embeddings for text
    """
    vectorstore = get_vectorstore(embedding_function)
    return vectorstore.add_texts(texts, metadatas)
