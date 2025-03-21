import os
from typing import List, Dict, Any, Optional
from langchain.vectorstores.base import VectorStore
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from db.operations import embeddings_collection, save_vector_embedding, search_similar_embeddings, create_vector_search_index

class MongoDBVectorStore(VectorStore):
    """Custom MongoDB vector store implementation for LangChain"""
    
    def __init__(self, embedding: Embeddings):
        self.embedding = embedding
    
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
            # Embed texts
            embeddings = self.embedding.embed_documents(texts)
            
            # Save to MongoDB
            ids = []
            for text, embedding, metadata in zip(texts, embeddings, metadatas):
                try:
                    doc_id = save_vector_embedding(text, embedding, metadata)
                    ids.append(doc_id)
                except Exception as e:
                    print(f"Error saving vector embedding: {e}")
                    # Continue with the next item instead of failing completely
            
            if not ids:
                print("Warning: No documents were successfully saved to the vector store")
            
            return ids
        except Exception as e:
            print(f"Error in add_texts: {e}")
            # Return empty list instead of raising exception
            return []
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Find similar documents to the query string
        
        Args:
            query: Query string
            k: Number of documents to return
            
        Returns:
            List of documents most similar to the query
        """
        try:
            # Generate embedding for the query
            query_embedding = self.embedding.embed_query(query)
            
            # Search for similar embeddings
            results = search_similar_embeddings(query_embedding, n_results=k)
            
            # Convert results to LangChain Documents
            documents = []
            for result in results:
                documents.append(
                    Document(
                        page_content=result["text"],
                        metadata=result["metadata"]
                    )
                )
            
            return documents
        except Exception as e:
            print(f"Error in similarity_search: {e}")
            # Return empty list instead of raising exception
            return []
    
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> "MongoDBVectorStore":
        """
        Create a vector store from texts
        
        Args:
            texts: List of text strings to embed and store
            embedding: Embedding model to use for encoding text
            metadatas: Metadata for each text (optional)
            
        Returns:
            A MongoDBVectorStore instance
        """
        # Create instance
        instance = cls(embedding)
        
        # Add texts
        if texts:
            instance.add_texts(texts, metadatas)
        
        return instance

def get_vectorstore(embeddings: Embeddings):
    """
    Initialize and return a MongoDB vector store instance
    that integrates with langchain for document retrieval.
    
    Args:
        embeddings: Embedding model to use for encoding text
        
    Returns:
        A configured vector store instance
    """
    # Create the vector store
    vector_store = MongoDBVectorStore(embeddings)
    
    return vector_store

def add_texts_to_vectorstore(
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    embeddings: Embeddings
):
    """
    Add texts and their embeddings to the vector store
    
    Args:
        texts: List of text strings to embed and store
        metadatas: Metadata for each text
        embeddings: Embedding model to use
    """
    # Get vector store
    vector_store = get_vectorstore(embeddings)
    
    # Add texts
    doc_ids = vector_store.add_texts(texts=texts, metadatas=metadatas)
    
    return doc_ids
