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
        import logging
        
        logging.info(f"Similarity search for query: {query}")
        
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_function(query)
            logging.info(f"Generated embedding of length: {len(query_embedding)}")
            
            # Count documents in embeddings collection
            doc_count = embeddings_collection.count_documents({})
            logging.info(f"Number of documents in embeddings collection: {doc_count}")
            
            if doc_count == 0:
                logging.warning("No documents in embeddings collection!")
                return []
            
            # Forsøk en alternativ metode - hent alle dokumenter og sorter manuelt
            # Dette er bare en midlertidig løsning da det ikke er skalerbart for store databaser
            logging.info("Bruker alternativ søkemetode som ikke krever vektorindeks")
            
            # Begrenset til maksimalt 500 dokumenter for å unngå ytelsesproblem
            limit = min(500, doc_count)
            all_docs = list(embeddings_collection.find().limit(limit))
            logging.info(f"Hentet {len(all_docs)} dokumenter for sammenligning")
            
            # Forbered resultater med skåre
            results_with_scores = []
            import numpy as np
            
            for doc in all_docs:
                doc_embedding = doc.get("embedding")
                if not doc_embedding:
                    continue
                
                # Beregn cosine similarity
                try:
                    similarity = np.dot(query_embedding, doc_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                    )
                    results_with_scores.append((doc, similarity))
                except Exception as e:
                    logging.error(f"Feil ved beregning av similarity: {e}")
                    continue
            
            # Sorter etter skåre
            results_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Ta de k beste resultatene
            top_results = results_with_scores[:k]
            
            # Konverter til dokument-formatet
            documents = []
            for doc, score in top_results:
                documents.append({
                    "page_content": doc["text"],
                    "metadata": {**doc.get("metadata", {}), "score": round(score, 3)}
                })
            
            logging.info(f"Fant {len(documents)} relevante dokumenter")
            return documents
            
        except Exception as e:
            logging.error(f"Error in similarity_search: {e}")
            print(f"Error in similarity_search: {e}")
            import traceback
            logging.error(traceback.format_exc())
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
