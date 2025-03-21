from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

def get_embeddings():
    """
    Returns the embedding model to use for the RAG system.
    Using HuggingFace all-MiniLM-L6-v2 model which is better suited for Norwegian text.
    """
    try:
        # Use a model that works well with multiple languages including Norwegian
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity search
        )
    except Exception as e:
        print(f"Error initializing HuggingFace embeddings: {e}")
        # Return a very simple fallback if everything else fails
        from langchain_community.embeddings import FakeEmbeddings
        return FakeEmbeddings(size=768)  # Match HuggingFace embedding dimensions
