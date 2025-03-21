from typing import List, Dict, Any, Optional, Callable
import os
import logging
import io
import re
from dotenv import load_dotenv
import openai
from openai import OpenAI

# Importer streamlit for secrets håndtering
try:
    import streamlit as st
    has_streamlit = True
except ImportError:
    has_streamlit = False

# Load environment variables
load_dotenv()

# Hent API-nøkkel fra Streamlit secrets hvis tilgjengelig
if has_streamlit and hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
    os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
    logging.info("Using OpenAI API key from Streamlit secrets")

def get_openai_client():
    """
    Get an instance of the OpenAI client
    """
    # Sjekk om API-nøkkelen finnes
    if not os.environ.get("OPENAI_API_KEY"):
        error_msg = "OPENAI_API_KEY not found in environment variables or Streamlit secrets."
        if has_streamlit:
            st.error(error_msg)
        raise ValueError(error_msg)
    
    # Opprett en OpenAI klient uten ekstra parametere som kan skape problemer
    try:
        # Prøv først med standard konfigurasjon
        return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    except TypeError as e:
        # Hvis det er argument-problemer, logg det og prøv en enklere konfigurasjon
        logging.warning(f"Error creating OpenAI client with standard config: {e}")
        # Import direkte for å sikre vi bruker korrekt klasse
        from openai import OpenAI as DirectOpenAI
        return DirectOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class SimpleRetriever:
    """Simple retriever that uses a vector store"""
    
    def __init__(self, vectorstore, k=5):
        """
        Initialize the retriever
        
        Args:
            vectorstore: Vector store to use for retrieval
            k: Number of documents to retrieve
        """
        self.vectorstore = vectorstore
        self.k = k
    
    def get_relevant_documents(self, query):
        """
        Get documents relevant to a query
        
        Args:
            query: Query string
            
        Returns:
            List of relevant documents
        """
        return self.vectorstore.similarity_search(query, k=self.k)

def get_retriever(embedding_function, k=5):
    """
    Get a retriever for RAG
    
    Args:
        embedding_function: Function for generating embeddings
        k: Number of documents to retrieve (default: 5)
        
    Returns:
        A configured retriever
    """
    from rag.vectorstore import get_vectorstore
    
    # Create a vector store with the embedding function
    vectorstore = get_vectorstore(embedding_function)
    
    # Create and return a retriever
    return SimpleRetriever(vectorstore, k=k)

def get_rag_chain(retriever, k=5):
    """
    Get a RAG chain that combines retrieval with generation
    
    Args:
        retriever: Retriever to use
        k: Number of documents to retrieve
        
    Returns:
        A RAG chain
    """
    return CustomRAGChain(retriever)

class CustomRAGChain:
    """Simple RAG chain implementation"""
    
    def __init__(self, retriever):
        """
        Initialize the RAG chain
        
        Args:
            retriever: Retriever to use for fetching documents
        """
        self.retriever = retriever
        self.client = get_openai_client()
    
    def _format_documents(self, docs):
        """
        Format a list of documents into a context string
        
        Args:
            docs: List of documents
            
        Returns:
            String containing the formatted context
        """
        context = ""
        for i, doc in enumerate(docs):
            context += f"\nDocument {i+1}:\n"
            context += f"Content: {doc.get('page_content', '')}\n"
            
            # Add metadata if available
            metadata = doc.get('metadata', {})
            if metadata:
                context += "Metadata: "
                context += ", ".join([f"{k}: {v}" for k, v in metadata.items()])
                context += "\n"
        
        return context
    
    def invoke(self, input_dict):
        """
        Invoke the RAG chain
        
        Args:
            input_dict: Dictionary with "input" key containing the query
            
        Returns:
            Dictionary with response
        """
        query = input_dict.get("input")
        if not query:
            return {"answer": "Ingen spørsmål ble gitt.", "source_documents": []}
        
        # Retrieve relevant documents
        docs = self.retriever.get_relevant_documents(query)
        
        # Format documents as context
        context = self._format_documents(docs)
        
        # Prepare the prompt for OpenAI
        prompt = f"""Du er en hjelpsom assistent som svarer på spørsmål om ALS (Amyotrofisk Lateral Sklerose) på norsk. 
Basert på følgende informasjon, besvar brukerens spørsmål så godt du kan. 
Hvis informasjonen du har ikke er tilstrekkelig, si fra om det og gi ditt beste svar basert på din generelle kunnskap.

Kontekst:
{context}

Brukers spørsmål: {query}

VIKTIG:
1. Svar på norsk
2. Vær empatisk, forståelsesfull og respektfull
3. Vær konkret og praktisk i dine svar
4. Ta hensyn til at brukeren kan være en ALS-pasient eller pårørende
5. Vær ærlig om begrensningene i din kunnskap
"""
        
        try:
            # Generate a response using OpenAI
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Du er en hjelpsom assistent for ALS-pasienter som svarer på norsk. Vær konkret, praktisk og empatisk."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
            )
            
            answer = response.choices[0].message.content
            
            return {
                "answer": answer,
                "source_documents": docs
            }
        except Exception as e:
            error_msg = f"Feil ved generering av svar: {str(e)}"
            logging.error(error_msg)
            return {
                "answer": f"Beklager, jeg kunne ikke generere et svar på grunn av en teknisk feil. {error_msg}",
                "source_documents": docs
            }
