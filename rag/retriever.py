from typing import List, Dict, Any, Optional, Callable
import os
import logging
import io
import re
from dotenv import load_dotenv
import openai
from openai import OpenAI

# Patch for OpenAI client to fix TypeError with 'proxies' parameter
import importlib
if hasattr(openai, '_base_client'):
    base_client = importlib.import_module('openai._base_client')
    if hasattr(base_client, 'SyncHttpxClientWrapper'):
        orig_init = base_client.SyncHttpxClientWrapper.__init__
        
        def patched_init(self, *args, **kwargs):
            # Remove problematic proxies parameter if it exists
            if 'proxies' in kwargs:
                del kwargs['proxies']
            return orig_init(self, *args, **kwargs)
        
        base_client.SyncHttpxClientWrapper.__init__ = patched_init
        logging.info("Successfully patched OpenAI client to handle proxy issues")

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
    
    # Opprett OpenAI klient med bare de nødvendige parametrene
    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    return client

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
        Format documents for the prompt
        
        Args:
            docs: List of documents
            
        Returns:
            Formatted string
        """
        if not docs:
            return "Ingen relevante dokumenter funnet."
            
        formatted_docs = []
        
        for i, doc in enumerate(docs):
            content = doc.get('page_content', 'Tomt dokument.')
            metadata = doc.get('metadata', {})
            
            # Format metadata for better context
            metadata_info = []
            if metadata.get('kategori'):
                metadata_info.append(f"Kategori: {metadata.get('kategori')}")
            if metadata.get('innholdstype'):
                metadata_info.append(f"Type: {metadata.get('innholdstype')}")
            if metadata.get('title'):
                metadata_info.append(f"Tittel: {metadata.get('title')}")
            if metadata.get('tags'):
                metadata_info.append(f"Tagger: {', '.join(metadata.get('tags'))}")
            if metadata.get('opprettet_av'):
                metadata_info.append(f"Forfatter: {metadata.get('opprettet_av')}")
            if metadata.get('vanskelighetsgrad'):
                metadata_info.append(f"Vanskelighetsgrad: {metadata.get('vanskelighetsgrad')}")
            if metadata.get('score'):
                metadata_info.append(f"Relevans: {metadata.get('score')}")
                
            # Format the document with numbered header and metadata
            doc_header = f"DOKUMENT {i+1}"
            if metadata_info:
                doc_header += f" ({'; '.join(metadata_info)})"
            
            # Fremhev hvis dette er en personlig erfaring
            if metadata.get('innholdstype') == 'personlig_erfaring':
                doc_header = f"{doc_header} - PERSONLIG ERFARING"
                
            formatted_doc = f"{doc_header}:\n{content}\n"
            formatted_docs.append(formatted_doc)
            
        return "\n\n".join(formatted_docs)
    
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
        
        import logging
        logging.info(f"RAG chain invoked with query: {query}")
        
        # Retrieve relevant documents
        try:
            docs = self.retriever.get_relevant_documents(query)
            logging.info(f"Retrieved {len(docs)} relevant documents")
            
            # Log top documents for debugging
            for i, doc in enumerate(docs[:3]):  # Only log first 3
                content = doc.get('page_content', '')[:100]  # Truncate to 100 chars
                metadata = doc.get('metadata', {})
                score = metadata.get('score', 'N/A')
                kategori = metadata.get('kategori', 'ukjent')
                innholdstype = metadata.get('innholdstype', 'ukjent')
                logging.info(f"Document {i+1}: Score={score}, Kategori={kategori}, Type={innholdstype}, Content={content}...")
            
            # Format documents as context
            context = self._format_documents(docs)
            
            # If no relevant docs found, log it but continue with generic knowledge
            if not docs:
                logging.warning("No relevant documents found for query, will use generic knowledge")
                context = "Ingen relevante dokumenter funnet i kunnskapsbasen. Bruk din generelle kunnskap."
            
            # Sjekk om vi har personlige erfaringer i dokumentene
            har_personlig_erfaring = any(
                doc.get('metadata', {}).get('innholdstype') == 'personlig_erfaring' 
                for doc in docs
            )
            
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
6. Hvis relevant dokumentasjon er tilgjengelig, henvis til den i svaret ditt
7. Gi utfyllende og detaljerte svar basert på konteksten når relevant informasjon finnes
8. Avslutt ALLTID svaret ditt med en henvisning til støttegruppen "Alltid litt sterkere" som en ressurs for ytterligere hjelp og støtte
9. Hvis den tilgjengelige informasjonen inneholder personlige erfaringer, fremhev disse tydelig i svaret ditt
10. Vær grundig og gi omfattende svar - ikke korte, generelle svar
"""
            
            try:
                logging.info("Sending request to OpenAI...")
                # Generate a response using OpenAI
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "Du er en hjelpsom assistent for ALS-pasienter som svarer på norsk. Vær konkret, praktisk og empatisk. Gi alltid utfyllende og grundige svar. Henvis alltid til ALS-foreningen 'Alltid litt sterkere' når det er relevant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2000  # Økt fra standard for å tillate mer omfattende svar
                )
                
                answer = response.choices[0].message.content
                logging.info(f"Generated response of length: {len(answer)}")
                
                # Sjekk om svaret inneholder referanse til "Alltid litt sterkere" - hvis ikke, legg til
                if "Alltid litt sterkere" not in answer:
                    answer += "\n\nFor ytterligere hjelp og støtte anbefaler jeg at du kontakter ALS-foreningen 'Alltid litt sterkere', som kan gi deg personlig veiledning og støtte i din situasjon."
                
                return {
                    "answer": answer,
                    "source_documents": docs
                }
            except Exception as e:
                error_msg = f"Feil ved generering av svar fra OpenAI: {str(e)}"
                logging.error(error_msg)
                import traceback
                logging.error(traceback.format_exc())
                return {
                    "answer": f"Beklager, jeg kunne ikke generere et svar på grunn av en teknisk feil med OpenAI-tjenesten. {error_msg}",
                    "source_documents": docs
                }
        except Exception as e:
            error_msg = f"Feil ved henting av relevante dokumenter: {str(e)}"
            logging.error(error_msg)
            import traceback
            logging.error(traceback.format_exc())
            return {
                "answer": f"Beklager, jeg kunne ikke søke i kunnskapsbasen på grunn av en teknisk feil. {error_msg}",
                "source_documents": []
            }
