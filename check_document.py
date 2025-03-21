"""
Script for å sjekke om et dokument ble lagret i databasen og lagt til i RAG-systemet
"""

import os
from dotenv import load_dotenv
from db.connection import get_database
from db.document_models import Document
import gridfs

# Last inn miljøvariabler
load_dotenv()

# Hent databasen
db = get_database()
documents_collection = db["documents"]
fs = gridfs.GridFS(db)

# Sjekk dokumenter i databasen
print("=== Dokumenter i databasen ===")
docs = list(documents_collection.find({"source_url": {"$exists": True}}))
print(f"Fant {len(docs)} dokumenter med source_url")

for doc in docs:
    print(f"\nDokument: {doc.get('name')}")
    print(f"URL: {doc.get('source_url')}")
    print(f"Type: {doc.get('type')}")
    print(f"Størrelse: {doc.get('size')} bytes")
    print(f"Prosessert for RAG: {doc.get('processed', False)}")
    print(f"Tags: {doc.get('tags', [])}")
    
    # Sjekk om dokumentet har et fil-ID
    file_id_str = doc.get('file_id')
    if file_id_str:
        try:
            from bson import ObjectId
            file_id = ObjectId(file_id_str)
            
            # Sjekk om filen finnes i GridFS
            if fs.exists(file_id):
                print(f"Filen finnes i GridFS")
                
                # Hent filen fra GridFS
                grid_out = fs.get(file_id)
                content_length = len(grid_out.read())
                print(f"Filinnhold størrelse: {content_length} bytes")
            else:
                print(f"Filen finnes IKKE i GridFS")
        except Exception as e:
            print(f"Feil ved sjekk av fil: {e}")
    
    print("---")

# Test RAG-systemet med en relatert spørring
print("\n=== Test av RAG-systemet ===")
try:
    from rag.vectorstore import get_vectorstore
    from rag.retriever import get_retriever
    
    vector_store = get_vectorstore()
    retriever = get_retriever()
    
    # Lag en relatert spørring
    query = "Hva er GAIN-studien?"
    
    # Hent relevante dokumenter
    docs = retriever.get_relevant_documents(query)
    
    print(f"Spørring: {query}")
    print(f"Fant {len(docs)} relevante dokumenter:")
    
    for i, doc in enumerate(docs, 1):
        print(f"\nDokument {i}:")
        print(f"Kilde: {doc.metadata.get('source', 'Ukjent')}")
        print(f"Dokument ID: {doc.metadata.get('document_id', 'Ukjent')}")
        
        # Vis litt av innholdet
        content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        print(f"Innhold: {content_preview}")
        
except Exception as e:
    print(f"Feil ved test av RAG-system: {e}")
