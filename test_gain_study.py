"""
Script for å teste om GAIN-studien er tilgjengelig i vektorlageret
"""

import os
from dotenv import load_dotenv
from db.connection import get_database
from rag.embeddings import get_embeddings
from langchain.schema import Document

# Last inn miljøvariabler
load_dotenv()

# Hent databasen
db = get_database()
embeddings_collection = db["embeddings"]

print("=== Søk etter GAIN-studien i vektorlageret ===")

# Enkelt søk i embeddings-samlingen
gain_docs = list(embeddings_collection.find({"text": {"$regex": "GAIN", "$options": "i"}}))
print(f"Fant {len(gain_docs)} dokumenter med 'GAIN' i teksten")

for doc in gain_docs[:3]:  # Vis de første 3 dokumentene
    print(f"\nDokument:")
    print(f"Tekst (utdrag): {doc.get('text')[:300]}...")
    print(f"Metadata: {doc.get('metadata', {})}")
    print("---")

# Test med embedding-basert søk
print("\n=== Semantisk søk etter GAIN-studien ===")
try:
    # Hent embeddings-modell
    embeddings_model = get_embeddings()
    
    # Lag embedding for spørringen
    query = "GAIN-studien genteknologi ALS forskning"
    query_embedding = embeddings_model.embed_query(query)
    
    # Søk etter lignende embeddings i databasen
    from db.operations import search_similar_embeddings
    
    similar_docs = search_similar_embeddings(query_embedding, limit=5)
    print(f"Fant {len(similar_docs)} relevante dokumenter med semantisk søk")
    
    for doc in similar_docs:
        print(f"\nDokument:")
        print(f"Tekst (utdrag): {doc.get('text')[:300]}...")
        print(f"Metadata: {doc.get('metadata', {})}")
        print(f"Similarity score: {doc.get('score', 0)}")
        print("---")
        
except Exception as e:
    print(f"Feil ved semantisk søk: {e}")

# Test med direkte spørring til OpenAI
print("\n=== Test med OpenAI RAG ===")
try:
    from langchain_openai import ChatOpenAI
    from langchain.schema import Document
    from langchain.prompts import ChatPromptTemplate
    
    # Hent dokumenter om GAIN-studien
    gain_docs = list(embeddings_collection.find({"text": {"$regex": "GAIN", "$options": "i"}}))
    
    # Konverter til LangChain-dokumenter
    documents = []
    for doc in gain_docs[:5]:  # Bruk de første 5 dokumentene
        text = doc.get('text', '')
        metadata = doc.get('metadata', {})
        documents.append(Document(page_content=text, metadata=metadata))
    
    # Lag en prompt
    prompt = ChatPromptTemplate.from_template("""
    Du er en hjelpsomhet assistent for ALS-pasienter og pårørende. 
    Bruk kun informasjon fra følgende dokumenter til å svare på spørsmålet.
    Hvis du ikke finner informasjon i dokumentene, si at du ikke har tilstrekkelig informasjon.
    
    Dokumenter:
    {context}
    
    Spørsmål: {question}
    """)
    
    # Hent LLM
    # Bruk miljøvariabelen fra .env
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    # Kombiner dokumentene til en kontekst
    context = "\n\n".join([f"Dokument {i+1}:\n{doc.page_content}" for i, doc in enumerate(documents)])
    
    # Generer svar
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": "Hva er GAIN-studien og hvilken genetisk forskning gjør de på ALS?"})
    
    print(f"Spørring: Hva er GAIN-studien og hvilken genetisk forskning gjør de på ALS?")
    print(f"\nSvar fra OpenAI:")
    print(response.content)
    
except Exception as e:
    print(f"Feil ved OpenAI RAG-test: {e}")
