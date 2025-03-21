from typing import List, Dict, Any, Optional, Callable, Tuple
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
        self.vectorstore = retriever.vectorstore
        self.llm = self.client
    
    def invoke(self, query: str) -> Dict[str, Any]:
        """
        Invoke the RAG chain
        
        Args:
            query: The user query
            
        Returns:
            Dict with answer and source documents
        """
        sanitized_query = query.strip()
        logging.info(f"RAG Chain invoked with query: '{sanitized_query}'")
        
        if not sanitized_query:
            return {"answer": "Vennligst skriv inn spørsmålet ditt om ALS.", "docs": []}
        
        # Analyse spørringens kompleksitet og type
        query_complexity, query_type = self._analyze_query_complexity(sanitized_query)
        logging.info(f"Spørringskompleksitet: {query_complexity}, Type: {query_type}")
        
        # Juster antall dokumenter basert på kompleksitet
        k_docs = self._get_k_docs_for_complexity(query_complexity)
        
        # Hent dokumenter fra vectorstore
        try:
            # Start med vektorsøk
            docs_with_scores = []
            if self.vectorstore:
                # Prøv først hybrid søk hvis tilgjengelig
                try:
                    if hasattr(self.vectorstore, "hybrid_search"):
                        docs_with_scores = self.vectorstore.hybrid_search(
                            sanitized_query, 
                            k=k_docs,
                            keyword_weight=0.3
                        )
                        if docs_with_scores:
                            logging.info(f"Hybrid søk returnerte {len(docs_with_scores)} dokumenter")
                    else:
                        logging.info("Hybrid søk er ikke tilgjengelig, bruker similarity_search_with_score")
                        docs_with_scores = self.vectorstore.similarity_search_with_score(sanitized_query, k=k_docs)
                        if docs_with_scores:
                            logging.info(f"Vektorsøk returnerte {len(docs_with_scores)} dokumenter")
                except Exception as e:
                    logging.error(f"Feil i hybrid_search eller similarity_search: {e}")
                    try:
                        # Fallback til vanlig vektorsøk
                        docs_with_scores = self.vectorstore.similarity_search_with_score(sanitized_query, k=k_docs)
                    except Exception as e2:
                        logging.error(f"Feil i similarity_search fallback: {e2}")
                
                # Sjekk gjennomsnittlig score for å vurdere kvaliteten på resultatene
                if docs_with_scores:
                    avg_score = sum(score for _, score in docs_with_scores) / len(docs_with_scores)
                    logging.info(f"Gjennomsnittlig score for resultater: {avg_score:.4f}")
                    
                    # Hvis gjennomsnittlig score er lav, prøv å finne personlige erfaringer
                    if avg_score < 0.75 and not hasattr(self.vectorstore, "hybrid_search"):
                        logging.info("Lav gjennomsnittlig score, prøver å søke etter personlige erfaringer...")
                        try:
                            personal_experiences = self.vectorstore.find_personal_experiences(sanitized_query, k=2)
                            if personal_experiences:
                                # Kombiner med eksisterende resultater, men prioriter personlige erfaringer
                                logging.info(f"Fant {len(personal_experiences)} personlige erfaringer")
                                combined_docs = []
                                
                                # Legg til personlige erfaringer først
                                for doc in personal_experiences:
                                    # Bruk en høy score for å sikre at de kommer først
                                    combined_docs.append((doc, 0.95))
                                
                                # Legg til vektorsøkresultater, men unngå duplikater
                                for doc, score in docs_with_scores:
                                    if doc not in [d for d, _ in combined_docs]:
                                        combined_docs.append((doc, score))
                                
                                docs_with_scores = combined_docs[:k_docs]  # Begrens til k_docs
                        except Exception as e:
                            logging.error(f"Feil ved søk etter personlige erfaringer: {e}")
            
            if not docs_with_scores:
                logging.warning("Ingen dokumenter funnet")
                return {
                    "answer": "Jeg kunne ikke finne noen spesifikk informasjon om dette i ALS-kunnskapsbasen. Vennligst prøv å omformulere spørsmålet eller spør om et relatert tema innen ALS.",
                    "docs": []
                }
            
            logging.info(f"Fant {len(docs_with_scores)} relevante dokumenter")
            
            # Velg modell og parametere basert på kompleksitet og type
            model, temperature, max_tokens = self._select_model_for_query(query_complexity, query_type)
            logging.info(f"Valgt modell: {model}, temperatur: {temperature}, max_tokens: {max_tokens}")
            
            # Bygg prompt til LLM basert på dokumenter og spørringstype
            prompt = self._build_prompt(sanitized_query, docs_with_scores)
            
            # Generer svar fra LLM
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": prompt["system"]},
                        {"role": "user", "content": prompt["user"]}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    presence_penalty=0.1,
                    frequency_penalty=0.2,
                    response_format={"type": "text"},
                )
                answer = response.choices[0].message.content.strip()
                
                # Logg bruk av tokens
                if hasattr(response, 'usage') and response.usage:
                    prompt_tokens = getattr(response.usage, 'prompt_tokens', 0)
                    completion_tokens = getattr(response.usage, 'completion_tokens', 0)
                    total_tokens = getattr(response.usage, 'total_tokens', 0)
                    logging.info(f"Token bruk: {prompt_tokens} prompt, {completion_tokens} completion, {total_tokens} total")
                
                # Forbered dokument-metadata for retur
                docs_metadata = []
                for doc, score in docs_with_scores:
                    doc_metadata = doc.get("metadata", {}).copy() if isinstance(doc, dict) else {}
                    doc_metadata["score"] = float(score) if isinstance(score, (int, float)) else 0.0
                    if isinstance(doc, dict) and "page_content" in doc:
                        doc_metadata["content_preview"] = doc["page_content"][:100] + "..." if len(doc["page_content"]) > 100 else doc["page_content"]
                    docs_metadata.append(doc_metadata)
                
                # Returner svaret sammen med metadata
                return {
                    "answer": answer,
                    "docs": docs_metadata,
                    "query_metadata": {
                        "complexity": query_complexity,
                        "type": query_type,
                        "model": model,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "tokens_used": total_tokens if 'total_tokens' in locals() else None
                    }
                }
            
            except Exception as e:
                logging.error(f"Feil ved generering av svar fra LLM: {e}")
                import traceback
                logging.error(traceback.format_exc())
                return {
                    "answer": "Beklager, jeg kunne ikke generere et svar akkurat nå. Vennligst prøv igjen senere.",
                    "docs": [],
                    "error": str(e)
                }
        
        except Exception as e:
            logging.error(f"Feil i RAG chain: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return {
                "answer": "Beklager, det oppstod en feil ved behandling av spørsmålet ditt. Vennligst prøv igjen.",
                "docs": [],
                "error": str(e)
            }

    def _select_model_for_query(self, complexity: str, query_type: str) -> Tuple[str, float, int]:
        """
        Select the appropriate model and parameters based on query complexity and type
        
        Args:
            complexity: Query complexity (LOW, MEDIUM, HIGH)
            query_type: Query type (FACTUAL, PERSONAL_EXPERIENCE, ADVICE, EMOTIONAL)
            
        Returns:
            Tuple of (model, temperature, max_tokens)
        """
        # Default modell er gpt-3.5-turbo
        model = "gpt-3.5-turbo-0125"
        
        # Sjekk om vi har tilgang til GPT-4
        try:
            # Bruk GPT-4 for komplekse eller emosjonelle spørsmål hvis tilgjengelig
            if complexity == "HIGH" or query_type == "EMOSJONELL":
                # Prøv å bruke gpt-4 for høy kompleksitet eller emosjonelle spørsmål
                model = "gpt-4-turbo-preview"
                logging.info(f"Valgte GPT-4 for {complexity} kompleksitet, {query_type} spørsmål")
        except Exception as e:
            logging.warning(f"Kunne ikke bruke GPT-4 modell: {e}, fallback til GPT-3.5")
            model = "gpt-3.5-turbo-0125"
        
        # Juster temperature basert på spørsmålstype
        if query_type == "FAKTABASERT":
            temperature = 0.1  # Lavere temperatur for mer presise, faktabaserte svar
            max_tokens = 800 
        elif query_type == "PERSONLIG_ERFARING":
            temperature = 0.5  # Litt høyere for personlige erfaringer, men fortsatt faktabasert
            max_tokens = 1000  # Lengre svar for personlige erfaringer
        elif query_type == "RÅD":
            temperature = 0.4  # Balansert temperatur for råd
            max_tokens = 1000  # Mer plass for detaljerte anbefalinger
        elif query_type == "EMOSJONELL":
            temperature = 0.7  # Høyere temperatur for mer empatiske svar
            max_tokens = 1000  # Plass for mer empatiske og støttende svar
        else:
            # Default verdier
            temperature = 0.3
            max_tokens = 800
        
        # Juster max_tokens basert på kompleksitet
        if complexity == "HIGH":
            max_tokens = min(max_tokens + 200, 1500)  # Øk tokens, men maks 1500
        elif complexity == "LOW":
            max_tokens = max(max_tokens - 100, 500)   # Reduser tokens, men minst 500
            
        logging.info(f"Valgte modellparametre: modell={model}, temp={temperature}, max_tokens={max_tokens}")
        return model, temperature, max_tokens

    def _get_k_docs_for_complexity(self, complexity: str) -> int:
        """
        Determine the number of documents to retrieve based on query complexity
        
        Args:
            complexity: Query complexity (LOW, MEDIUM, HIGH)
            
        Returns:
            Number of documents to retrieve
        """
        if complexity == "HIGH":
            return 8
        elif complexity == "MEDIUM":
            return 6
        else:
            return 4

    def _analyze_query_complexity(self, query: str) -> Tuple[str, str]:
        """
        Analyze the complexity and type of the query
        
        Args:
            query: The user query
            
        Returns:
            Tuple of (complexity, query_type)
        """
        # Vurder kompleksitet basert på lengde, antall spørsmålsord og nøkkelord
        query_length = len(query)
        
        # Norske spørsmålsord
        question_words = ["hva", "hvordan", "hvorfor", "når", "hvor", "hvem", "hvilke", "hvilken", 
                         "kan", "er", "vil", "må", "bør", "finnes", "hjelp", "fortell", "forklar"]
        
        # Kompleksitetsrelaterte ord
        complex_keywords = ["sammenheng", "årsak", "virkning", "sammenlignet", "forskjell", "likheter", 
                            "mekanisme", "detaljert", "dyptgående", "faglig", "vitenskapelig", 
                            "forskning", "studie", "konsekvens", "prognose", "utvikling", 
                            "predikere", "forutsi", "forebygge", "behandlingsalternativer"]
        
        # Tell forekomster
        question_word_count = sum(1 for word in question_words if word.lower() in query.lower())
        complex_keyword_count = sum(1 for word in complex_keywords if word.lower() in query.lower())
        
        # Vurder kompleksitet basert på disse metrikker
        if query_length > 100 or complex_keyword_count >= 2 or (query_length > 80 and question_word_count >= 2):
            complexity = "HIGH"
        elif query_length > 50 or complex_keyword_count >= 1 or question_word_count >= 2:
            complexity = "MEDIUM"
        else:
            complexity = "LOW"
        
        # Søk etter indikasjoner på ulike spørsmålstyper (norsk)
        factual_indicators = ["hva er", "hva betyr", "definer", "forklar", "hvordan fungerer", 
                             "hva innebærer", "fakta om", "informasjon om", "hva forårsaker",
                             "symptomer på", "kjennetegn", "diagnose", "behandling", "medisin", 
                             "statistikk", "data", "forskjell mellom", "sammenheng med", "genet", 
                             "mutasjon", "sod1"]
        
        personal_indicators = ["erfaring", "personlig", "opplevelse", "hvordan håndterer", 
                              "mestring", "noen som har", "andre med", "eksempel på", 
                              "historier om", "dele", "fortelle om", "hvordan var det", 
                              "hvordan opplevde", "hva gjorde du", "levd med", "lever med",
                              "håndtere", "takle", "hjelpemidler", "tilpasset"]
        
        advice_indicators = ["råd", "tips", "hjelp", "anbefalinger", "hvordan kan jeg", 
                            "hvordan bør", "hva anbefaler", "beste måte", "strategi", 
                            "teknikk", "verktøy", "metode", "fremgangsmåte", "steg", 
                            "veiledning", "guide", "hvordan skal", "hva kan jeg gjøre",
                            "hvordan forhindre", "forebygge", "unngå", "forbedre"]
        
        emotional_indicators = ["føler", "redd", "bekymret", "engstelig", "håp", "meningsfull", 
                               "deprimert", "trist", "frustrert", "sint", "ensom", "alene", 
                               "støtte", "takknemlig", "glad", "motivasjon", "mot", "styrke", 
                               "kjærlighet", "forhold", "familie", "pårørende", "hjelpe meg", 
                               "vanskelig", "utfordrende", "tøft", "slitsomt"]
        
        # Sjekk hvilken type spørring dette er
        query_lower = query.lower()
        
        # Telle indikatorer for hver type
        factual_count = sum(1 for indicator in factual_indicators if indicator.lower() in query_lower)
        personal_count = sum(1 for indicator in personal_indicators if indicator.lower() in query_lower)
        advice_count = sum(1 for indicator in advice_indicators if indicator.lower() in query_lower)
        emotional_count = sum(1 for indicator in emotional_indicators if indicator.lower() in query_lower)
        
        # Vektede scorer for å prioritere ulike typer
        factual_score = factual_count
        personal_score = personal_count * 1.2  # Gi litt høyere vekt til personlige erfaringer
        advice_score = advice_count * 1.1    # Gi litt høyere vekt til råd
        emotional_score = emotional_count * 1.3  # Gi høyere vekt til emosjonelle spørsmål
        
        # Bestem type basert på høyeste score
        max_score = max(factual_score, personal_score, advice_score, emotional_score)
        
        if max_score == 0:
            # Hvis ingen spesifikke indikatorer, bruk defaultverdier basert på kompleksitet
            if complexity == "HIGH":
                query_type = "FAKTABASERT"
            elif complexity == "MEDIUM":
                query_type = "RÅD"
            else:
                query_type = "PERSONLIG_ERFARING"
        elif max_score == factual_score:
            query_type = "FAKTABASERT"
        elif max_score == personal_score:
            query_type = "PERSONLIG_ERFARING"
        elif max_score == advice_score:
            query_type = "RÅD"
        else:
            query_type = "EMOSJONELL"
            
        logging.info(f"Analysert spørring '{query}' - Kompleksitet: {complexity}, Type: {query_type}")
        logging.info(f"Score: Faktabasert={factual_score}, Personlig={personal_score}, Råd={advice_score}, Emosjonell={emotional_score}")
        
        return complexity, query_type

    def _build_prompt(self, query: str, docs_with_scores: List[tuple]) -> Dict[str, str]:
        """
        Builds the prompt for the LLM
        
        Args:
            query: The user query
            docs_with_scores: The retrieved documents with their scores
            
        Returns:
            Dict with system and user messages
        """
        # Sorter dokumenter etter score (høyest først)
        sorted_docs = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)
        
        # Hent personlige erfaringer hvis tilgjengelig
        personal_experiences = []
        try:
            if self.vectorstore and hasattr(self.vectorstore, "find_personal_experiences"):
                personal_experiences = self.vectorstore.find_personal_experiences(query, k=2)
                if personal_experiences:
                    logging.info(f"Fant {len(personal_experiences)} personlige erfaringer for '{query}'")
        except Exception as e:
            logging.error(f"Feil ved henting av personlige erfaringer: {e}")
        
        # Kombiner dokumenter, med personlige erfaringer først
        combined_docs = []
        
        # Legg til personlige erfaringer først med høy score for å prioritere dem
        for doc in personal_experiences:
            if doc not in [d for d, _ in combined_docs]:
                combined_docs.append((doc, 0.95))
        
        # Legg til de sorterte dokumentene, men unngå duplikater
        for doc, score in sorted_docs:
            if doc not in [d for d, _ in combined_docs]:
                combined_docs.append((doc, score))
        
        # Begrens til max 8 dokumenter for å unngå å overstige token-grenser
        combined_docs = combined_docs[:8]
        
        # Formater dokumentene
        formatted_docs = "\n\n".join([
            self._format_document(doc, i+1) for i, (doc, _) in enumerate(combined_docs)
        ])
        
        # Analyser spørsmålets kompleksitet og type
        complexity, query_type = self._analyze_query_complexity(query)
        
        # Bygger systemprompten basert på spørsmålstype
        system_prompt = f"""Du er en hjelpsom assistent for personer med ALS (Amyotrofisk Lateral Sklerose) og deres pårørende i Norge. 
Du skal svare på spørsmål basert på informasjonen i kunnskapsbasen, men også bruke din generelle kunnskap når det er nødvendig.

VIKTIGE INSTRUKSJONER FOR FORMATERING OG STRUKTUR:
1. Svar ALLTID på norsk, med en vennlig, empatisk og informativ tone.
2. Strukturer ALLTID svaret ditt med tydelige overskrifter og avsnitt som følger:
   - Start med en "Innledning" som kort introduserer temaet
   - Fortsett med "Detaljer om [relevant tema]" som går i dybden
   - Inkluder alltid separate seksjoner for viktig informasjon
   - Avslutt med "Tilleggsanbefalinger" eller lignende konklusjon
3. Bruk ALLTID punktlister, nummererte lister og overskrifter for å gjøre informasjonen lett tilgjengelig.
4. Inkluder ALLTID personlige erfaringer når de er tilgjengelige, tydelig markert i en egen seksjon.
5. Fremhev viktig informasjon med fet skrift.

SPESIFIKKE INSTRUKSJONER BASERT PÅ SPØRSMÅLSTYPE:
"""

        # Legg til spesifikke instruksjoner basert på spørsmålstype
        if query_type == "FAKTABASERT":
            system_prompt += """
For faktabaserte spørsmål:
- Start med en klar, konsis innledning som definerer nøkkelbegreper
- Organiser informasjonen i logiske seksjoner med overskrifter
- Inkluder medisinsk terminologi der det er relevant, men forklar alltid i enkelt språk
- Avslutt med praktiske implikasjoner for ALS-pasienter
"""
        elif query_type == "PERSONLIG_ERFARING":
            system_prompt += """
For spørsmål om personlige erfaringer:
- Fremhev personlige erfaringer i en dedikert seksjon kalt "Personlige erfaringer"
- Presenter erfaringene som direkte sitater eller historier, med respekt for kilden
- Trekk ut konkrete råd og innsikter fra erfaringene
- Tilby ulike perspektiver hvis tilgjengelig
"""
        elif query_type == "RÅD":
            system_prompt += """
For rådgivende spørsmål:
- Presenter en nummerert liste med praktiske, konkrete tiltak
- Prioriter anbefalingene fra mest til minst viktig
- Forklar begrunnelsen bak hvert råd
- Inkluder både kortsiktige og langsiktige strategier
- Nevn når det er lurt å konsultere fagpersonell
"""
        elif query_type == "EMOSJONELL":
            system_prompt += """
For emosjonelle eller støtterelaterte spørsmål:
- Anerkjenn følelsene med empatiske formuleringer
- Tilby støtte og forståelse før praktiske råd
- Inkluder eksempler på mestringsstrategier
- Nevn støttegrupper og ressurser som "Alltid litt sterkere"
"""
        else:
            system_prompt += """
For generelle spørsmål:
- Gi en balansert oversikt over temaet
- Inkluder både medisinsk informasjon og praktiske implikasjoner
- Fremhev viktige punkter med overskrifter
- Avslutt med konkrete neste skritt eller ressurser
"""

        # Legg til instruksjoner for kildehåndtering
        system_prompt += """
HÅNDTERING AV KILDER:
- Basere svaret ditt hovedsakelig på informasjonen i kunnskapsbasen.
- Hvis du bruker generell kunnskap utover dokumentene, sørg for at den er nøyaktig og oppdatert.
- Vær tydelig på når du presenterer personlige erfaringer versus medisinsk informasjon.

STRUKTUR OG STIL:
- Formater svaret med tydelige seksjoner, overskrifter og punktlister (bruk markdown).
- Bruk en empatisk, støttende tone, spesielt ved sensitive temaer.
- Svar grundig og detaljert, i et format som ligner på følgende:

```
# Innledning
[Kort introduksjon til temaet]

## Detaljer om [tema]
[Utfyllende informasjon strukturert i avsnitt]

### Personlige erfaringer
[Hvis tilgjengelig, presenter personlige erfaringer]

## [Relevant underoverskrift]
- Punkt 1
- Punkt 2
- Punkt 3

## Tilleggsanbefalinger
[Oppsummering og eventuelle praktiske råd]
```

Husk at mottakeren kan være en ALS-pasient eller pårørende, så vær informativ, støttende og respektfull.
"""

        # Brukerprompten med spørsmål og kontekst
        user_prompt = f"""Her er spørsmålet mitt om ALS: {query}

Her er relevant informasjon fra ALS-kunnskapsbasen:

{formatted_docs}

Vennligst gi et detaljert, strukturert svar med tydelige overskrifter, avsnitt og punktlister. 
Inkluder personlige erfaringer hvis de er relevant, og sørg for at informasjonen er organisert logisk.
Følg instruksjonene for formatering og struktur nøye."""

        return {
            "system": system_prompt,
            "user": user_prompt
        }

    def _format_document(self, doc: Dict, index: int) -> str:
        """Format a document for display"""
        page_content = doc.get("page_content", "")
        if not page_content and isinstance(doc, dict):
            page_content = doc.get("page_content", "")
            
        # Hvis fortsatt ingen innhold
        if not page_content:
            return f"DOKUMENT {index}: [Ingen innhold]"
            
        metadata = doc.get("metadata", {}) if isinstance(doc, dict) else {}
        doc_type = metadata.get("type", "ukjent_type")
        category = metadata.get("category", "")
        title = metadata.get("title", "")
        
        # Bygg innholdsformatering basert på dokumenttype
        header = f"## DOKUMENT {index}: "
        
        if title:
            header += f"{title.upper()}"
            if category:
                header += f" ({category})"
        else:
            header += f"{doc_type.upper()}"
            if category:
                header += f" om {category}"
                
        content = ""
        
        # Formater innhold basert på dokumenttype
        if doc_type == "personal_experience" or doc_type == "personlig_erfaring":
            header = f"## PERSONLIG ERFARING {index}: {title.upper() if title else 'ALS-ERFARING'}"
            timeframe = metadata.get("timeframe", metadata.get("tidsperiode", ""))
            effectiveness = metadata.get("effectiveness", metadata.get("effektivitetsvurdering", ""))
            
            content = f"**Type**: Personlig erfaring\n"
            if timeframe:
                content += f"**Tidsperiode**: {timeframe}\n"
            if effectiveness:
                content += f"**Effektivitet**: {effectiveness}\n"
            content += f"\n{page_content}\n"
            
            # Formatering for tips
            tips = metadata.get("tips", [])
            if tips and isinstance(tips, list) and len(tips) > 0:
                content += "\n**Tips fra erfaringen:**\n"
                for tip in tips:
                    content += f"- {tip}\n"
            
        elif doc_type == "guide":
            steps = metadata.get("steps", [])
            time_estimate = metadata.get("time_estimate", metadata.get("tidsestimat", ""))
            
            content = f"**Type**: Guide\n"
            if time_estimate:
                content += f"**Tidsestimat**: {time_estimate}\n"
            content += f"\n{page_content}\n"
            
            if steps and isinstance(steps, list) and len(steps) > 0:
                content += "\n**Steg-for-steg:**\n"
                for i, step in enumerate(steps):
                    content += f"{i+1}. {step}\n"
                    
        elif doc_type == "tips_og_triks" or doc_type == "tips_and_tricks":
            tips_list = metadata.get("tips_list", [])
            difficulty = metadata.get("difficulty", metadata.get("vanskelighetsgrad", ""))
            
            content = f"**Type**: Tips og triks\n"
            if difficulty:
                content += f"**Vanskelighetsgrad**: {difficulty}\n"
            content += f"\n{page_content}\n"
            
            if tips_list and isinstance(tips_list, list) and len(tips_list) > 0:
                content += "\n**Tips:**\n"
                for tip in tips_list:
                    content += f"- {tip}\n"
                    
        elif doc_type == "forskning" or doc_type == "research":
            source = metadata.get("source", metadata.get("kilde", ""))
            pub_date = metadata.get("publication_date", metadata.get("publiseringsdato", ""))
            
            content = f"**Type**: Forskningsbasert informasjon\n"
            if source:
                content += f"**Kilde**: {source}\n"
            if pub_date:
                content += f"**Publisert**: {pub_date}\n"
            content += f"\n{page_content}\n"
            
        else:
            # Standard formatering for andre dokumenttyper
            content = f"**Type**: {doc_type.capitalize() if doc_type else 'Informasjon'}\n"
            # Inkluder opptil 3 mest relevante metadata-felter
            important_fields = ["author", "date", "summary", "difficulty", "forfatter", "dato", "sammendrag", "vanskelighetsgrad"]
            included = 0
            
            for field in important_fields:
                if field in metadata and included < 3:
                    content += f"**{field.capitalize()}**: {metadata[field]}\n"
                    included += 1
            
            content += f"\n{page_content}\n"
        
        # Legg til relaterte temaer og tags hvis tilgjengelig
        subtopics = metadata.get("subtopics", metadata.get("undertemaer", []))
        tags = metadata.get("tags", [])
        
        if subtopics and isinstance(subtopics, list) and len(subtopics) > 0:
            content += f"\n**Relaterte undertemaer**: {', '.join(subtopics)}\n"
            
        if tags and isinstance(tags, list) and len(tags) > 0:
            content += f"**Tags**: {', '.join(tags)}\n"
            
        # Returner ferdig formatert dokument
        return f"{header}\n{content}"
