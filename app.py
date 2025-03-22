import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from PIL import Image
import io
import time
from datetime import datetime
import traceback
import json
import base64
from io import BytesIO
import uuid
import tempfile
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import re

# Konfigurer miljøvariabler
load_dotenv()

# Sett opp custom CSS for bedre mobilopplevelse
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except Exception as e:
        print(f"Kunne ikke laste CSS-fil: {e}")

# Last inn tilpasset CSS hvis filen eksisterer
try:
    local_css(".streamlit/style.css")
except:
    print("CSS-fil ikke funnet. Bruker standard Streamlit-styling.")

# Sett API-nøkkel fra Streamlit secrets eller miljøvariabler
if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
    os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
    print("Using OpenAI API key from Streamlit secrets")

# Importerer MongoDB-relaterte pakker
try:
    import gridfs
    from bson import ObjectId
except ImportError as e:
    st.error(f"Failed to import MongoDB-related packages: {e}")
    raise

# Import project components
from rag.embeddings import get_embeddings
from rag.vectorstore import get_vectorstore, add_texts_to_vectorstore
from rag.retriever import get_retriever, get_rag_chain
from document_processor.processor import process_text, process_pdf, process_image, process_word_document, process_url, create_metadata_from_contribution
from db.models import Contribution, MAIN_CATEGORIES, CONTENT_TYPES
from db.document_models import Document, Folder
from db.operations import (save_contribution, get_all_contributions, create_vector_search_index, 
                          get_file_by_id, get_contributions_by_category, get_contributions_by_content_type,
                          search_contributions, get_available_tags, save_content_template, get_content_templates)
from db.document_operations import (save_document, get_document_by_id, get_document_content, 
                                   create_folder, get_folder_by_path, get_folders_in_folder, 
                                   get_documents_in_folder, process_document_for_rag, fetch_content_from_url)

# Load environment variables
load_dotenv(find_dotenv())  # Standard .env-fil
load_dotenv(".env_openai", override=True)  # OpenAI-spesifikke variabler

# Debugging for OpenAI API key
openai_api_key = os.environ.get("OPENAI_API_KEY", "")
if openai_api_key:
    masked_key = openai_api_key[:5] + "..." + openai_api_key[-4:]
    print(f"OpenAI API key loaded: {masked_key}")
else:
    print("WARNING: No OpenAI API key found!")

# Globale innstillinger for appen
st.set_page_config(
    page_title="ALS Kunnskapsbase",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.alltidlittsterkere.org',
        'Report a bug': 'https://www.alltidlittsterkere.org/kontakt',
        'About': 'ALS Kunnskapsbank er en plattform for deling av kunnskap om ALS.'
    }
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize session state for RAG components
if "rag_initialized" not in st.session_state:
    st.session_state.rag_initialized = False
    st.session_state.rag_error = None

# Initialize embedding_function in session state
if "embedding_function" not in st.session_state:
    st.session_state.embedding_function = None

# Initialize MongoDB vector index (with error handling)
try:
    create_vector_search_index()
    st.session_state.vector_search_available = True
    print("Vector search index created successfully")
except Exception as e:
    error_message = f"Advarsel: Kunne ikke opprette vektor søkeindeks. Bruker fallback søke metode. Feilmelding: {str(e)}"
    print(error_message)
    st.session_state.rag_error = str(e)
    st.session_state.vector_search_available = False

# Define custom avatars for chat messages
USER_AVATAR = """
<svg width="36px" height="36px" viewBox="0 0 36 36" version="1.1" xmlns="http://www.w3.org/2000/svg">
    <circle cx="18" cy="18" r="18" fill="#4F6CDE"/>
    <g transform="translate(8, 8)">
        <circle cx="10" cy="6" r="5" fill="white" stroke="#2D4073" stroke-width="1.5"/>
        <path d="M1,20 C1,14 5,12 10,12 C15,12 19,14 19,20" fill="white" stroke="#2D4073" stroke-width="1.5" stroke-linecap="round"/>
    </g>
</svg>
"""

ASSISTANT_AVATAR = """
<svg width="36px" height="36px" viewBox="0 0 36 36" version="1.1" xmlns="http://www.w3.org/2000/svg">
    <circle cx="18" cy="18" r="18" fill="#6B9E78"/>
    <g transform="translate(6, 6)">
        <path d="M12,2 C16,2 20,5 20,10 C20,15 16,18 12,18 C8,18 4,15 4,10 C4,5 8,2 12,2" fill="white" stroke="#2D5738" stroke-width="1.5"/>
        <path d="M8,10 C8,8 9,6 12,6 C15,6 16,8 16,10" fill="none" stroke="#2D5738" stroke-width="1.5"/>
        <path d="M10,14 L14,14" stroke="#2D5738" stroke-width="1.5" stroke-linecap="round"/>
        <circle cx="8" cy="8" r="1" fill="#2D5738"/>
        <circle cx="16" cy="8" r="1" fill="#2D5738"/>
    </g>
</svg>
"""

# Convert SVG to base64 encoded data URL
def svg_to_data_url(svg_content):
    svg_bytes = svg_content.encode('utf-8')
    base64_str = base64.b64encode(svg_bytes).decode('utf-8')
    return f"data:image/svg+xml;base64,{base64_str}"

# Create base64 data URLs from SVG content
USER_AVATAR_DATA_URL = svg_to_data_url(USER_AVATAR)
ASSISTANT_AVATAR_DATA_URL = svg_to_data_url(ASSISTANT_AVATAR)

# Legg til CSS styling
st.markdown("""
<style>
    /* Streamlit original CSS overrides - FIX SIDEBAR */
    .st-emotion-cache-16txtl3 {
        color: #ffffff !important;
    }
    .st-emotion-cache-7oyrr6 {
        color: #ffffff !important;
    }
    .st-emotion-cache-16idsys {
        color: #ffffff !important;
    }
    .st-emotion-cache-1e10r2x p, 
    .st-emotion-cache-1e10r2x span, 
    .st-emotion-cache-1e10r2x div,
    .st-emotion-cache-1e10r2x label {
        color: #ffffff !important;
    }
    
    /* Regular styling */
    .reportview-container .main .block-container {
        max-width: 1200px;
        padding: 1rem 1.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 4px;
        font-weight: 500;
        margin-bottom: 8px;
        transition: all 0.3s;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
    .chat-message.user {
        background-color: #4E4E4E;
    }
    .chat-message.assistant {
        background-color: #2D5738;
    }
    
    /* Sidebar styling - med synlig tekst */
    [data-testid="stSidebar"] {
        border-right: 1px solid #4E4E4E;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding: 1rem;
    }
    [data-testid="stSidebar"] h3 {
        border-bottom: 2px solid #6B9E78;
        padding-bottom: 8px;
        margin-bottom: 16px;
        color: #ffffff !important;
        font-weight: 600;
    }
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] div {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] a {
        color: #88A3FF !important;
        text-decoration: none;
        font-weight: 500;
    }
    [data-testid="stSidebar"] a:hover {
        text-decoration: underline;
    }
    [data-testid="stSidebar"] hr {
        margin: 1.5rem 0;
        border: 0;
        height: 1px;
        background-image: linear-gradient(to right, rgba(255, 255, 255, 0), rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
    }
    .copyright {
        font-size: 0.8rem;
        color: #cccccc;
        margin-top: 2rem;
        text-align: center;
    }
    /* Ekstra styling for å sikre at tekst i sidebar er synlig */
    .st-bq, .st-c8, .st-c7, .st-c5, .st-c4, .st-c3, .st-c2, .st-c1, .st-at, .st-as {
        color: #ffffff !important;
    }
    button[kind="secondary"] {
        color: #ffffff !important;
    }
    
    /* Bidrag-grid */
    .contribution-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
        gap: 1rem;
    }
    .card {
        background-color: #f0f0f0;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        cursor: pointer;
    }
    .card.type-personlig_erfaring {
        border-left: 4px solid #4F6CDE;
    }
    .card.type-guide {
        border-left: 4px solid #2D5738;
    }
    .card.type-tips_og_triks {
        border-left: 4px solid #6B9E78;
    }
    .card.type-forskning {
        border-left: 4px solid #8B9467;
    }
    .category-chip {
        font-size: 0.8rem;
        padding: 0.2rem 0.5rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .category-annet {
        background-color: #cccccc;
        color: #666;
    }
    .category-helse {
        background-color: #4F6CDE;
        color: white;
    }
    .category-ernæring {
        background-color: #2D5738;
        color: white;
    }
    .category-hjelpemidler {
        background-color: #6B9E78;
        color: white;
    }
    .category-mobilitet {
        background-color: #8B9467;
        color: white;
    }
    .category-respirasjon {
        background-color: #FFC107;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("ALS Kunnskapsbank")

# Navigasjon - Bruk knapper i stedet for radio
st.sidebar.markdown('<h3 style="color: #ffffff;">Navigasjon</h3>', unsafe_allow_html=True)

# Initialisere session state for navigasjon hvis den ikke finnes
if 'page' not in st.session_state:
    st.session_state.page = "Chat"

# CSS for å gjøre knappene større og penere
st.markdown("""
<style>
    /* Knapp-styling */
    div[data-testid="stButton"] > button {
        width: 100%;
        padding: 12px 8px;
        border-radius: 6px;
        font-weight: 500;
        margin-bottom: 8px;
        transition: all 0.3s;
    }
    
    /* Hovedknapper */
    div[data-testid="stButton"] > button:nth-child(1) {
        background-color: #2D5738;
        color: white;
        font-size: 16px;
        border: none;
    }
    div[data-testid="stButton"] > button:hover {
        background-color: #3B6B49;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        transform: translateY(-1px);
    }
    
    /* Admin knapper */
    .admin-buttons div[data-testid="stButton"] > button {
        background-color: #3A3A3A;
        color: #cccccc;
        font-size: 14px;
        padding: 8px;
        margin-bottom: 4px;
    }
    .admin-buttons div[data-testid="stButton"] > button:hover {
        background-color: #4E4E4E;
    }
    
    /* Aktiv knapp */
    div[data-testid="stButton"] > button.active {
        background-color: #4F6CDE !important;
        box-shadow: 0 0 0 2px rgba(79, 108, 222, 0.5);
    }
    
    /* Markere aktiv side */
    .main-nav {
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Wrapper for hovedmenyen
st.sidebar.markdown('<div class="main-nav">', unsafe_allow_html=True)

# Hovedmeny-knapper med større størrelse og klartekst
btn_chat_style = "background-color: #4F6CDE;" if st.session_state.page == "Chat" else ""
btn_contribute_style = "background-color: #4F6CDE;" if st.session_state.page == "Bidra med kunnskap" else ""
btn_view_style = "background-color: #4F6CDE;" if st.session_state.page == "Se bidrag" else ""

if st.sidebar.button("💬 Chat", use_container_width=True, key="btn_chat", 
                     help="Snakk med chatboten om ALS"):
    st.session_state.page = "Chat"

if st.sidebar.button("➕ Bidra med kunnskap", use_container_width=True, key="btn_contribute", 
                     help="Del din kunnskap og erfaringer"):
    st.session_state.page = "Bidra med kunnskap"

if st.sidebar.button("📚 Se bidrag", use_container_width=True, key="btn_view", 
                     help="Utforsk kunnskapsbasen"):
    st.session_state.page = "Se bidrag"

st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Sidefot - Legg til informasjon om støttegruppe og NAV
with st.sidebar:
    st.markdown('<h3 style="color: #ffffff;">Støttegrupper</h3>', unsafe_allow_html=True)
    st.markdown('<a href="https://www.alltidlittsterkere.org" style="color: #88A3FF;">Alltid litt sterkere</a>', unsafe_allow_html=True)
    st.markdown('<a href="https://www.facebook.com/groups/269144579819107" style="color: #88A3FF;">Facebook-gruppe</a>', unsafe_allow_html=True)
    
    st.markdown('<h3 style="color: #ffffff;">Hjelpemidler</h3>', unsafe_allow_html=True)
    st.markdown('<a href="https://finnhjelpemiddel.nav.no" style="color: #88A3FF;">NAV hjelpemiddeldatabase</a>', unsafe_allow_html=True)
    
    # Admin-funksjoner
    st.markdown('<h3 style="color: #cccccc; font-size: 0.9rem; margin-top: 2rem;">Admin-funksjoner</h3>', unsafe_allow_html=True)
    
    # Wrapper for admin-knapper
    st.markdown('<div class="admin-buttons">', unsafe_allow_html=True)
    
    # Admin-knapper
    btn_docs_style = "background-color: #4F6CDE;" if st.session_state.page == "Dokumenter" else ""
    btn_templates_style = "background-color: #4F6CDE;" if st.session_state.page == "Kunnskapsmaler" else ""
    
    if st.sidebar.button("📄 Dokumenter", use_container_width=True, key="btn_docs", 
                         help="Administrer dokumenter i kunnskapsbasen"):
        st.session_state.page = "Dokumenter"
        
    if st.sidebar.button("🔖 Kunnskapsmaler", use_container_width=True, key="btn_templates", 
                         help="Administrer maler for kunnskapsbidrag"):
        st.session_state.page = "Kunnskapsmaler"
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Copyright footer
    st.markdown('<hr style="margin: 1.5rem 0;">', unsafe_allow_html=True)
    st.markdown('<p style="color: #cccccc; font-size: 0.8rem; text-align: center;"> 2025 ALS Kunnskapsbank</p>', unsafe_allow_html=True)
    st.markdown('<p style="color: #cccccc; font-size: 0.8rem; text-align: center;">Drevet av AI og samfunnskunnskap</p>', unsafe_allow_html=True)
    st.markdown('<p style="color: #cccccc; font-size: 0.8rem; text-align: center;">Utviklet av Glenn Råna</p>', unsafe_allow_html=True)

# Hent side fra session state
page = st.session_state.page

if page == "Chat":
    st.header("ALS Kunnskapsassistent")
    
    # Mobiloptimalisert introduksjon - kompakt på små skjermer
    with st.container():
        st.markdown("""
        <div class="chat-intro">
            <p>Hei! Jeg er ALS-kunnskapsassistenten. Jeg kan hjelpe deg med informasjon om ALS, hjelpemidler, 
            forskning, ernæring og mye mer.</p>
            <p>Still meg gjerne et spørsmål!</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Initialize components if not already done
    if not st.session_state.rag_initialized:
        with st.spinner("Initialiserer kunnskapsbasen... Vennligst vent."):
            embedding_function, error_msg = initialize_rag()
            
            if error_msg:
                st.error(f"Kunne ikke initialisere kunnskapsbasen: {error_msg}")
                st.stop()  # Bruk st.stop() i stedet for return
            
            # Get RAG components
            try:
                vectorstore = get_vectorstore(embedding_function)
                retriever = get_retriever(vectorstore)
                st.session_state.rag_chain = get_rag_chain(retriever)
                st.session_state.rag_initialized = True
            except Exception as e:
                error_msg = str(e)
                st.error(f"Kunne ikke initialisere RAG-komponenter: {error_msg}")
                st.stop()  # Bruk st.stop() i stedet for return
    
    # Initialize chat history if not already done
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        avatar = USER_AVATAR_DATA_URL if message["role"] == "user" else ASSISTANT_AVATAR_DATA_URL
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
    
    # Accept user input
    if prompt := st.chat_input("Skriv inn ditt spørsmål...", key="mobile_optimized_input"):
        # Display user message in chat message container
        st.chat_message("user", avatar=USER_AVATAR_DATA_URL).markdown(prompt)
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate and display assistant response
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR_DATA_URL):
            # Initialize a placeholder for streaming response
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Get streaming response from RAG chain
                if st.session_state.rag_chain:
                    with st.spinner("Henter informasjon..."):
                        response = st.session_state.rag_chain.invoke(prompt)
                        
                        # Display full response
                        full_response = response
                        message_placeholder.markdown(full_response)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    message_placeholder.error("RAG-systemet er ikke initialisert. Vennligst prøv igjen senere.")
            except Exception as e:
                error_message = f"Beklager, jeg kunne ikke svare på spørsmålet ditt akkurat nå: {str(e)}"
                message_placeholder.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                print(f"Error generating response: {traceback.format_exc()}")

elif page == "Bidra med kunnskap":
    st.header("Del dine erfaringer")
    st.write("Din kunnskap kan hjelpe andre som lever med ALS. Del dine erfaringer, tips eller innsikt om hjelpemidler, medisiner, symptomhåndtering eller andre aspekter ved å leve med ALS.")
    
    # Oppretter kolonner for et bedre layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Velg innholdstype og kategori
        selected_content_type = st.selectbox(
            "Velg type innhold:", 
            options=CONTENT_TYPES,
            help="Velg hvilken type innhold du vil dele. Dette hjelper oss å organisere kunnskapsbasen."
        )
        
        selected_category = st.selectbox(
            "Velg hovedkategori:", 
            options=MAIN_CATEGORIES,
            help="Velg en kategori som best beskriver emnet for ditt bidrag."
        )
        
        # Velg undertemaer/tags
        sub_categories = st.multiselect(
            "Velg undertemaer (valgfritt):", 
            options=MAIN_CATEGORIES,
            help="Velg eventuelle undertemaer som er relevante for ditt bidrag."
        )
        
        # Fri-tekst for tags
        custom_tags = st.text_input(
            "Legg til stikkord (adskilt med komma):", 
            help="Legg til relevante stikkord som gjør det lettere å finne ditt bidrag. F.eks. 'matvansker, spise, bestikk'"
        )
        tags = [tag.strip() for tag in custom_tags.split(",")] if custom_tags else []
        
        # Tittel for bidraget
        title = st.text_input(
            "Tittel på bidraget:", 
            help="Gi bidraget ditt en kort og beskrivende tittel."
        )
    
    with col2:
        # Vanskelighetsgrad
        difficulty_level = st.radio(
            "Vanskelighetsgrad:",
            options=["lett", "middels", "avansert"],
            help="Angi hvor utfordrende det er å implementere eller bruke denne løsningen."
        )
        
        # Sammendrag
        summary = st.text_area(
            "Kort sammendrag:",
            height=80,
            help="Gi en kort oppsummering av bidraget ditt (1-2 setninger)."
        )
        
        contributor_name = st.text_input(
            "Ditt navn (valgfritt):", 
            help="Du kan bidra anonymt hvis du ønsker det."
        )
        
        # Relaterte lenker
        related_links_text = st.text_area(
            "Relaterte lenker (én per linje, format: lenke|beskrivelse):",
            height=80,
            help="Legg til relevante nettsider eller ressurser. F.eks. 'https://nav.no/hjelpemidler|NAVs side om hjelpemidler'"
        )
        
        # Parser relaterte lenker
        related_links = []
        if related_links_text:
            for line in related_links_text.split("\n"):
                if "|" in line:
                    url, desc = line.split("|", 1)
                    related_links.append({"url": url.strip(), "description": desc.strip()})
                elif line.strip():
                    related_links.append({"url": line.strip(), "description": line.strip()})
    
    # Hovedinnhold basert på type bidrag
    st.markdown("### Hovedinnhold")
    
    # Standardfelt
    problem = st.text_area(
        "Beskriv problemet eller symptomet:", 
        height=100, 
        help="Forklar utfordringen eller symptomet så detaljert som mulig. F.eks. 'Vanskeligheter med å holde bestikk ved måltider' eller 'Problemer med å snu seg i sengen'."
    )
    
    aids_used = st.text_area(
        "Hvilke hjelpemidler eller tilpasninger har du brukt for å håndtere dette?", 
        height=100,
        help="Beskriv løsninger, strategier eller hjelpemidler som har fungert for deg. F.eks. 'Spesialbestikk med tykkere håndtak' eller 'Sengebøyle for støtte ved forflytning'."
    )
    
    medicine_info = st.text_area(
        "Informasjon om medisin (valgfritt):", 
        height=100,
        help="Hvis relevant, inkluder informasjon om medisiner eller behandlinger. Merk at dette ikke er medisinske råd, men deling av erfaringer."
    )
    
    # Strukturert innhold basert på innholdstype
    structured_content = {}
    
    # Hvis det er personlig erfaring, spør om ekstra detaljer
    if selected_content_type == "personlig_erfaring":
        structured_content["periode"] = st.text_input(
            "Tidsperiode for erfaringen:",
            help="Når hadde du denne erfaringen? F.eks. '2022-2023' eller 'De første 6 månedene etter diagnose'"
        )
        
        structured_content["effektivitet"] = st.select_slider(
            "Hvor effektiv var løsningen?",
            options=["Ikke effektiv", "Litt effektiv", "Moderat effektiv", "Svært effektiv", "Revolusjonerende"]
        )
    
    # Hvis det er en guide, spør om steg-for-steg instruksjoner
    elif selected_content_type == "guide":
        steps_text = st.text_area(
            "Steg-for-steg instruksjoner (ett steg per linje):",
            height=150,
            help="Beskriv hvert steg på en ny linje. Vær så spesifikk som mulig."
        )
        structured_content["steps"] = [step.strip() for step in steps_text.split("\n") if step.strip()]
        
        structured_content["tid_påkrevd"] = st.text_input(
            "Tid påkrevd:",
            help="Hvor lang tid tar det å implementere denne guiden? F.eks. '15 minutter' eller '1-2 dager'"
        )
    
    # For tips og triks
    elif selected_content_type == "tips_og_triks":
        tips_text = st.text_area(
            "Tips og triks (ett tips per linje):",
            height=150,
            help="Legg inn ett tips per linje. Hold det kort og konkret."
        )
        structured_content["tips"] = [tip.strip() for tip in tips_text.split("\n") if tip.strip()]
    
    # For forskning
    elif selected_content_type == "forskning":
        structured_content["forskningskilde"] = st.text_input(
            "Kilde til forskningen:",
            help="Oppgi kilden til forskningen, f.eks. en vitenskapelig artikkel eller nettside"
        )
        structured_content["publiseringsdato"] = st.text_input(
            "Publiseringsdato:",
            help="Når ble forskningen publisert? F.eks. 'Mars 2023'"
        )
    
    # Fil-opplasting
    st.markdown("### Dokumentasjon av hjelpemidler")
    st.markdown("""
    Dokumentasjon kan gjøre bidraget ditt enda mer verdifullt. Du kan legge til:
    - **Bilder** av hjelpemidler eller tilpasninger
    - **Tekst** med ytterligere detaljer eller instruksjoner
    - **PDF** med brosjyrer eller veiledninger
    """)
    
    upload_type = st.radio(
        "Opplastningstype", 
        ["Ingen vedlegg", "Bilde av hjelpemidler", "Tekstdokument med instruksjoner", "PDF med ytterligere informasjon"],
        help="Velg hva slags dokumentasjon du vil legge ved bidraget ditt. Bilder av hjelpemidler er spesielt nyttige."
    )
    
    uploaded_file = None
    file_content = None
    file_type = None
    
    if upload_type != "Ingen vedlegg":
        file_types = {
            "Bilde av hjelpemidler": ["png", "jpg", "jpeg"],
            "Tekstdokument med instruksjoner": ["txt", "md"],
            "PDF med ytterligere informasjon": ["pdf"]
        }
        
        current_type = upload_type
        allowed_types = file_types.get(current_type, ["txt", "pdf", "png", "jpg", "jpeg"])
        
        uploaded_file = st.file_uploader(
            f"Last opp {current_type.lower()}", 
            type=allowed_types,
            help="Velg en fil fra din datamaskin. Filen vil bli lagret sammen med bidraget ditt og kan vises til andre ALS-pasienter."
        )

    if st.button("Send inn bidrag"):
        if problem and aids_used and title and selected_category and selected_content_type:
            # Process uploaded file if any
            processed_chunks = []
            
            if uploaded_file:
                file_content = uploaded_file.read()
                
                # Determine file type
                if upload_type == "Bilde av hjelpemidler":
                    file_type = "Bilde"
                elif upload_type == "Tekstdokument med instruksjoner":
                    file_type = "Tekst"
                elif upload_type == "PDF med ytterligere informasjon":
                    file_type = "PDF"
                
                # Process file based on type
                try:
                    if file_type == "Tekst":
                        processed_chunks = process_text(file_content.decode("utf-8"))
                    elif file_type == "PDF":
                        processed_chunks = process_pdf(file_content)
                    elif file_type == "Bilde":
                        image = Image.open(io.BytesIO(file_content))
                        processed_chunks = process_image(image)
                except Exception as e:
                    st.error(f"Feil ved prosessering av fil: {e}")
                    processed_chunks = []
                
                # Create contribution
                contribution = Contribution(
                    problem=problem,
                    aids_used=aids_used,
                    medicine_info=medicine_info,
                    contributor_name=contributor_name if contributor_name else "Anonym",
                    file_name=uploaded_file.name,
                    file_type=file_type,
                    file_content=file_content,
                    timestamp=datetime.now(),
                    title=title,
                    category=selected_category,
                    sub_categories=sub_categories,
                    content_type=selected_content_type,
                    tags=tags,
                    summary=summary,
                    structured_content=structured_content,
                    difficulty_level=difficulty_level,
                    related_links=related_links
                )
                
                # Save to database
                contribution_id = save_contribution(contribution)
                
                # Create combined text for embedding
                combined_text = f"Tittel: {title}\nKategori: {selected_category}\nProblemnivå: {difficulty_level}\nProblem: {problem}\nHjelpemidler: {aids_used}"
                if medicine_info:
                    combined_text += f"\nMedisin-info: {medicine_info}"
                if summary:
                    combined_text += f"\nSammendrag: {summary}"
                if tags:
                    combined_text += f"\nStikkord: {', '.join(tags)}"
                
                # Add contributor name if provided
                if contributor_name:
                    combined_text += f"\nBidragsyter: {contributor_name}"
                
                # Create metadata
                metadata = create_metadata_from_contribution(
                    contribution_id=contribution_id,
                    problem=problem,
                    aids_used=aids_used,
                    file_type=file_type,
                    file_name=uploaded_file.name if uploaded_file else None,
                    title=title,
                    category=selected_category,
                    content_type=selected_content_type,
                    tags=tags
                )
                
                # Add the processed chunks as well if any
                texts = [combined_text] + processed_chunks
                metadatas = [metadata] * len(texts)
                
                # Add to vector store
                try:
                    if 'embedding_function' in st.session_state:
                        add_texts_to_vectorstore(texts, metadatas, st.session_state.embedding_function)
                    st.success("Takk for ditt bidrag! Din kunnskap vil hjelpe andre med ALS.")
                    st.balloons()
                except Exception as e:
                    st.error(f"Feil ved lagring i vektorbasen: {e}")
                    st.warning("Ditt bidrag ble lagret, men kan kanskje ikke søkes ennå.")
            
            else:
                # Create contribution without file
                contribution = Contribution(
                    problem=problem,
                    aids_used=aids_used,
                    medicine_info=medicine_info,
                    contributor_name=contributor_name if contributor_name else "Anonym",
                    timestamp=datetime.now(),
                    title=title,
                    category=selected_category,
                    sub_categories=sub_categories,
                    content_type=selected_content_type,
                    tags=tags,
                    summary=summary,
                    structured_content=structured_content,
                    difficulty_level=difficulty_level,
                    related_links=related_links
                )
                
                # Save to database
                contribution_id = save_contribution(contribution)
                
                # Create combined text for embedding
                combined_text = f"Tittel: {title}\nKategori: {selected_category}\nProblemnivå: {difficulty_level}\nProblem: {problem}\nHjelpemidler: {aids_used}"
                if medicine_info:
                    combined_text += f"\nMedisin-info: {medicine_info}"
                if summary:
                    combined_text += f"\nSammendrag: {summary}"
                if tags:
                    combined_text += f"\nStikkord: {', '.join(tags)}"
                
                # Add contributor name if provided
                if contributor_name:
                    combined_text += f"\nBidragsyter: {contributor_name}"
                
                # Create metadata
                metadata = create_metadata_from_contribution(
                    contribution_id=contribution_id,
                    problem=problem,
                    aids_used=aids_used,
                    title=title,
                    category=selected_category,
                    content_type=selected_content_type,
                    tags=tags
                )
                
                # Add to vector store
                try:
                    if 'embedding_function' in st.session_state:
                        add_texts_to_vectorstore([combined_text], [metadata], st.session_state.embedding_function)
                    st.success("Takk for ditt bidrag! Din kunnskap vil hjelpe andre med ALS.")
                    st.balloons()
                except Exception as e:
                    st.error(f"Feil ved lagring i vektorbasen: {e}")
                    st.warning("Ditt bidrag ble lagret, men kan kanskje ikke søkes ennå.")
        else:
            st.error("Vennligst fyll ut alle påkrevde felt: tittel, kategori, innholdstype, problem og hjelpemidler.")

elif page == "Se bidrag":
    st.header("Se delt kunnskap")
    st.write("Utforsk bidrag fra ALS-samfunnet i Norge.")
    
    # Kolonner for filter og søk
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### Filtrer")
        
        # Filtrer på kategori
        category_filter = st.selectbox(
            "Kategori:",
            options=["Alle"] + MAIN_CATEGORIES,
        )
        
        # Filtrer på innholdstype
        content_type_filter = st.selectbox(
            "Type innhold:",
            options=["Alle"] + CONTENT_TYPES,
        )
        
        # Filtrer på vanskelighetsgrad
        difficulty_filter = st.selectbox(
            "Vanskelighetsgrad:",
            options=["Alle", "lett", "middels", "avansert"],
        )
    
    with col2:
        st.markdown("### Søk")
        search_term = st.text_input("Søk etter spesifikke problemer, hjelpemidler eller stikkord:")
        
        # Avansert søk (f.eks. MongoDB-stil syntaks)
        advanced_search = st.checkbox("Avansert søk")
        advanced_query = {}
        
        if advanced_search:
            advanced_query_text = st.text_area(
                "MongoDB spørring (JSON format):",
                value='{"tags": {"$in": ["spising"]}}',
                help="Bruk MongoDB spørringssyntaks for å filtrere bidrag. F.eks. {'tags': {'$in': ['spising']}}"
            )
            try:
                advanced_query = json.loads(advanced_query_text)
            except Exception as e:
                st.error(f"Ugyldig JSON: {e}")
    
    # Get contributions based on filters
    with st.spinner("Laster bidrag..."):
        # Base query
        if advanced_search and advanced_query:
            filtered_contributions = search_contributions(advanced_query)
        else:
            # Start with all contributions
            filtered_contributions = get_all_contributions()
            
            # Apply filters
            if category_filter != "Alle":
                filtered_contributions = [c for c in filtered_contributions if c.category == category_filter]
            
            if content_type_filter != "Alle":
                filtered_contributions = [c for c in filtered_contributions if c.content_type == content_type_filter]
            
            if difficulty_filter != "Alle":
                filtered_contributions = [c for c in filtered_contributions if c.difficulty_level == difficulty_filter]
            
            # Apply search term
            if search_term:
                search_term_lower = search_term.lower()
                
                # Search in problem, aids_used, title, summary and tags
                def matches_search(contribution):
                    # Check if the contribution is None
                    if contribution is None:
                        return False

                    if hasattr(contribution, 'problem') and contribution.problem and search_term_lower in contribution.problem.lower():
                        return True
                    if hasattr(contribution, 'aids_used') and contribution.aids_used and search_term_lower in contribution.aids_used.lower():
                        return True
                    if hasattr(contribution, 'title') and contribution.title and search_term_lower in contribution.title.lower():
                        return True
                    if hasattr(contribution, 'summary') and contribution.summary and search_term_lower in contribution.summary.lower():
                        return True
                    if hasattr(contribution, 'tags') and contribution.tags:
                        for tag in contribution.tags:
                            if tag and search_term_lower in tag.lower():
                                return True
                    return False
                
                filtered_contributions = [c for c in filtered_contributions if matches_search(c)]
    
    # Display contributions
    if filtered_contributions:
        st.markdown(f"**{len(filtered_contributions)} bidrag funnet**")
        
        # Sorter etter dato
        filtered_contributions.sort(key=lambda x: x.timestamp if hasattr(x, 'timestamp') else datetime.min, reverse=True)
        
        # Vis bidrag i en responsiv grid
        st.markdown('<div class="contribution-grid">', unsafe_allow_html=True)
        
        for contribution in filtered_contributions:
            # Lag en fin kortvisning for bidraget
            title_display = getattr(contribution, 'title', 'Bidrag') or f"Bidrag om {contribution.problem[:20]}..."
            
            # Kategorimerking med farge
            category = getattr(contribution, 'category', 'annet')
            content_type = getattr(contribution, 'content_type', 'annet')
            
            # Generer HTML for kortet med responsiv styling
            card_html = f"""
            <div class="card type-{content_type}" onclick="selectContribution('{contribution.id}')">
                <h3>{title_display}</h3>
                <div class="category-chip category-{category}">{category}</div>
                <p>{getattr(contribution, 'summary', '')[:100]}...</p>
                <div style="font-size: 0.8rem; color: #666;">
                    {getattr(contribution, 'contributor_name', 'Anonym')} | 
                    {getattr(contribution, 'timestamp', '').strftime('%d.%m.%Y') if hasattr(contribution, 'timestamp') and contribution.timestamp else ''}
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Legg til JavaScript for å håndtere klikk på bidrag
        js = """
        <script>
        function selectContribution(id) {
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                value: id
            }, '*');
        }
        </script>
        """
        st.markdown(js, unsafe_allow_html=True)
        
        # Håndtere valg av bidrag via Streamlit-callback
        if hasattr(st.session_state, 'selected_contribution'):
            selected_id = st.session_state.selected_contribution
            selected = next((c for c in filtered_contributions if c.id == selected_id), None)
            
            if selected:
                st.markdown("---")
                st.markdown(f"# {getattr(selected, 'title', 'Detaljert visning')}")
                
                # Metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**Kategori:** {getattr(selected, 'category', 'Ikke kategorisert')}")
                    if hasattr(selected, 'sub_categories') and selected.sub_categories:
                        st.markdown(f"**Undertemaer:** {', '.join(selected.sub_categories)}")
                
                with col2:
                    st.markdown(f"**Type innhold:** {getattr(selected, 'content_type', 'Ikke spesifisert')}")
                    st.markdown(f"**Vanskelighetsgrad:** {getattr(selected, 'difficulty_level', 'Ikke spesifisert')}")
                
                with col3:
                    st.markdown(f"**Dato:** {selected.timestamp.strftime('%d.%m.%Y')}")
                    st.markdown(f"**Bidragsyter:** {selected.contributor_name}")
                
                # Tags
                if hasattr(selected, 'tags') and selected.tags:
                    st.markdown(f"**Stikkord:** {', '.join(selected.tags)}")
                
                # Sammendrag
                if hasattr(selected, 'summary') and selected.summary:
                    st.markdown("### Sammendrag")
                    st.markdown(selected.summary)
                
                # Hovedinnhold
                st.markdown("### Problem")
                st.markdown(selected.problem)
                
                st.markdown("### Løsning")
                st.markdown(selected.aids_used)
                
                if hasattr(selected, 'medicine_info') and selected.medicine_info:
                    st.markdown("### Medisinsk informasjon")
                    st.markdown(selected.medicine_info)
                
                # Strukturert innhold
                if hasattr(selected, 'structured_content') and selected.structured_content:
                    st.markdown("### Ytterligere detaljer")
                    
                    content_type = getattr(selected, 'content_type', None)
                    if content_type == "personlig_erfaring":
                        if "periode" in selected.structured_content:
                            st.markdown(f"**Periode:** {selected.structured_content['periode']}")
                        if "effektivitet" in selected.structured_content:
                            st.markdown(f"**Effektivitet:** {selected.structured_content['effektivitet']}")
                    
                    elif content_type == "guide":
                        if "steps" in selected.structured_content:
                            st.markdown("#### Steg-for-steg guide")
                            for i, step in enumerate(selected.structured_content['steps']):
                                st.markdown(f"{i+1}. {step}")
                        if "tid_påkrevd" in selected.structured_content:
                            st.markdown(f"**Tid påkrevd:** {selected.structured_content['tid_påkrevd']}")
                    
                    elif content_type == "tips_og_triks":
                        if "tips" in selected.structured_content:
                            st.markdown("#### Tips")
                            for tip in selected.structured_content['tips']:
                                st.markdown(f"- {tip}")
                    
                    elif content_type == "forskning":
                        if "forskningskilde" in selected.structured_content:
                            st.markdown(f"**Kilde:** {selected.structured_content['forskningskilde']}")
                        if "publiseringsdato" in selected.structured_content:
                            st.markdown(f"**Publisert:** {selected.structured_content['publiseringsdato']}")
                
                # Relaterte lenker
                if hasattr(selected, 'related_links') and selected.related_links:
                    st.markdown("### Relaterte lenker")
                    for link in selected.related_links:
                        url = link.get('url', '')
                        desc = link.get('description', url)
                        st.markdown(f"- [{desc}]({url})")
                
                # Vedlegg
                if hasattr(selected, 'file_name') and selected.file_name and hasattr(selected, 'file_type') and selected.file_type:
                    st.markdown(f"### Vedlegg: {selected.file_name}")
                    
                    # Display file content if possible
                    try:
                        if selected.file_type == "Bilde" and selected.file_content:
                            st.image(Image.open(io.BytesIO(selected.file_content)))
                        elif selected.file_type == "Tekst" and selected.file_content:
                            st.code(selected.file_content.decode("utf-8"))
                        elif selected.file_type == "PDF":
                            st.write("PDF-fil tilknyttet (forhåndsvisning ikke tilgjengelig)")
                    except Exception as e:
                        st.error(f"Kunne ikke vise fil: {e}")
                
                # Tilbake-knapp
                if st.button("Tilbake til oversikt"):
                    del st.session_state.selected_contribution
                    st.experimental_rerun()
    else:
        st.info("Ingen bidrag funnet. Vær den første til å dele din kunnskap!")

elif page == "Dokumenter":
    st.header("Dokumenter")
    st.write("Her kan du laste opp, organisere og dele dokumenter som kan være nyttige for ALS-pasienter og deres pårørende.")
    
    # Opprette kolonner for bedre layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Mappestruktur")
        
        # Initialisere session state for nåværende sti hvis den ikke finnes
        if 'current_folder_path' not in st.session_state:
            st.session_state.current_folder_path = "/"
        
        # Vis mappehierarki
        current_path = st.session_state.current_folder_path
        folder_parts = [p for p in current_path.split("/") if p]
        
        # Navigasjon
        if current_path != "/":
            if st.button("⬆️ Opp et nivå"):
                # Gå opp ett nivå
                parent_path = "/" + "/".join(folder_parts[:-1])
                if not parent_path.endswith("/"):
                    parent_path += "/"
                st.session_state.current_folder_path = parent_path
                st.rerun()
        
        # Vis gjeldende sti
        st.markdown(f"**Gjeldende sti:** {current_path}")
        
        # Vis mapper i gjeldende mappe
        st.markdown("#### Mapper:")
        folders = get_folders_in_folder(current_path)
        
        for folder in folders:
            col_folder1, col_folder2 = st.columns([3, 1])
            with col_folder1:
                if st.button(f"📁 {folder.name}", key=f"folder_{folder.id}"):
                    # Naviger til mappen
                    st.session_state.current_folder_path = folder.path
                    st.rerun()
            
        # Opprette ny mappe
        st.markdown("#### Opprett ny mappe")
        new_folder_name = st.text_input("Mappenavn", key="new_folder_name")
        
        if st.button("Opprett mappe", key="create_folder_button"):
            if new_folder_name:
                # Lag ny mappesti
                new_path = os.path.join(current_path, new_folder_name)
                if not new_path.endswith("/"):
                    new_path += "/"
                
                # Sjekk om mappen allerede eksisterer
                existing_folder = get_folder_by_path(new_path)
                
                if existing_folder:
                    st.error(f"Mappen '{new_folder_name}' eksisterer allerede.")
                else:
                    # Opprett ny mappe
                    folder = Folder(
                        name=new_folder_name,
                        path=new_path,
                        parent_path=current_path
                    )
                    
                    create_folder(folder)
                    st.success(f"Mappen '{new_folder_name}' ble opprettet.")
                    st.rerun()
    
    with col2:
        st.markdown("### Dokumenter")
        
        # Vis dokumenter i gjeldende mappe
        documents = get_documents_in_folder(st.session_state.current_folder_path)
        
        if documents:
            st.markdown(f"**{len(documents)} dokumenter funnet i denne mappen**")
            
            for doc in documents:
                with st.expander(f"{doc.name} ({doc.type})"):
                    st.write(f"**Type:** {doc.content_type}")
                    st.write(f"**Størrelse:** {doc.size} bytes")
                    st.write(f"**Lastet opp:** {doc.created_at.strftime('%Y-%m-%d %H:%M')}")
                    
                    if doc.tags:
                        st.write(f"**Tags:** {', '.join(doc.tags)}")
                    
                    if doc.source_url:
                        st.write(f"**Kilde:** {doc.source_url}")
                    
                    # Handlinger
                    col_action1, col_action2 = st.columns(2)
                    
                    with col_action1:
                        if st.button("Last ned", key=f"download_{doc.id}"):
                            content = get_document_content(doc.id)
                            
                            if content:
                                # Konverter til base64 for nedlasting
                                import base64
                                b64 = base64.b64encode(content).decode()
                                href = f'<a href="data:{doc.content_type};base64,{b64}" download="{doc.name}">Klikk her for å laste ned</a>'
                                st.markdown(href, unsafe_allow_html=True)
                    
                    with col_action2:
                        if not doc.processed:
                            if st.button("Legg til i kunnskapsbasen", key=f"process_{doc.id}"):
                                with st.spinner("Prosesserer dokument..."):
                                    success = process_document_for_rag(doc.id)
                                    
                                    if success:
                                        st.success("Dokumentet ble lagt til i kunnskapsbasen.")
                                    else:
                                        st.error("Kunne ikke prosessere dokumentet.")
                        else:
                            st.info("Dokumentet er allerede i kunnskapsbasen.")
        else:
            st.info("Ingen dokumenter funnet i denne mappen.")
        
        # Dokument-opplasting
        st.markdown("### Last opp dokument")
        
        # Tags
        doc_tags = st.text_input(
            "Tags (kommaseparert)",
            help="Legg til relevante nøkkelord for å gjøre dokumentet lettere å finne"
        )
        
        uploaded_file = st.file_uploader(
            "Velg fil", 
            type=["pdf", "docx", "txt", "jpg", "jpeg", "png"],
            help="Støttede filtyper: PDF, Word (docx), tekst, bilder (jpg, png)"
        )

        if uploaded_file:
            # Prosesser opplastet fil
            file_name = uploaded_file.name
            file_content = uploaded_file.read()
            file_type = uploaded_file.type
            file_size = len(file_content)
            
            # Konverter tags til liste
            tag_list = [tag.strip() for tag in doc_tags.split(",")] if doc_tags else []
            
            if st.button("Last opp dokument", key="upload_doc_button"):
                # Opprette dokument
                document = Document(
                    name=file_name,
                    type=file_type,
                    size=file_size,
                    folder_path=st.session_state.current_folder_path,
                    content_type=file_type,
                    tags=tag_list
                )
                
                # Lagre dokument
                doc_id = save_document(document, file_content)
                
                if doc_id:
                    st.success(f"Dokumentet '{file_name}' ble lastet opp.")
                    
                    # Spør om å legge til i kunnskapsbasen
                    if st.checkbox("Legg til i kunnskapsbasen", value=True):
                        with st.spinner("Prosesserer dokument..."):
                            success = process_document_for_rag(doc_id)
                            
                            if success:
                                st.success("Dokumentet ble lagt til i kunnskapsbasen.")
                            else:
                                st.error("Kunne ikke prosessere dokumentet for kunnskapsbasen.")
                    
                    # Rerun for å oppdatere visningen
                    st.rerun()
        
        # Web Scraping
        st.markdown("### Hent innhold fra nettside")
        url = st.text_input("URL", help="Angi URL til nettsiden du vil hente innhold fra")
        
        # Tags for web content
        web_tags = st.text_input(
            "Tags (kommaseparert)",
            key="web_tags",
            help="Legg til relevante nøkkelord for å gjøre websideinnholdet lettere å finne"
        )
        
        if st.button("Hent innhold", key="fetch_web_button"):
            if url:
                with st.spinner(f"Henter innhold fra {url}..."):
                    # Konverter tags til liste
                    web_tag_list = [tag.strip() for tag in web_tags.split(",")] if web_tags else []
                    
                    # Hent innhold fra URL
                    doc_id = fetch_content_from_url(url, st.session_state.current_folder_path, web_tag_list)
                    
                    if doc_id:
                        st.success(f"Innholdet fra {url} ble hentet og lagret.")
                        
                        # Rerun for å oppdatere visningen
                        st.rerun()
                    else:
                        st.error(f"Kunne ikke hente innhold fra {url}.")
            else:
                st.error("Vennligst angi en URL.")

elif page == "Kunnskapsmaler":
    st.title("Kunnskapsmaler")
    
    # Her kan brukeren administrere og opprette maler
    st.write("På denne siden kan du administrere maler for å forenkle innlegging av kunnskap.")
    
    # Kolonner for malredigering og forhåndsvisning
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Opprett ny mal")
        
        # Velg kategorier
        template_category = st.selectbox(
            "Kategori denne malen gjelder for:", 
            options=MAIN_CATEGORIES
        )
        
        template_content_type = st.selectbox(
            "Type innhold denne malen gjelder for:", 
            options=CONTENT_TYPES
        )
        
        # Mal-innhold
        template_name = st.text_input(
            "Navn på malen:", 
            help="Gi malen et beskrivende navn, f.eks. 'Guide for tilpasset bestikk'"
        )
        
        template_description = st.text_area(
            "Beskrivelse av malen:", 
            help="Forklar hva denne malen skal brukes til og når den er relevant"
        )
        
        # Struktur for malen
        st.markdown("#### Feltstruktur")
        st.markdown("Definer hvilke felt som skal inkluderes i malen og hvordan de skal presenteres.")
        
        # Strukturert innhold
        template_fields = st.text_area(
            "Feltdefinisjoner (JSON-format):", 
            value="""
{
  "problem_template": "Beskriv spesifikt hvilken type [kategori]-problem denne løsningen adresserer.",
  "solution_steps": [
    "Steg 1: ",
    "Steg 2: ",
    "Steg 3: "
  ],
  "effectiveness_rating": "Hvordan vil du rangere effektiviteten av denne løsningen?"
}
            """,
            height=300,
            help="Definer feltene i malen i JSON-format. Dette vil brukes som utgangspunkt for nye bidrag."
        )
        
        # Lagre malen
        if st.button("Lagre mal"):
            try:
                template_data = {
                    "name": template_name,
                    "category": template_category,
                    "content_type": template_content_type,
                    "description": template_description,
                    "fields": json.loads(template_fields)
                }
                
                template_id = save_content_template(template_data)
                st.success(f"Malen '{template_name}' ble lagret med ID: {template_id}")
            except Exception as e:
                st.error(f"Feil ved lagring av mal: {e}")
    
    with col2:
        st.markdown("### Eksisterende maler")
        
        # Hent alle maler
        templates = get_content_templates()
        
        if templates:
            for template in templates:
                with st.expander(f"{template.get('name', 'Mal')} ({template.get('category', 'Alle kategorier')})"):
                    st.markdown(f"**Kategori:** {template.get('category', 'Ikke spesifisert')}")
                    st.markdown(f"**Innholdstype:** {template.get('content_type', 'Ikke spesifisert')}")
                    st.markdown(f"**Beskrivelse:** {template.get('description', 'Ingen beskrivelse')}")
                    
                    st.markdown("**Feltstruktur:**")
                    st.json(template.get('fields', {}))
                    
                    # Knapper for redigering/sletting
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Rediger", key=f"edit_{template.get('id', '')}"):
                            # TODO: Implementer redigering
                            st.info("Redigering av maler kommer i neste versjon.")
                    with col2:
                        if st.button("Slett", key=f"delete_{template.get('id', '')}"):
                            # TODO: Implementer sletting
                            st.info("Sletting av maler kommer i neste versjon.")
        else:
            st.info("Ingen maler er definert ennå. Opprett en ny mal.")

if __name__ == "__main__":
    # Create necessary directories if they don't exist
    os.makedirs("data/documents", exist_ok=True)
