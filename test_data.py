"""
Test data script to populate the ALS Knowledge database with sample contributions.
Run this script to add test data for the RAG system to use.
"""

import os
from dotenv import load_dotenv
from db.models import Contribution
from db.operations import save_contribution
from rag.embeddings import get_embeddings
from rag.vectorstore import add_texts_to_vectorstore
from document_processor.processor import create_metadata_from_contribution

# Load environment variables
load_dotenv()

# Sample contributions data
sample_contributions = [
    {
        "problem": "Problemer med å spise og svelge mat",
        "aids_used": "Vi har hatt god erfaring med slurry-mat, smoothies, og spesialskjeer med tykkere håndtak. Consistaid pulver kan tilsettes i væsker for å gjøre dem tykkere og lettere å svelge. Man kan også bruke spisebeskytter for å unngå søl.",
        "medicine_info": "Ingen spesifikke medisiner, men det er viktig å opprettholde næringsinntak. Noen bruker ernæringsshaker.",
        "contributor_name": "Kari Normann"
    },
    {
        "problem": "Vanskeligheter med å kommunisere og tale",
        "aids_used": "Vi bruker en kombinasjon av nettbrett med taleprogram (GridPlayer app), øyestyrt kommunikasjonsutstyr fra Nav, og enkle bildekort for daglige behov. Tobii Dynavox har vært en stor hjelp.",
        "medicine_info": "",
        "contributor_name": "Per Hansen"
    },
    {
        "problem": "Utfordringer med mobilitet og bevegelse",
        "aids_used": "Elektrisk rullestol med spesialtilpasset sete, personløfter, tilpasset seng med elektrisk hev/senk funksjon, og sklimatter for forflytning. NAV Hjelpemiddelsentralen har hjulpet med å tilpasse disse hjelpemidlene.",
        "medicine_info": "Baclofen har hjulpet litt med muskelstivhet. Fysioterapi to ganger i uken har også vært verdifullt.",
        "contributor_name": "Jonas Olsen"
    },
    {
        "problem": "Pustevansker, spesielt om natten",
        "aids_used": "BiPAP-maskin fra ResMed har vært uvurderlig for nattlig pustestøtte. Slimsuger for å fjerne slim fra luftveiene. Puste-øvelser fra fysioterapeut hjelper også.",
        "medicine_info": "",
        "contributor_name": "Liv Andersen"
    },
    {
        "problem": "Fatigue og utmattelse",
        "aids_used": "Energibesparende teknikker, planlegging av aktiviteter med pauser, tilpasset dagsprogram med hvile. Elektrisk seng som gjør det enkelt å endre stilling uten anstrengelse.",
        "medicine_info": "Modafinil har i noen tilfeller blitt foreskrevet for å redusere fatigue, men med varierende resultater.",
        "contributor_name": "Erik Johansen"
    },
    {
        "problem": "Følelse av isolasjon og behov for støtte og fellesskap",
        "aids_used": "Støttegruppen 'Alltid litt sterkere' har vært uvurderlig for oss. De tilbyr både en Facebook-gruppe hvor man kan dele erfaringer og få råd fra andre i samme situasjon, samt fysiske treff hvor man kan møte andre ALS-pasienter og pårørende. Deres hjemmeside (www.alltidlittsterkere.org) har også mye nyttig informasjon og ressurser. Det å kunne snakke med andre som virkelig forstår hva man går gjennom har betydd enormt mye for hele familien.",
        "medicine_info": "",
        "contributor_name": "Maria Kristiansen"
    },
    {
        "problem": "Vanskeligheter med å finne passende hjelpemidler",
        "aids_used": "NAV hjelpemiddeldatabase (finnhjelpemiddel.nav.no) har vært en fantastisk ressurs for oss. Der kan man søke etter mange ulike typer hjelpemidler basert på behov og funksjonstap. Vi har funnet alt fra spesialbestikk til forflytningshjelpemidler der. Det er også nyttig at man kan se bilder og få tekniske spesifikasjoner før man kontakter hjelpemiddelsentralen. Når vi hadde funnet aktuelle hjelpemidler i databasen, tok vi kontakt med ergoterapeuten som hjalp oss med å søke og tilpasse hjelpemidlene til våre spesifikke behov.",
        "medicine_info": "",
        "contributor_name": "Anders Bergersen"
    }
]

def add_test_data():
    """Add sample test data to the database and vector store"""
    print("Adding sample contributions to the database...")
    
    # Get embeddings model
    try:
        embeddings = get_embeddings()
        embeddings_available = True
    except Exception as e:
        print(f"Warning: Could not initialize embeddings: {e}")
        print("Contributions will be saved to database but not added to vector store.")
        embeddings_available = False
    
    # Process each contribution
    for sample in sample_contributions:
        # Create contribution object
        contribution = Contribution(
            problem=sample["problem"],
            aids_used=sample["aids_used"],
            medicine_info=sample["medicine_info"],
            contributor_name=sample["contributor_name"]
        )
        
        try:
            # Save to database
            contribution_id = save_contribution(contribution)
            print(f"Added contribution from {sample['contributor_name']}")
            
            # Only try to add to vector store if embeddings are available
            if embeddings_available:
                # Create text and metadata for vector store
                text = f"Problem: {sample['problem']}\nAids Used: {sample['aids_used']}"
                if sample["medicine_info"]:
                    text += f"\nMedicine Info: {sample['medicine_info']}"
                
                metadata = create_metadata_from_contribution(
                    contribution_id=contribution_id,
                    problem=sample["problem"],
                    aids_used=sample["aids_used"]
                )
                
                # Add to vector store
                try:
                    add_texts_to_vectorstore([text], [metadata], embeddings)
                    print(f"Added to vector store: {sample['problem'][:30]}...")
                except Exception as e:
                    print(f"Error adding to vector store: {e}")
        
        except Exception as e:
            print(f"Error adding contribution: {e}")
    
    print("Sample data addition complete!")

if __name__ == "__main__":
    add_test_data()
