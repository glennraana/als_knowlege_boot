"""
This script diagnoses issues with vector embeddings in the database.
It will:
1. List all contributions
2. List all vector embeddings
3. Identify contributions that don't have corresponding vector embeddings
4. Add missing vector embeddings
"""

import os
from dotenv import load_dotenv
from db.operations import get_all_contributions, get_vector_embeddings, save_vector_embedding
from db.models import Contribution
from rag.embeddings import get_embeddings

# Load environment variables
load_dotenv()

def main():
    # Get embeddings model
    print("Loading embedding model...")
    embedding_model = get_embeddings()
    
    # Get all contributions
    print("Getting all contributions...")
    contributions = get_all_contributions()
    print(f"Found {len(contributions)} contributions")
    
    # Get all vector embeddings
    print("Getting all vector embeddings...")
    vector_embeddings = get_vector_embeddings()
    print(f"Found {len(vector_embeddings)} vector embeddings")
    
    # Create a set of texts that already have embeddings
    embedded_texts = set()
    for vec in vector_embeddings:
        if "text" in vec:
            embedded_texts.add(vec["text"])
    
    # Identify contributions that don't have vector embeddings
    missing_embeddings = []
    for contrib in contributions:
        combined_text = f"{contrib.problem} {contrib.aids_used}"
        if contrib.medicine_info:
            combined_text += f" {contrib.medicine_info}"
        
        if combined_text not in embedded_texts:
            missing_embeddings.append((contrib, combined_text))
    
    print(f"Found {len(missing_embeddings)} contributions without vector embeddings")
    
    # Add missing embeddings
    if missing_embeddings:
        print("Adding missing embeddings...")
        for contrib, text in missing_embeddings:
            print(f"Adding embedding for: {text[:50]}...")
            
            # Create metadata
            metadata = {
                "contribution_id": contrib.id,
                "problem": contrib.problem,
                "aids_used": contrib.aids_used
            }
            if contrib.medicine_info:
                metadata["medicine_info"] = contrib.medicine_info
            if hasattr(contrib, 'file_name') and contrib.file_name:
                metadata["file_name"] = contrib.file_name
            
            # Generate embedding
            try:
                embeddings = embedding_model.embed_documents([text])
                if embeddings and len(embeddings) > 0:
                    # Save embedding
                    embedding_id = save_vector_embedding(text, embeddings[0], metadata)
                    print(f"Successfully added embedding with ID: {embedding_id}")
                else:
                    print(f"Failed to generate embedding for text: {text[:50]}...")
            except Exception as e:
                print(f"Error generating embedding: {e}")
    
    print("Done!")

if __name__ == "__main__":
    main()
