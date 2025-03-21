from openai import OpenAI
import numpy as np
import os

def get_embeddings():
    """
    Returns a function that generates embeddings using OpenAI's API.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def get_embedding(text):
        """
        Generate an embedding for the given text using OpenAI's API.
        """
        if not text or text.strip() == "":
            # Return a zero embedding for empty text
            return np.zeros(1536)
        
        try:
            # Using the text-embedding-ada-002 model, which is good for multilingual text
            response = client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Return a zero embedding in case of failure
            return np.zeros(1536)
    
    return get_embedding
