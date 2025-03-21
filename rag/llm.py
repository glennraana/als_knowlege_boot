"""
LLM module for handling different language models
"""

import os
from typing import Optional, Union, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI

def get_llm(use_for: Optional[str] = None) -> BaseLanguageModel:
    """
    Get an LLM instance
    
    Args:
        use_for: What the LLM will be used for (affects model selection)
        
    Returns:
        LLM instance or None if not available
    """
    # Check if the API key is available
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(f"Warning: No OpenAI API key found for {use_for}")
        return None
        
    try:
        # Model selection based on usage
        if use_for == "query_expansion":
            # For query expansion, use a faster, cheaper model
            return ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.3,
                max_tokens=100
            )
        else:
            # For main responses, use a more powerful model with higher creativity
            return ChatOpenAI(
                model="gpt-4o",  # Use GPT-4o for higher quality responses
                temperature=0.7,  # Slightly higher temperature for more creative responses
                max_tokens=1000   # Allow longer responses
            )
            
    except Exception as e:
        print(f"Error creating LLM: {e}")
        import traceback
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    # Test if LLM can be initialized
    llm = get_llm()
    print(f"Initialized LLM: {type(llm).__name__}")
