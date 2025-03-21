from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.vectorstores.base import VectorStore
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceEndpoint, HuggingFacePipeline
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from typing import Dict, Any, List, Optional, Mapping
import os
import logging
import io
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Debugging for OpenAI API key
api_key = os.environ.get("OPENAI_API_KEY", "")
print(f"API key loaded (first 5 chars): {api_key[:5] if api_key else 'None'}")
print(f"API key length: {len(api_key) if api_key else 0}")

def get_llm(use_for="chat"):
    """
    Get the LLM to use for chat or compression.
    
    Args:
        use_for: The purpose of the LLM, either "chat" or "compression".
        
    Returns:
        A ChatOpenAI instance.
    """
    # Bruk miljøvariabelen fra .env
    # IKKE hardkode API-nøkler her
    
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    if use_for == "chat":
        # Use GPT-4o for chat responses
        return ChatOpenAI(
            model="gpt-4o",
            temperature=0.7,
            max_tokens=1000,
        )
    else:
        # Use GPT-3.5 for document compression (cheaper option)
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.0,
            max_tokens=300,
        )

from typing import List, Dict, Any, Optional, Callable
import os
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
from langchain.schema.language_model import BaseLanguageModel
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

from rag.vectorstore import get_vectorstore

class MultiQueryRetriever:
    """Retriever that generates multiple queries from a single query to improve retrieval."""
    
    def __init__(self, retriever, llm=None, num_queries=3):
        """
        Initialize the multi-query retriever
        
        Args:
            retriever: Base retriever to use
            llm: Language model to use for query expansion (if None, don't expand)
            num_queries: Number of queries to generate
        """
        self.retriever = retriever
        self.llm = llm
        self.num_queries = num_queries if llm is not None else 1
        
    def _generate_queries(self, query: str) -> List[str]:
        """Generate multiple queries from a single query"""
        if self.llm is None:
            return [query]
            
        try:
            # Create a prompt for generating multiple search queries
            prompt = PromptTemplate(
                template="""Du skal generere {num_queries} forskjellige søkefraser basert på følgende spørsmål. 
                Søkefrasene skal være på norsk og være relatert til ALS (Amyotrofisk lateral sklerose).
                Formålet er å finne relevant informasjon som hjelper med å svare på spørsmålet.
                
                Spørsmål: {question}
                
                Gi bare søkefrasene, én per linje, uten nummerering eller andre forklaringer.
                """,
                input_variables=["question", "num_queries"]
            )
            
            # Generate alternative search queries
            chain = prompt | self.llm | StrOutputParser()
            result = chain.invoke({"question": query, "num_queries": self.num_queries})
            
            # Parse result into a list of queries
            queries = [q.strip() for q in result.split("\n") if q.strip()]
            
            # Ensure the original query is included
            if query not in queries:
                queries.append(query)
                
            # Limit to requested number
            return queries[:self.num_queries]
            
        except Exception as e:
            print(f"Error generating multiple queries: {e}")
            return [query]
    
    def invoke(self, query: str) -> List[Document]:
        """
        Retrieve documents using multiple queries
        
        Args:
            query: Original query
            
        Returns:
            List of documents
        """
        # Generate multiple queries
        queries = self._generate_queries(query)
        print(f"Generated queries: {queries}")
        
        # Retrieve documents for each query
        all_docs = []
        seen_content = set()
        
        for q in queries:
            try:
                docs = self.retriever.invoke(q)
                
                # Filter out duplicates
                for doc in docs:
                    content = doc.page_content
                    if content not in seen_content:
                        seen_content.add(content)
                        all_docs.append(doc)
            except Exception as e:
                print(f"Error retrieving documents for query '{q}': {e}")
        
        # Sort by relevance (if metadata has score)
        try:
            all_docs = sorted(all_docs, key=lambda d: d.metadata.get("score", 0), reverse=True)
        except Exception as e:
            print(f"Error sorting documents: {e}")
        
        return all_docs

def get_retriever(embeddings=None, k=20):
    """
    Get a retriever for RAG
    
    Args:
        embeddings: Embedding model to use for encoding text
        k: Number of documents to retrieve (default: 20)
        
    Returns:
        A configured retriever
    """
    # Import here to avoid circular imports
    from rag.embeddings import get_embeddings
    
    if embeddings is None:
        embeddings = get_embeddings()
    
    # Get the vector store
    vector_store = get_vectorstore(embeddings)
    
    # Create a retriever with the vector store
    base_retriever = VectorStoreRetriever(
        vectorstore=vector_store,
        search_kwargs={"k": k}
    )
    
    # Get LLM for query expansion
    try:
        from rag.llm import get_llm
        llm = get_llm("query_expansion")
        
        # Use multi-query retriever if LLM is available
        if llm is not None:
            print("Using multi-query retriever")
            return MultiQueryRetriever(base_retriever, llm=llm, num_queries=3)
    except Exception as e:
        print(f"Error setting up multi-query retriever: {e}")
    
    # Fallback to basic retriever
    print("Using basic retriever")
    return base_retriever

class VectorStoreRetriever:
    """Custom retriever that wraps a vector store and handles errors gracefully"""
    
    def __init__(self, vectorstore: VectorStore, search_kwargs: Dict[str, Any] = None):
        self.vectorstore = vectorstore
        self.search_kwargs = search_kwargs or {"k": 5}
        self.search_type = "similarity"
        
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Get documents relevant to a query
        
        Args:
            query: Query string
            
        Returns:
            List of relevant documents
        """
        try:
            if self.search_type == "similarity":
                docs = self.vectorstore.similarity_search(query, **self.search_kwargs)
                return docs
            else:
                raise ValueError(f"Invalid search type: {self.search_type}")
        except Exception as e:
            import traceback
            print(f"Error in vector store retrieval: {str(e)}")
            print(traceback.format_exc())
            # Return empty list on error
            return []
            
    def invoke(self, input):
        """
        Invoke the retriever with the input (for compatibility with LangChain)
        
        Args:
            input: Input string or dictionary with "input" key
            
        Returns:
            List of relevant documents
        """
        if isinstance(input, dict) and "input" in input:
            query = input["input"]
        else:
            query = input
            
        return self.get_relevant_documents(query)

class CustomRetrievalChain:
    """
    Custom retrieval chain that combines retrieval with LLM generation
    """
    
    def __init__(self, llm: BaseLanguageModel, retriever, prompt_template: str):
        self.llm = llm
        self.retriever = retriever
        self.prompt = PromptTemplate.from_template(prompt_template)
        
    def format_docs(self, docs: List[Document]) -> str:
        """Format a list of documents into a string"""
        if not docs:
            return "No relevant documents found."
            
        formatted_docs = []
        for i, doc in enumerate(docs):
            doc_string = f"Document {i+1}:\n"
            doc_string += f"Content: {doc.page_content}\n"
            if hasattr(doc, 'metadata') and doc.metadata:
                doc_string += f"Metadata: {doc.metadata}\n"
            formatted_docs.append(doc_string)
            
        return "\n\n".join(formatted_docs)
        
    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke the retrieval chain
        
        Args:
            input_dict: Input dictionary with "input" key
            
        Returns:
            Dictionary with "answer" and "source_documents" keys
        """
        query = input_dict["input"]
        vector_search_error = None
        
        try:
            # Retrieve relevant documents
            docs = self.retriever.invoke(query)
            
            # If no documents found, make a second attempt with adjusted query
            if not docs or len(docs) == 0:
                # Try with a more generic version of the query
                simplified_query = " ".join(query.split()[:5])  # Use first 5 words
                print(f"No documents found with original query. Trying simplified query: {simplified_query}")
                docs = self.retriever.invoke(simplified_query)
            
            # Format documents for the prompt
            formatted_docs = self.format_docs(docs)
            
            # Check if LLM is available
            if self.llm is None:
                return self._generate_fallback_answer(query, docs)
            
            # Prepare the prompt
            prompt_input = {
                "context": formatted_docs,
                "question": query
            }
            
            # Generate answer using LLM
            chain = (
                self.prompt 
                | self.llm 
                | StrOutputParser()
            )
            answer = chain.invoke(prompt_input)
            
            # Forbedre svaret med personlige erfaringer (dersom tilgjengelig)
            if docs and len(docs) > 0:
                answer += self._enhance_with_personal_experiences(docs, answer)
            
            # Add standard footer
            from rag.templates import STANDARD_FOOTER
            answer = answer + STANDARD_FOOTER
            
            # Filter unwanted organization references
            answer = self._filter_unwanted_references(answer)
            
            # Return result
            return {
                "answer": answer,
                "source_documents": docs,
                "vector_search_error": vector_search_error
            }
            
        except Exception as e:
            import traceback
            vector_search_error = str(e)
            print(f"Error in retrieval chain: {vector_search_error}")
            print(traceback.format_exc())
            
            # Return fallback answer
            return self._generate_fallback_answer(query, [])
            
    def _filter_unwanted_references(self, text: str) -> str:
        """Filter out references to unwanted organizations"""
        import re
        
        # Define patterns to match and replace
        unwanted_orgs = [
            (r'ALS Norge', 'støttegruppen "Alltid litt sterkere"'),
            (r'ALS Ligaen', 'støttegruppen "Alltid litt sterkere"'),
            (r'ALS-Norge', 'støttegruppen "Alltid litt sterkere"'),
            (r'ALS-Ligaen', 'støttegruppen "Alltid litt sterkere"'),
            (r'ALS Foreningen', 'støttegruppen "Alltid litt sterkere"'),
            (r'ALS-Foreningen', 'støttegruppen "Alltid litt sterkere"')
        ]
        
        # Replace all unwanted references
        filtered_text = text
        for pattern, replacement in unwanted_orgs:
            filtered_text = re.sub(pattern, replacement, filtered_text, flags=re.IGNORECASE)
            
        return filtered_text

    def _generate_fallback_answer(self, query: str, docs: List[Document]) -> Dict[str, Any]:
        """Generate a fallback answer when LLM is not available"""
        
        # Generate a simple response based on available documents
        if docs and len(docs) > 0:
            answer = (
                f"Her er informasjon relatert til ditt spørsmål om '{query}':\n\n"
                f"Basert på våre bidrag fra ALS-pasienter og pårørende, har vi funnet følgende:\n\n"
            )
            
            for i, doc in enumerate(docs):
                answer += f"Kilde {i+1}: {doc.page_content}\n\n"
            
            # Add standardized footer with better formatting
            from rag.templates import STANDARD_FOOTER
            answer += STANDARD_FOOTER
                
        else:
            answer = (
                f"Jeg kunne ikke finne spesifikk informasjon om '{query}' i vår kunnskapsbase.\n\n"
                "For personlig veiledning anbefaler jeg at du kontakter helsepersonell."
            )
            
            # Add standardized footer
            from rag.templates import STANDARD_FOOTER
            answer += STANDARD_FOOTER
        
        # Filter any unwanted references
        answer = self._filter_unwanted_references(answer)
        
        return {
            "answer": answer,
            "source_documents": docs,
            "vector_search_error": "OpenAI API key not configured or service unavailable"
        }

    def _enhance_with_personal_experiences(self, docs: List[Document], current_answer: str) -> str:
        """
        Enhance the answer with personal experiences from the documents if appropriate
        
        Returns:
            Additional text to add to the answer, or empty string if no enhancement
        """
        # Extract question keywords for matching
        import re
        question_keywords = set(re.findall(r'\b[A-Za-zæøåÆØÅ]{3,}\b', current_answer.lower()))
        
        # Get main topic from the answer (first few sentences)
        first_paragraph = " ".join(current_answer.split("\n\n")[0:1])
        main_topic_words = set(re.findall(r'\b[A-Za-zæøåÆØÅ]{4,}\b', first_paragraph.lower()))
        
        # Extract potential action words and problem areas from the answer
        action_words = set()
        problem_areas = set()
        for sentence in current_answer.lower().split('.'):
            if any(word in sentence for word in ["hvordan", "hjelpe", "bruke", "løse", "håndtere", "takle"]):
                action_words.update(re.findall(r'\b[A-Za-zæøåÆØÅ]{4,}\b', sentence))
            if any(word in sentence for word in ["problem", "utfordring", "vansker", "vanskelighet", "vanskelig"]):
                problem_areas.update(re.findall(r'\b[A-Za-zæøåÆØÅ]{4,}\b', sentence))
        
        # Filter out terms that suggest we're talking about general information or research
        research_only_terms = ["forskning", "studie", "vitenskapelig", "medisinsk forskning", 
                              "klinisk studie", "forskere", "vitenskapelig studie"]
        
        # Check if the answer is about research only
        is_research_only = any(term in current_answer.lower() for term in research_only_terms) and not any(
            term in current_answer.lower() for term in [
                "symptom", "behandling", "hjelpemiddel", "rullestol", "ernæring", 
                "kommunikasjon", "respirasjon", "hverdagsliv", "mestring"
            ]
        )
        
        # Don't include personal experiences for pure research questions
        if is_research_only:
            return ""
        
        # Look for documents with personal experiences
        personal_experiences = []
        for doc in docs:
            content = doc.page_content.lower()
            title = doc.metadata.get("document_name", "") or doc.metadata.get("title", "")
            
            # Look for first-person narratives
            is_personal = any(term in content for term in ["jeg opplevde", "min erfaring", 
                                           "for meg", "jeg har", "jeg føler", "jeg brukte",
                                           "min hverdag", "min situasjon"])
            
            # Skip if not personal experience
            if not is_personal:
                continue
                
            # Extract keywords from document
            doc_keywords = set(re.findall(r'\b[A-Za-zæøåÆØÅ]{3,}\b', content))
            
            # Get the main topic words in the document title or first paragraph
            title_words = set(re.findall(r'\b[A-Za-zæøåÆØÅ]{4,}\b', title.lower()))
            first_para = content.split(".")[0] + "." if "." in content else content[:200]
            doc_main_topic = set(re.findall(r'\b[A-Za-zæøåÆØÅ]{4,}\b', first_para.lower()))
            
            # Check if main topic matches strictly
            main_topic_match = len(main_topic_words.intersection(doc_main_topic.union(title_words))) >= 2
            
            # Count how many key terms from the answer appear in this document
            term_matches = sum(1 for term in question_keywords if term in content)
            
            # Check if this document addresses the same problem area
            problem_match = False
            if problem_areas:
                problem_match = any(problem in content for problem in problem_areas)
            
            # Check if this document mentions the same actions 
            action_match = False
            if action_words:
                action_match = len(action_words.intersection(doc_keywords)) >= 1
            
            # Get metadata
            doc_category = doc.metadata.get("kategori", "").lower() if "kategori" in doc.metadata else ""
            doc_description = doc.metadata.get("problem", "").lower() if "problem" in doc.metadata else ""
            
            # Calculate specific content similarity based on problem and solution
            # This checks if the document talks about the same problem or solution as in the question
            problem_solution_match = False
            if doc_description:
                desc_words = set(re.findall(r'\b[A-Za-zæøåÆØÅ]{4,}\b', doc_description.lower()))
                problem_solution_match = len(desc_words.intersection(question_keywords)) >= 2
            
            # Only consider relevant if it passes multiple relevance tests
            # Needs to match main topic AND have several keyword matches AND either match problem or action
            is_relevant = (main_topic_match and term_matches >= 4 and 
                          (problem_match or action_match or problem_solution_match))
            
            # Special case handling - exact category match
            if doc_category:
                # Check if document category keywords appear in our question
                category_terms = set(re.findall(r'\b[A-Za-zæøåÆØÅ]{3,}\b', doc_category))
                cat_matches = len(category_terms.intersection(question_keywords))
                
                # Category must match AND have term matches
                if cat_matches >= 2 and term_matches >= 3:
                    is_relevant = True
            
            # If we have category and tags, use them for better matching
            if "tags" in doc.metadata and doc.metadata["tags"]:
                tags = doc.metadata["tags"] if isinstance(doc.metadata["tags"], list) else []
                tag_matches = sum(1 for tag in tags if tag.lower() in current_answer.lower())
                # Strong tag match can also indicate relevance
                if tag_matches >= 2:
                    is_relevant = True
            
            if is_relevant:
                # Calculate a comprehensive relevance score
                relevance_score = (
                    term_matches + 
                    (3 if main_topic_match else 0) + 
                    (2 if problem_match else 0) + 
                    (2 if action_match else 0) + 
                    (3 if problem_solution_match else 0)
                )
                
                # Store tuple of (content, relevance_score, title)
                personal_experiences.append((doc.page_content, relevance_score, title))
                
        if not personal_experiences or len(personal_experiences) < 1:
            return ""
            
        # Sort by relevance score (descending)
        personal_experiences.sort(key=lambda x: x[1], reverse=True)
        
        # Only show if relevance score is high enough
        top_experience = personal_experiences[0]
        if top_experience[1] < 7:  # Minimum relevance threshold
            return ""
            
        # Add a section with personlige erfaringer
        experience_text = "\n\n## Personlige erfaringer\n\n"
        experience_text += "Her er en personlig erfaring delt av ALS-pasienter eller pårørende som kan være relevant:\n\n"
        
        # Add only the most relevant personal experience
        exp, _, title = top_experience
        # Trim and clean up the experience
        cleaned_exp = exp.strip()
        if len(cleaned_exp) > 300:  # Limit length
            cleaned_exp = cleaned_exp[:300] + "..."
        
        experience_text += f"- *\"{cleaned_exp}\"*\n\n"
            
        return experience_text

def get_rag_chain(retriever, k=20):
    """
    Get a RAG chain that combines retrieval with generation
    
    Args:
        retriever: Retriever to use
        k: Number of documents to retrieve
        
    Returns:
        A RAG chain
    """
    try:
        from rag.llm import get_llm
        from rag.templates import SYSTEM_TEMPLATE, HUMAN_TEMPLATE, STANDARD_FOOTER
        
        # Get LLM
        llm = get_llm()
        
        # Create RAG chain
        return CustomRetrievalChain(
            llm=llm,
            retriever=retriever,
            prompt_template=SYSTEM_TEMPLATE + "\n\n" + HUMAN_TEMPLATE
        )
    except Exception as e:
        print(f"Error creating RAG chain: {e}")
        import traceback
        print(traceback.format_exc())
        
        # Return a chain with no LLM that will use fallback answers
        return CustomRetrievalChain(
            llm=None,
            retriever=retriever,
            prompt_template="Dette blir ignorert siden LLM er None"
        )

def get_rag_chain_old(retriever):
    """
    Create a RAG chain that uses the retriever to fetch relevant
    documents and generates a response
    
    Args:
        retriever: The document retriever
        
    Returns:
        A configured RAG chain
    """
    # Create the LLM
    llm = get_llm(use_for="chat")
    
    # Create system template that includes ALS-specific context and formatting
    system_template = """Du er en AI-assistent for ALS (Amyotrofisk Lateral Sklerose) pasienter og omsorgspersoner i Norge.
    Du gir nyttig, nøyaktig og medfølende informasjon om ALS-symptomer, hjelpemidler, medisiner og ressurser.
    
    Bruk følgende kontekst for å svare på brukerens spørsmål. Formater "Problem:", "Hjelpemidler:" og "Medisin:" i fet skrift (med ** rundt), 
    og sørg for at hver kategori er på en egen linje. For eksempel:
    
    **Problem:**
    Beskrivelse av problemet...
    
    **Hjelpemidler:**
    Beskrivelse av hjelpemidlene...
    
    Hvis du ikke vet svaret, vær ærlig og foreslå at de rådfører seg med helsepersonell.
    
    I dine svar bør du inkludere informasjon om:
    1. Støttegruppen "Alltid litt sterkere" som er en viktig støttegruppe for ALS-pasienter og pårørende i Norge.
       Nevn at de har en hjemmeside på https://www.alltidlittsterkere.org og en Facebook-gruppe hvor pasienter og pårørende kan utveksle erfaringer.
    2. NAV hjelpemiddeldatabase (https://finnhjelpemiddel.nav.no) når du diskuterer hjelpemidler, hvor brukere kan søke etter og finne informasjon
       om tilgjengelige hjelpemidler for ulike problemer.
    
    Kontekst: {context}
    
    Spørsmål: {input}
    
    I dine svar:
    1. Vær spesifikk og gi praktiske råd når det er mulig
    2. Henvis til norske helseressurser når det er hensiktsmessig
    3. Vær følsom for de emosjonelle aspektene ved ALS
    4. Skill tydelig mellom medisiner, hjelpemidler og generelle råd
    5. Erkjenn når informasjon kommer fra andre ALS-pasienters erfaringer
    6. Fokuser på å gi informasjon som er relevant for ALS-pasienter i Norge
    7. Hvis det finnes alternative behandlinger eller tilnærminger, nevn dem
    8. Formater alltid "Problem:", "Hjelpemidler:" og "Medisin:" i fet skrift og på egne linjer
    9. Inkluder alltid informasjon om støttegruppen "Alltid litt sterkere" (https://www.alltidlittsterkere.org) når det er naturlig å nevne støttegrupper
    10. Henvis til NAV hjelpemiddeldatabase (https://finnhjelpemiddel.nav.no) når du diskuterer spesifikke hjelpemidler
    
    Svar:"""
    
    # Create the document chain
    prompt = ChatPromptTemplate.from_template(system_template)
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create the standard retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # Create a safe retrieval wrapper class
    class SafeRetrievalChain:
        def __init__(self, retrieval_chain, retriever, document_chain):
            self.retrieval_chain = retrieval_chain
            self.retriever = retriever
            self.document_chain = document_chain
        
        def invoke(self, inputs):
            try:
                # First try using the standard retrieval chain
                return self.retrieval_chain.invoke(inputs)
            except Exception as e:
                print(f"Error during standard retrieval: {e}")
                try:
                    # Fall back to a manual implementation if standard fails
                    query = inputs["input"]
                    
                    # Safely retrieve documents
                    try:
                        docs = self.retriever.get_relevant_documents(query)
                        vector_search_error = None
                    except Exception as retriever_error:
                        print(f"Error during document retrieval: {retriever_error}")
                        docs = []
                        vector_search_error = str(retriever_error)
                    
                    # Generate answer with available documents
                    answer = self.document_chain.invoke({"input": query, "context": docs})
                    
                    # Add error information and source documents
                    if vector_search_error:
                        answer["vector_search_error"] = vector_search_error
                    
                    answer["source_documents"] = docs
                    return answer
                except Exception as fallback_error:
                    print(f"Error during fallback retrieval: {fallback_error}")
                    # Ultimate fallback - return a minimal valid response
                    return {
                        "answer": "Jeg beklager, men jeg kunne ikke hente relevant informasjon. Vennligst prøv igjen senere.",
                        "vector_search_error": str(e),
                        "source_documents": []
                    }
        
        def __call__(self, inputs):
            # Support for function-call syntax
            return self.invoke(inputs)
    
    # Return a wrapped chain that handles errors gracefully
    return SafeRetrievalChain(retrieval_chain, retriever, document_chain)
