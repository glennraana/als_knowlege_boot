import os
from typing import List, Dict, Any, Optional
import tempfile
from PIL import Image
import pytesseract
from pypdf import PdfReader
import io
import re

class SimpleTextSplitter:
    """En enkel implementasjon av text splitter for å erstatte langchain avhengighet"""
    
    def __init__(self, chunk_size=1000, chunk_overlap=200, length_function=len):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
    
    def split_text(self, text: str) -> List[str]:
        """
        Del tekst inn i overlappende chunks av angitt størrelse
        """
        # Hvis teksten er kortere enn chunk_size, returner hele teksten
        if self.length_function(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        # Del teksten på avsnitt og setninger først
        paragraphs = text.split("\n\n")
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            # Del paragraf i setninger for mer presise chunks
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                sentence_length = self.length_function(sentence)
                
                # Hvis en setning alene er større enn chunk_size, del den videre
                if sentence_length > self.chunk_size:
                    # Legg til eventuell eksisterende chunk først
                    if current_chunk:
                        chunks.append("".join(current_chunk).strip())
                        
                    # Del lang setning i mindre deler
                    words = sentence.split()
                    current_chunk = []
                    current_length = 0
                    
                    for word in words:
                        if current_length + len(word) + 1 > self.chunk_size:
                            chunks.append("".join(current_chunk).strip())
                            # Start ny chunk med overlap
                            overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                            current_chunk = current_chunk[overlap_start:]
                            current_length = self.length_function("".join(current_chunk))
                            
                        current_chunk.append(word + " ")
                        current_length += len(word) + 1
                    
                    if current_chunk:
                        current_chunk_text = "".join(current_chunk).strip()
                        chunks.append(current_chunk_text)
                        # Start med et overlap for neste chunk
                        words_overlap = current_chunk_text.split()[-min(len(current_chunk_text.split()), 
                                                                      self.chunk_overlap//10)]
                        current_chunk = [" ".join(words_overlap) + " "]
                        current_length = self.length_function("".join(current_chunk))
                else:
                    # Hvis current_chunk + sentence er større enn chunk_size, avslutt current_chunk
                    if current_length + sentence_length + 1 > self.chunk_size:
                        chunks.append("".join(current_chunk).strip())
                        # Start med et overlap for neste chunk
                        overlap_words = min(10, len("".join(current_chunk).split()))
                        overlap_text = " ".join("".join(current_chunk).split()[-overlap_words:])
                        current_chunk = [overlap_text + " "]
                        current_length = self.length_function("".join(current_chunk))
                    
                    # Legg til setningen i current_chunk
                    current_chunk.append(sentence + " ")
                    current_length += sentence_length + 1
        
        # Legg til gjenværende chunk hvis den ikke er tom
        if current_chunk:
            chunks.append("".join(current_chunk).strip())
            
        return chunks

def process_text(text: str) -> List[str]:
    """
    Process raw text and split it into chunks for embedding
    
    Args:
        text: The text content to process
        
    Returns:
        List of text chunks
    """
    # Create text splitter
    text_splitter = SimpleTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    # Split text into chunks
    chunks = text_splitter.split_text(text)
    
    return chunks

def process_pdf(pdf_content: bytes) -> List[str]:
    """
    Extract text from PDF content and split into chunks
    
    Args:
        pdf_content: Binary PDF content
        
    Returns:
        List of text chunks
    """
    # Create a BytesIO object from the PDF content
    pdf_file = io.BytesIO(pdf_content)
    
    # Create a PDF reader
    pdf_reader = PdfReader(pdf_file)
    
    # Extract text from each page
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    # Process extracted text
    return process_text(text)

def process_image(image: Image.Image) -> List[str]:
    """
    Extract text from an image using OCR
    
    Args:
        image: PIL Image object
        
    Returns:
        List of text chunks
    """
    # Perform OCR on the image
    text = pytesseract.image_to_string(image, lang='nor+eng')
    
    # Process extracted text
    return process_text(text)

def process_word_document(docx_content: bytes) -> List[str]:
    """
    Extract text from a Word document and split into chunks
    
    Args:
        docx_content: Binary Word document content
        
    Returns:
        List of text chunks
    """
    try:
        # Write binary content to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as temp_file:
            temp_file.write(docx_content)
            temp_path = temp_file.name
        
        # Use python-docx to extract text
        try:
            import docx
            doc = docx.Document(temp_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
        
        # Process extracted text
        return process_text(text)
    except ImportError:
        print("python-docx not installed. Using fallback method.")
        # Fallback: simple text extraction using basic structure
        text = docx_content.decode('utf-8', errors='ignore')
        return process_text(text)
    except Exception as e:
        print(f"Error processing Word document: {e}")
        return []

def process_file(file_path: str) -> Optional[List[str]]:
    """
    Process a file and extract text based on its type
    
    Args:
        file_path: Path to the file
        
    Returns:
        List of text chunks if successful, None otherwise
    """
    try:
        # Determine file type based on extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return process_text(text)
        
        elif ext == '.pdf':
            with open(file_path, 'rb') as f:
                pdf_content = f.read()
            return process_pdf(pdf_content)
        
        elif ext == '.docx':
            with open(file_path, 'rb') as f:
                docx_content = f.read()
            return process_word_document(docx_content)
        
        elif ext in ['.jpg', '.jpeg', '.png']:
            image = Image.open(file_path)
            return process_image(image)
        
        else:
            print(f"Unsupported file type: {ext}")
            return None
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def process_url(url: str) -> Optional[List[str]]:
    """
    Process content from a URL
    
    Args:
        url: URL to process
        
    Returns:
        List of text chunks if successful, None otherwise
    """
    try:
        import requests
        from bs4 import BeautifulSoup
        
        # Fetch URL content
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get the main content (you may need to adjust this based on website structure)
        main_content = soup.find("main") or soup.find("article") or soup.find("div", {"id": "content"}) or soup.body
        
        if main_content:
            # Extract text content
            text = main_content.get_text(separator='\n', strip=True)
        else:
            # Fall back to entire page
            text = soup.get_text(separator='\n', strip=True)
        
        # Process extracted text
        return process_text(text)
    
    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        return None

def create_metadata_from_contribution(contribution_id: str, problem: str, aids_used: str, 
                                     file_type: str = None, file_name: str = None, file_id: str = None,
                                     title: str = None, category: str = None, content_type: str = None,
                                     tags: List[str] = None) -> Dict[str, Any]:
    """
    Create metadata dictionary from contribution data
    
    Args:
        contribution_id: ID of the contribution
        problem: Problem description
        aids_used: Aids used description
        file_type: Type of the file (optional)
        file_name: Name of the file (optional)
        file_id: ID of the file in GridFS (optional)
        title: Title of the contribution (optional)
        category: Category of the contribution (optional)
        content_type: Type of content (optional)
        tags: List of tags (optional)
        
    Returns:
        Metadata dictionary
    """
    metadata = {
        "source": "user_contribution",
        "contribution_id": contribution_id,
        "problem": problem,
        "aids_used": aids_used
    }
    
    # Add file information if available
    if file_type:
        metadata["file_type"] = file_type
    
    if file_name:
        metadata["file_name"] = file_name
    
    if file_id:
        metadata["file_id"] = file_id
    
    # Add new metadata fields if available
    if title:
        metadata["title"] = title
    
    if category:
        metadata["category"] = category
    
    if content_type:
        metadata["content_type"] = content_type
    
    if tags:
        metadata["tags"] = tags
        
    return metadata
