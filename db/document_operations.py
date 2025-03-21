"""
Database operations for document management system.
"""

import os
from typing import List, Dict, Any, Optional, BinaryIO
import gridfs
from pymongo.bson.objectid import ObjectId
from pymongo import ASCENDING, DESCENDING
from datetime import datetime

from db.connection import get_database
from db.document_models import Document, Folder
from document_processor.processor import process_text, process_pdf, process_image, process_url
from rag.vectorstore import add_texts_to_vectorstore

# Get database and collections
db = get_database()
documents_collection = db["documents"]
folders_collection = db["folders"]
fs = gridfs.GridFS(db)

def save_document(document: Document, file_content: bytes) -> str:
    """
    Save a document to the database
    
    Args:
        document: Document model
        file_content: Binary content of the file
        
    Returns:
        Document ID
    """
    # Save file content to GridFS
    file_id = fs.put(
        file_content,
        filename=document.name,
        content_type=document.content_type
    )
    
    # Update document with file_id
    document.file_id = str(file_id)
    
    # Save document metadata to collection
    document_dict = document.to_dict()
    result = documents_collection.insert_one(document_dict)
    
    return document.id

def get_document_by_id(document_id: str) -> Optional[Document]:
    """
    Get a document by its ID
    
    Args:
        document_id: ID of the document
        
    Returns:
        Document if found, None otherwise
    """
    document_data = documents_collection.find_one({"id": document_id})
    
    if document_data:
        return Document.from_dict(document_data)
    
    return None

def get_document_content(document_id: str) -> Optional[bytes]:
    """
    Get the content of a document by its ID
    
    Args:
        document_id: ID of the document
        
    Returns:
        Document content if found, None otherwise
    """
    document = get_document_by_id(document_id)
    
    if document and document.file_id:
        try:
            file_id = ObjectId(document.file_id)
            grid_out = fs.get(file_id)
            return grid_out.read()
        except Exception as e:
            print(f"Error retrieving document content: {e}")
    
    return None

def create_folder(folder: Folder) -> str:
    """
    Create a folder in the document management system
    
    Args:
        folder: Folder model
        
    Returns:
        Folder ID
    """
    folder_dict = folder.to_dict()
    result = folders_collection.insert_one(folder_dict)
    
    return folder.id

def get_folder_by_path(path: str) -> Optional[Folder]:
    """
    Get a folder by its path
    
    Args:
        path: Path of the folder
        
    Returns:
        Folder if found, None otherwise
    """
    folder_data = folders_collection.find_one({"path": path})
    
    if folder_data:
        return Folder.from_dict(folder_data)
    
    return None

def get_folders_in_folder(parent_path: str) -> List[Folder]:
    """
    Get all folders in a parent folder
    
    Args:
        parent_path: Path of the parent folder
        
    Returns:
        List of folders
    """
    folder_data = folders_collection.find({"parent_path": parent_path})
    
    return [Folder.from_dict(folder) for folder in folder_data]

def get_documents_in_folder(folder_path: str) -> List[Document]:
    """
    Get all documents in a folder
    
    Args:
        folder_path: Path of the folder
        
    Returns:
        List of documents
    """
    document_data = documents_collection.find({"folder_path": folder_path})
    
    return [Document.from_dict(doc) for doc in document_data]

def process_document_for_rag(document_id: str) -> bool:
    """
    Process a document and add it to the RAG system
    
    Args:
        document_id: ID of the document
        
    Returns:
        True if successful, False otherwise
    """
    document = get_document_by_id(document_id)
    
    if not document:
        return False
    
    # Get document content
    content = get_document_content(document_id)
    
    if not content:
        return False
    
    try:
        # Process document based on type
        chunks = []
        
        if document.type == "application/pdf":
            chunks = process_pdf(content)
        elif document.type in ["text/plain", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
            # For Word documents, we'll need to extract text first
            if document.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                # In a real implementation, we would use a library like python-docx
                # For now, we'll just use the file name as a placeholder
                text = f"Content from {document.name} would be extracted here"
            else:
                text = content.decode("utf-8")
            
            chunks = process_text(text)
        elif document.type in ["image/jpeg", "image/png"]:
            # Convert bytes to PIL Image for processing
            import io
            from PIL import Image
            
            image = Image.open(io.BytesIO(content))
            chunks = process_image(image)
        elif document.type == "text/html":
            # For HTML documents, extract text
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(content, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Get text
            text = soup.get_text(separator='\n', strip=True)
            chunks = process_text(text)
        
        # Create metadata for the document
        metadata = {
            "source": "uploaded_document",
            "document_id": document.id,
            "document_name": document.name,
            "document_type": document.type,
            "folder_path": document.folder_path,
            "tags": document.tags
        }
        
        # Add chunks to the vector store
        if chunks:
            # Create metadata for each chunk
            metadatas = [metadata for _ in chunks]
            
            # Get embeddings
            from rag.embeddings import get_embeddings
            embeddings = get_embeddings()
            
            # Add to vector store
            add_texts_to_vectorstore(chunks, metadatas, embeddings)
            
            # Mark document as processed
            documents_collection.update_one(
                {"id": document_id},
                {"$set": {"processed": True}}
            )
            
            return True
    
    except Exception as e:
        print(f"Error processing document: {e}")
    
    return False

def fetch_content_from_url(url: str, folder_path: str = "/", tags: List[str] = []) -> Optional[str]:
    """
    Fetch content from a URL and save it as a document
    
    Args:
        url: URL to fetch
        folder_path: Path to save the document in
        tags: Tags to associate with the document
        
    Returns:
        Document ID if successful, None otherwise
    """
    try:
        import requests
        from bs4 import BeautifulSoup
        
        # Fetch URL content
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title and content
        title = soup.title.string if soup.title else url.split("/")[-1]
        text_content = soup.get_text(separator=' ', strip=True)
        
        # Create document
        document = Document(
            name=title,
            type="text/html",
            size=len(text_content),
            folder_path=folder_path,
            content_type="text/html",
            tags=tags,
            source_url=url
        )
        
        # Save document content
        document_id = save_document(document, response.content)
        
        # Process document for RAG
        process_document_for_rag(document_id)
        
        return document_id
    
    except Exception as e:
        print(f"Error fetching content from URL {url}: {e}")
        return None

def initialize_root_folder():
    """
    Initialize the root folder if it doesn't exist
    """
    root_folder = get_folder_by_path("/")
    
    if not root_folder:
        root_folder = Folder(
            name="Root",
            path="/",
            parent_path=None
        )
        
        create_folder(root_folder)

# Initialize root folder on module import
initialize_root_folder()
