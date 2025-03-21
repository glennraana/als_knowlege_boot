"""
Models for document management system.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import uuid
from datetime import datetime

class Document(BaseModel):
    """Model for documents uploaded to the system."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: str
    size: int
    folder_path: str = "/"  # Default to root folder
    content_type: str
    metadata: Dict[str, Any] = {}
    tags: List[str] = []
    created_at: datetime = Field(default_factory=datetime.now)
    file_id: Optional[str] = None  # ID of the file in GridFS
    processed: bool = False
    source_url: Optional[str] = None  # For web scraped documents
    
    class Config:
        arbitrary_types_allowed = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "size": self.size,
            "folder_path": self.folder_path,
            "content_type": self.content_type,
            "metadata": self.metadata,
            "tags": self.tags,
            "created_at": self.created_at,
            "file_id": self.file_id,
            "processed": self.processed,
            "source_url": self.source_url
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create a model from a dictionary."""
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


class Folder(BaseModel):
    """Model for folders in the document management system."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    path: str
    parent_path: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "path": self.path,
            "parent_path": self.parent_path,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Folder':
        """Create a model from a dictionary."""
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)
