from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
import uuid
from datetime import datetime

# Define categories and content types as constants
MAIN_CATEGORIES = [
    "ernæring", 
    "hjelpemidler", 
    "mobilitet", 
    "kommunikasjon", 
    "respirasjon", 
    "hverdagsliv", 
    "helse", 
    "forskning", 
    "ressurser", 
    "sosial_støtte"
]

CONTENT_TYPES = [
    "personlig_erfaring", 
    "guide", 
    "tips_og_triks", 
    "forskning", 
    "diskusjon", 
    "anbefaling"
]

class Contribution(BaseModel):
    """Model for user contributions to the ALS knowledge base"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    problem: str
    aids_used: str
    medicine_info: Optional[str] = None
    contributor_name: str = "Anonym"
    file_name: Optional[str] = None
    file_type: Optional[str] = None
    file_content: Optional[bytes] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Nye felt for kategorisering
    title: Optional[str] = None
    category: Optional[str] = None
    sub_categories: Optional[List[str]] = None
    content_type: Optional[str] = None
    tags: Optional[List[str]] = None
    summary: Optional[str] = None
    structured_content: Optional[Dict[str, Any]] = None
    difficulty_level: Optional[str] = None
    related_links: Optional[List[Dict[str, str]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for MongoDB storage."""
        data = self.dict(exclude={"file_content"})
        # Handle binary content separately
        if self.file_content:
            data["file_content"] = self.file_content
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Contribution":
        """Create a Contribution instance from a dictionary"""
        return cls(**data)

class ContentTemplate(BaseModel):
    """Template for structured content based on content type"""
    template_name: str
    content_type: str
    fields: List[Dict[str, Any]]
    description: str
    example: Dict[str, Any]
    
    class Config:
        schema_extra = {
            "example": {
                "template_name": "Personlig erfaring med hjelpemiddel",
                "content_type": "personlig_erfaring",
                "description": "Mal for å dele personlige erfaringer med hjelpemidler",
                "fields": [
                    {"name": "hjelpemiddel", "type": "string", "required": True, "description": "Navn på hjelpemiddelet"},
                    {"name": "bruksperiode", "type": "string", "required": False, "description": "Hvor lenge hjelpemiddelet har vært brukt"},
                    {"name": "fordeler", "type": "array", "required": True, "description": "Fordeler med hjelpemiddelet"},
                    {"name": "ulemper", "type": "array", "required": False, "description": "Eventuelle ulemper eller utfordringer"},
                    {"name": "anskaffelse", "type": "string", "required": False, "description": "Hvordan hjelpemiddelet ble anskaffet"}
                ],
                "example": {
                    "hjelpemiddel": "Elektrisk rullestol med joystick",
                    "bruksperiode": "1 år",
                    "fordeler": ["Lett å manøvrere", "God batterilevetid", "Kompakt design"],
                    "ulemper": ["Kan være vanskelig å komme inn i trange rom"],
                    "anskaffelse": "Gjennom NAV hjelpemiddelsentral etter søknad fra ergoterapeut"
                }
            }
        }
