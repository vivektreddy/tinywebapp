from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class Source(BaseModel):
    citation_number: int
    title: str
    url: Optional[str] = None
    excerpt: str

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    sources: List[Source] = Field(default_factory=list)
