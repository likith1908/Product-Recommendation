from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
from uuid import uuid4
import json


class Message(BaseModel):
    """Individual message in a conversation"""
    message_id: str = Field(default_factory=lambda: str(uuid4()))
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tool_called: Optional[str] = None
    uploaded_image_url: Optional[str] = None
    products_data: Optional[str] = None  # JSON string of products array
    
    @property
    def products(self) -> Optional[List[Dict[str, Any]]]:
        """Parse products_data JSON string into list of dicts"""
        if self.products_data:
            try:
                return json.loads(self.products_data)
            except:
                return None
        return None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ConversationBase(BaseModel):
    """Base conversation model"""
    title: Optional[str] = None


class ConversationCreate(ConversationBase):
    """Model for creating a new conversation"""
    user_id: str


class Conversation(ConversationBase):
    """Full conversation model"""
    conversation_id: str
    user_id: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0
    
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ConversationWithMessages(Conversation):
    """Conversation with full message history"""
    messages: List[Message] = []


class ConversationSummary(BaseModel):
    """Lightweight conversation summary for listing"""
    conversation_id: str
    title: Optional[str]
    created_at: datetime
    updated_at: datetime
    message_count: int
    last_message_preview: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }