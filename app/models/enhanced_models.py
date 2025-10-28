"""
Enhanced response models that include parsed product data in messages.
Place this file at: app/models/enhanced_models.py
"""

from typing import Optional, List, Dict, Any, TYPE_CHECKING
from pydantic import BaseModel
from datetime import datetime
import json

# Use TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    from app.models.conversation import Message, ConversationWithMessages


class MessageWithProducts(BaseModel):
    """Message model with parsed products array"""
    message_id: str
    role: str
    content: str
    timestamp: datetime
    tool_called: Optional[str] = None
    uploaded_image_url: Optional[str] = None
    products: Optional[List[Dict[str, Any]]] = None  # ← Parsed products
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @classmethod
    def from_message(cls, message: Any) -> "MessageWithProducts":  # Use Any instead of BaseMessage
        """Convert base Message to MessageWithProducts"""
        products = None
        if hasattr(message, 'products_data') and message.products_data:
            try:
                products = json.loads(message.products_data)
            except:
                products = None
        
        return cls(
            message_id=message.message_id,
            role=message.role,
            content=message.content,
            timestamp=message.timestamp,
            tool_called=message.tool_called,
            uploaded_image_url=message.uploaded_image_url,
            products=products
        )


class ConversationWithProductMessages(BaseModel):
    """Conversation with messages that include parsed products"""
    conversation_id: str
    user_id: str
    title: Optional[str]
    created_at: datetime
    updated_at: datetime
    message_count: int
    messages: List[MessageWithProducts] = []
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @classmethod
    def from_conversation(cls, conversation: Any) -> "ConversationWithProductMessages":  # Use Any
        """Convert base ConversationWithMessages to include parsed products"""
        return cls(
            conversation_id=conversation.conversation_id,
            user_id=conversation.user_id,
            title=conversation.title,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            message_count=conversation.message_count,
            messages=[MessageWithProducts.from_message(msg) for msg in conversation.messages]
        )