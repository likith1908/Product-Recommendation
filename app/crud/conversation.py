"""
CRUD operations for session-based conversations.
Structure: users -> conversations/sessions -> messages
"""

from datetime import datetime
from typing import Optional, List
from uuid import uuid4

from app.core.database import get_db_connection
from app.models.conversation import (
    Conversation, ConversationCreate, ConversationWithMessages,
    ConversationSummary, Message
)


def create_conversation(conversation: ConversationCreate) -> Conversation:
    """Create a new conversation session for a user"""
    conversation_id = str(uuid4())
    now = datetime.utcnow()
    
    # Generate default title with timestamp if not provided
    title = conversation.title or f"Chat {now.strftime('%b %d, %Y %I:%M %p')}"
    
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO conversations (conversation_id, user_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
        """, (conversation_id, conversation.user_id, title, now, now))
        conn.commit()
    
    return Conversation(
        conversation_id=conversation_id,
        user_id=conversation.user_id,
        title=title,
        created_at=now,
        updated_at=now,
        message_count=0
    )


def get_conversation(conversation_id: str, user_id: str) -> Optional[Conversation]:
    """Get a specific conversation (metadata only, no messages)"""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                c.conversation_id,
                c.user_id,
                c.title,
                c.created_at,
                c.updated_at,
                COUNT(m.message_id) as message_count
            FROM conversations c
            LEFT JOIN messages m ON c.conversation_id = m.conversation_id
            WHERE c.conversation_id = ? AND c.user_id = ?
            GROUP BY c.conversation_id
        """, (conversation_id, user_id))
        
        row = cur.fetchone()
        if not row:
            return None
        
        return Conversation(
            conversation_id=row['conversation_id'],
            user_id=row['user_id'],
            title=row['title'],
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']),
            message_count=row['message_count']
        )


def get_conversation_with_messages(
    conversation_id: str,
    user_id: str,
    limit: Optional[int] = None
) -> Optional[ConversationWithMessages]:
    """Get conversation with full message history"""
    conversation = get_conversation(conversation_id, user_id)
    if not conversation:
        return None
    
    with get_db_connection() as conn:
        cur = conn.cursor()
        
        # Get messages - NOW INCLUDING products_data
        query = """
            SELECT 
                message_id, role, content, timestamp,
                tool_called, uploaded_image_url, products_data
            FROM messages
            WHERE conversation_id = ?
            ORDER BY timestamp ASC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cur.execute(query, (conversation_id,))
        rows = cur.fetchall()
        
        messages = [
            Message(
                message_id=row['message_id'],
                role=row['role'],
                content=row['content'],
                timestamp=datetime.fromisoformat(row['timestamp']),
                tool_called=row['tool_called'],
                uploaded_image_url=row['uploaded_image_url'],
                products_data=row['products_data']  # ← Now retrieved from DB
            )
            for row in rows
        ]
    
    return ConversationWithMessages(
        conversation_id=conversation.conversation_id,
        user_id=conversation.user_id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        message_count=len(messages),
        messages=messages
    )


def list_user_conversations(
    user_id: str,
    limit: int = 50,
    offset: int = 0
) -> List[ConversationSummary]:
    """List all conversation sessions for a user with preview"""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                c.conversation_id,
                c.title,
                c.created_at,
                c.updated_at,
                COUNT(m.message_id) as message_count,
                (
                    SELECT content 
                    FROM messages 
                    WHERE conversation_id = c.conversation_id 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                ) as last_message
            FROM conversations c
            LEFT JOIN messages m ON c.conversation_id = m.conversation_id
            WHERE c.user_id = ?
            GROUP BY c.conversation_id
            ORDER BY c.updated_at DESC
            LIMIT ? OFFSET ?
        """, (user_id, limit, offset))
        
        rows = cur.fetchall()
        
        summaries = []
        for row in rows:
            # Create preview from last message
            last_msg = row['last_message']
            preview = None
            if last_msg:
                preview = last_msg[:100] + "..." if len(last_msg) > 100 else last_msg
            
            summaries.append(ConversationSummary(
                conversation_id=row['conversation_id'],
                title=row['title'],
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at']),
                message_count=row['message_count'],
                last_message_preview=preview
            ))
        
        return summaries


def add_message(
    conversation_id: str,
    user_id: str,
    role: str,
    content: str,
    tool_called: Optional[str] = None,
    uploaded_image_url: Optional[str] = None,
    products_data: Optional[str] = None  # ← Parameter exists
) -> Message:
    """Add a message to a conversation"""
    message_id = str(uuid4())
    now = datetime.utcnow()
    
    with get_db_connection() as conn:
        cur = conn.cursor()
        
        # Insert message - NOW INCLUDING products_data in INSERT
        cur.execute("""
            INSERT INTO messages (
                message_id, conversation_id, role, content,
                timestamp, tool_called, uploaded_image_url, products_data
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (message_id, conversation_id, role, content, now, tool_called, uploaded_image_url, products_data))
        
        # Update conversation timestamp
        cur.execute("""
            UPDATE conversations
            SET updated_at = ?
            WHERE conversation_id = ? AND user_id = ?
        """, (now, conversation_id, user_id))
        
        conn.commit()
    
    return Message(
        message_id=message_id,
        role=role,
        content=content,
        timestamp=now,
        tool_called=tool_called,
        uploaded_image_url=uploaded_image_url,
        products_data=products_data
    )


def update_conversation_title(
    conversation_id: str,
    user_id: str,
    title: str
) -> bool:
    """Update conversation title"""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE conversations
            SET title = ?, updated_at = ?
            WHERE conversation_id = ? AND user_id = ?
        """, (title, datetime.utcnow(), conversation_id, user_id))
        conn.commit()
        return cur.rowcount > 0


def delete_conversation(conversation_id: str, user_id: str) -> bool:
    """Delete a conversation and all its messages"""
    with get_db_connection() as conn:
        cur = conn.cursor()
        
        # Delete messages first (foreign key constraint)
        cur.execute("""
            DELETE FROM messages
            WHERE conversation_id = ?
        """, (conversation_id,))
        
        # Delete conversation
        cur.execute("""
            DELETE FROM conversations
            WHERE conversation_id = ? AND user_id = ?
        """, (conversation_id, user_id))
        
        conn.commit()
        return cur.rowcount > 0


def get_user_conversation_count(user_id: str) -> int:
    """Get total number of conversations for a user"""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*) as count
            FROM conversations
            WHERE user_id = ?
        """, (user_id,))
        return cur.fetchone()['count']