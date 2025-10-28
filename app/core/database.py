import sqlite3
from contextlib import contextmanager

from app.core.config import settings


@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(settings.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """
    Initialize database tables for session-based conversations.
    
    Structure (similar to Firestore):
    - conversations: Session metadata (like Firestore documents)
    - messages: Individual messages within sessions (like subcollections)
    """
    with get_db_connection() as conn:
        cur = conn.cursor()
        
        # Conversations table (sessions)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)
        
        # Messages table (messages within sessions)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                tool_called TEXT,
                uploaded_image_url TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
            )
        """)
        
        # Create indexes for better query performance
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_user_id 
            ON conversations(user_id)
        """)
        
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_updated 
            ON conversations(updated_at DESC)
        """)
        
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_conversation 
            ON messages(conversation_id)
        """)
        
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_timestamp 
            ON messages(timestamp)
        """)
        
        conn.commit()
        print("✅ Database initialized (with session-based conversation tables)")