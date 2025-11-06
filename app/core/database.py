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
    
    Note: 
    - orders, orders_updated, users, updated_essilor_products, policies tables 
      are expected to exist already
    - This only creates conversation-related tables
    """
    with get_db_connection() as conn:
        cur = conn.cursor()
        
        # Check if conversations table exists and has PRIMARY KEY
        cur.execute("""
            SELECT sql FROM sqlite_master 
            WHERE type='table' AND name='conversations'
        """)
        result = cur.fetchone()
        
        if result and 'PRIMARY KEY' not in result[0]:
            # Table exists but without PRIMARY KEY - need to recreate
            print("⚠️  Recreating conversations table with PRIMARY KEY constraint...")
            
            # Backup existing data
            cur.execute("SELECT * FROM conversations")
            existing_conversations = cur.fetchall()
            
            # Drop old table
            cur.execute("DROP TABLE IF EXISTS conversations")
            
            # Create new table with PRIMARY KEY
            cur.execute("""
                CREATE TABLE conversations (
                    conversation_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)
            
            # Restore data
            if existing_conversations:
                for row in existing_conversations:
                    cur.execute("""
                        INSERT INTO conversations 
                        (conversation_id, user_id, title, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?)
                    """, (row[0], row[1], row[2], row[3], row[4]))
                print(f"✅ Migrated {len(existing_conversations)} conversations")
        
        elif not result:
            # Table doesn't exist - create it
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
        
        # Messages table (should be fine)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                tool_called TEXT,
                uploaded_image_url TEXT,
                products_data TEXT,
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