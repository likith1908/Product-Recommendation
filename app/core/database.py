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
    Initialize database tables.
    
    Note: With ChromaDB migration, we only keep SQLite for:
    - Product catalog (updated_essilor_products table)
    - User authentication data
    
    Vector embeddings are now stored in ChromaDB.
    """
    with get_db_connection() as conn:
        cur = conn.cursor()
        
        # Keep the products table (source of truth for product data)
        # Embeddings are now in ChromaDB, not SQLite
        
        # Optional: You can drop the old embedding tables if you want
        # Uncomment these lines after successful migration:
        # cur.execute("DROP TABLE IF EXISTS product_text_embeddings")
        # cur.execute("DROP TABLE IF EXISTS product_image_embeddings")
        
        conn.commit()
        print("✅ Database initialized (embeddings now in ChromaDB)")