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
    """Initialize database tables."""
    with get_db_connection() as conn:
        cur = conn.cursor()
        
        # Text embeddings table (Sentence Transformer - 384 dimensions)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS product_text_embeddings (
                product_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                embedding_model TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (product_id) REFERENCES updated_essilor_products("Product ID")
            )
        """)
        
        # Image embeddings table (CLIP - 512 dimensions)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS product_image_embeddings (
                product_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                embedding_model TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (product_id) REFERENCES updated_essilor_products("Product ID")
            )
        """)
        
        # Create indexes for faster queries
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_text_embeddings_product 
            ON product_text_embeddings(product_id)
        """)
        
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_image_embeddings_product 
            ON product_image_embeddings(product_id)
        """)
        
        conn.commit()
        
        