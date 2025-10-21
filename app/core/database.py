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
        
        # Users table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE,
                full_name TEXT,
                password_hash TEXT NOT NULL,
                is_active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        
        # Create default test user (johndoe with password 'secret')
        cur.execute("SELECT 1 FROM users WHERE username = ?", ("johndoe",))
        if not cur.fetchone():
            # This is the hash for password 'secret'
            default_hash = "$argon2id$v=19$m=65536,t=3,p=4$wagCPXjifgvUFBzq4hqe3w$CYaIb8sB+wtD+Vu/P4uod1+Qof8h+1g7bbDlBID48Rc"
            cur.execute(
                "INSERT INTO users (username, email, full_name, password_hash, is_active) "
                "VALUES (?, ?, ?, ?, ?)",
                ("johndoe", "johndoe@example.com", "John Doe", default_hash, 1),
            )
            conn.commit()