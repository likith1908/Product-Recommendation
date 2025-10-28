"""
Migration script to add products_data column to messages table.

Run this if you already have an existing database:
    python migration_add_products_data.py
"""

import sqlite3
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.core.database import get_db_connection


def migrate_add_products_data():
    """Add products_data column to messages table"""
    print("="*60)
    print("🔄 MIGRATION: Adding products_data column to messages table")
    print("="*60)
    
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            
            # Check if column already exists
            cur.execute("PRAGMA table_info(messages)")
            columns = [row[1] for row in cur.fetchall()]
            
            if 'products_data' in columns:
                print("✅ Column 'products_data' already exists. No migration needed.")
                return True
            
            print("\n📝 Adding 'products_data' column...")
            
            # Add the new column
            cur.execute("""
                ALTER TABLE messages 
                ADD COLUMN products_data TEXT
            """)
            
            conn.commit()
            
            print("✅ Migration completed successfully!")
            print(f"   Database: {settings.DATABASE_PATH}")
            print(f"   Column 'products_data' added to messages table")
            
            return True
    
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e).lower():
            print("✅ Column already exists. No migration needed.")
            return True
        else:
            print(f"❌ Migration failed: {e}")
            return False
    
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = migrate_add_products_data()
    sys.exit(0 if success else 1)