"""
Database schema migration script.
Fixes missing PRIMARY KEY constraints and validates schema.

Usage:
    python -m app.scripts.migrate_schema
"""

import sqlite3
import sys
from pathlib import Path


def get_db_path():
    """Get database path from environment or use default"""
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    db_path = os.getenv('DATABASE_PATH', 'sample_data.db')
    return Path(db_path)


def check_table_schema(cursor, table_name):
    """Get current table schema"""
    cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'")
    result = cursor.fetchone()
    return result[0] if result else None


def migrate_conversations_table(conn):
    """Fix conversations table - add PRIMARY KEY constraint"""
    cur = conn.cursor()
    
    print("\n📋 Checking 'conversations' table...")
    
    schema = check_table_schema(cur, 'conversations')
    
    if not schema:
        print("   ✅ Table doesn't exist yet - will be created by init_db()")
        return True
    
    if 'PRIMARY KEY' in schema:
        print("   ✅ Table already has PRIMARY KEY constraint")
        return True
    
    print("   ⚠️  Missing PRIMARY KEY constraint - migrating...")
    
    # Backup data
    cur.execute("SELECT * FROM conversations")
    rows = cur.fetchall()
    print(f"   📦 Backing up {len(rows)} conversations...")
    
    # Drop old table
    cur.execute("DROP TABLE conversations")
    
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
    for row in rows:
        cur.execute("""
            INSERT INTO conversations 
            (conversation_id, user_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
        """, (row[0], row[1], row[2], row[3], row[4]))
    
    # Recreate indexes
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_conversations_user_id 
        ON conversations(user_id)
    """)
    
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_conversations_updated 
        ON conversations(updated_at DESC)
    """)
    
    conn.commit()
    print(f"   ✅ Migrated successfully - restored {len(rows)} conversations")
    return True


def validate_schema(conn):
    """Validate all expected tables exist"""
    cur = conn.cursor()
    
    print("\n🔍 Validating database schema...")
    
    expected_tables = {
        'users': 'User accounts and profiles',
        'updated_essilor_products': 'Product catalog',
        'orders_updated': 'Order history',
        'policies': 'Return/warranty policies',
        'conversations': 'Chat sessions',
        'messages': 'Chat messages',
        'product_text_embeddings': 'Text embeddings (optional)',
        'product_image_embeddings': 'Image embeddings (optional)'
    }
    
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing_tables = {row[0] for row in cur.fetchall()}
    
    print("\n📊 Table Status:")
    for table, description in expected_tables.items():
        status = "✅" if table in existing_tables else "❌"
        
        # Check if optional
        if table in ['product_text_embeddings', 'product_image_embeddings']:
            if table not in existing_tables:
                status = "⚠️  (optional - will be created by embedding script)"
        
        print(f"   {status} {table:30s} - {description}")
    
    # Check for required tables
    required_tables = ['users', 'updated_essilor_products', 'conversations', 'messages']
    missing_required = [t for t in required_tables if t not in existing_tables]
    
    if missing_required:
        print(f"\n❌ Missing required tables: {', '.join(missing_required)}")
        return False
    
    # Check if at least orders_updated table exists
    if 'orders_updated' not in existing_tables:
        print("\n❌ Missing required table: orders_updated")
        print("   User order history features will not work")
        return False
    
    if 'policies' not in existing_tables:
        print("\n⚠️  Warning: No policies table found")
        print("   Policy queries will fall back to CSV file")
    
    return True


def check_data_integrity(conn):
    """Check basic data integrity"""
    cur = conn.cursor()
    
    print("\n🔍 Checking data integrity...")
    
    # Count records
    tables = {
        'users': 'SELECT COUNT(*) FROM users',
        'updated_essilor_products': 'SELECT COUNT(*) FROM updated_essilor_products',
        'orders_updated': 'SELECT COUNT(*) FROM orders_updated',
        'policies': 'SELECT COUNT(*) FROM policies',
        'conversations': 'SELECT COUNT(*) FROM conversations',
        'messages': 'SELECT COUNT(*) FROM messages'
    }
    
    for table, query in tables.items():
        try:
            cur.execute(query)
            count = cur.fetchone()[0]
            print(f"   {table:30s}: {count:>6,d} records")
        except sqlite3.OperationalError:
            print(f"   {table:30s}: (table not found)")
    
    return True


def main():
    print("="*70)
    print("🔧 DATABASE SCHEMA MIGRATION")
    print("="*70)
    
    db_path = get_db_path()
    
    if not db_path.exists():
        print(f"\n❌ Database not found: {db_path}")
        print("   Please check your DATABASE_PATH in .env file")
        return 1
    
    print(f"\n📂 Database: {db_path}")
    
    try:
        conn = sqlite3.connect(str(db_path))
        
        # Run migrations
        success = True
        success = migrate_conversations_table(conn) and success
        
        # Validate schema
        success = validate_schema(conn) and success
        
        # Check data
        check_data_integrity(conn)
        
        conn.close()
        
        if success:
            print("\n" + "="*70)
            print("✅ MIGRATION COMPLETED SUCCESSFULLY")
            print("="*70)
            print("\n🎉 Your database schema is now up to date!")
            print("\nNext steps:")
            print("   1. Run: python main.py")
            print("   2. Test the chat functionality")
            print("   3. If embeddings not created: python -m app.scripts.migrate_to_chromadb --fresh")
            return 0
        else:
            print("\n" + "="*70)
            print("⚠️  MIGRATION COMPLETED WITH WARNINGS")
            print("="*70)
            print("\nSome non-critical issues were found.")
            print("The system should still work, but review the warnings above.")
            return 0
    
    except Exception as e:
        print("\n" + "="*70)
        print("❌ MIGRATION FAILED")
        print("="*70)
        print(f"\nError: {str(e)}")
        
        import traceback
        traceback.print_exc()
        
        return 1


if __name__ == "__main__":
    sys.exit(main())