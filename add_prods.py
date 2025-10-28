"""
Simple migration to add products_data column.
Run this from the project root: python migrate_add_products.py
"""

import sqlite3
import os
from pathlib import Path

# Get database path from environment or use default
DATABASE_PATH = os.environ.get("DATABASE_PATH", "sample_data.db")

print("="*70)
print("🔄 ADDING products_data COLUMN TO messages TABLE")
print("="*70)
print(f"\nDatabase: {DATABASE_PATH}")

# Check if database exists
if not Path(DATABASE_PATH).exists():
    print(f"\n❌ Error: Database not found at {DATABASE_PATH}")
    print("   Please check your DATABASE_PATH environment variable")
    exit(1)

try:
    # Connect to database
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    # Check current schema
    print("\n1️⃣  Checking current schema...")
    cur.execute("PRAGMA table_info(messages)")
    columns = [row[1] for row in cur.fetchall()]
    
    print(f"   Current columns: {', '.join(columns)}")
    
    # Check if column already exists
    if 'products_data' in columns:
        print("\n✅ Column 'products_data' already exists!")
        print("   No migration needed.")
        conn.close()
        exit(0)
    
    # Add the column
    print("\n2️⃣  Adding 'products_data' column...")
    cur.execute("""
        ALTER TABLE messages 
        ADD COLUMN products_data TEXT
    """)
    
    conn.commit()
    
    # Verify
    print("\n3️⃣  Verifying...")
    cur.execute("PRAGMA table_info(messages)")
    columns = [row[1] for row in cur.fetchall()]
    
    if 'products_data' in columns:
        print("   ✅ Column added successfully!")
    else:
        print("   ❌ Column not found after adding!")
        exit(1)
    
    conn.close()
    
    print("\n" + "="*70)
    print("✅ MIGRATION COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\n🎉 You can now restart your API server!")
    print("   python main.py")

except sqlite3.OperationalError as e:
    print(f"\n❌ Database error: {e}")
    exit(1)
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)