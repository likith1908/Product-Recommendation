"""
Migration script to move embeddings from SQLite to ChromaDB.

Usage:
    # Migrate existing embeddings from SQLite
    python -m app.scripts.migrate_to_chromadb
    
    # Start fresh (re-compute all embeddings)
    python -m app.scripts.migrate_to_chromadb --fresh
    
    # Use custom batch size
    python -m app.scripts.migrate_to_chromadb --fresh --batch-size=16
    
    # Verify migration without changes
    python -m app.scripts.migrate_to_chromadb --verify-only
"""

import sys
import argparse
import sqlite3
import numpy as np
from typing import Optional
from app.core.database import get_db_connection
from app.core.config import settings
from app.services.embedding_service import get_embedding_service


def check_sqlite_tables() -> dict:
    """Check if SQLite embedding tables exist and get counts."""
    with get_db_connection() as conn:
        cur = conn.cursor()
        
        # Check for old embedding tables
        cur.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name IN ('product_text_embeddings', 'product_image_embeddings')
        """)
        tables = {row[0] for row in cur.fetchall()}
        
        counts = {}
        
        if 'product_text_embeddings' in tables:
            cur.execute("SELECT COUNT(*) FROM product_text_embeddings")
            counts['text'] = cur.fetchone()[0]
        else:
            counts['text'] = 0
        
        if 'product_image_embeddings' in tables:
            cur.execute("SELECT COUNT(*) FROM product_image_embeddings")
            counts['image'] = cur.fetchone()[0]
        else:
            counts['image'] = 0
        
        # Check products table
        cur.execute("""
            SELECT COUNT(*) FROM updated_essilor_products
        """)
        counts['products'] = cur.fetchone()[0]
        
        return {
            'has_text_table': 'product_text_embeddings' in tables,
            'has_image_table': 'product_image_embeddings' in tables,
            'counts': counts
        }


def verify_chromadb_setup() -> dict:
    """Verify ChromaDB setup and get current counts."""
    try:
        embedding_service = get_embedding_service()
        stats = embedding_service.get_collection_stats()
        
        return {
            'initialized': True,
            'text_count': stats['text_collection']['count'],
            'image_count': stats['image_collection']['count'],
            'persist_dir': stats['persist_directory']
        }
    except Exception as e:
        return {
            'initialized': False,
            'error': str(e)
        }


def migrate_text_embeddings(embedding_service, batch_size: int = 100) -> int:
    """
    Migrate text embeddings from SQLite to ChromaDB.
    Returns the number of embeddings migrated.
    """
    print("\n1️⃣  Migrating text embeddings...")
    
    with get_db_connection() as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        # Get all text embeddings with product data
        cur.execute("""
            SELECT te.product_id, te.embedding, p.*
            FROM product_text_embeddings te
            JOIN updated_essilor_products p ON te.product_id = p."Product ID"
        """)
        
        rows = cur.fetchall()
        total = len(rows)
        
        if total == 0:
            print("   ⚠️  No text embeddings found in SQLite")
            return 0
        
        print(f"   Found {total} text embeddings to migrate")
        
        migrated = 0
        
        # Process in batches
        for i in range(0, total, batch_size):
            batch = rows[i:i+batch_size]
            
            ids = []
            embeddings = []
            documents = []
            metadatas = []
            
            for row in batch:
                product_dict = dict(row)
                product_id = product_dict['product_id']
                
                # Extract embedding from BLOB
                embedding_bytes = product_dict['embedding']
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                
                # Get product text representation
                text = embedding_service._get_product_text(product_dict)
                
                # Get metadata
                metadata = embedding_service._product_to_metadata(product_dict)
                
                ids.append(product_id)
                embeddings.append(embedding.tolist())
                documents.append(text)
                metadatas.append(metadata)
            
            # Add batch to ChromaDB
            try:
                embedding_service.text_collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
                migrated += len(batch)
                print(f"   ✓ Progress: {migrated}/{total} text embeddings")
            except Exception as e:
                print(f"   ❌ Error in batch {i//batch_size + 1}: {str(e)}")
                # Continue with next batch
        
        print(f"   ✅ Migrated {migrated}/{total} text embeddings")
        return migrated


def migrate_image_embeddings(embedding_service, batch_size: int = 100) -> int:
    """
    Migrate image embeddings from SQLite to ChromaDB.
    Returns the number of embeddings migrated.
    """
    print("\n2️⃣  Migrating image embeddings...")
    
    with get_db_connection() as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        # Get all image embeddings with product data
        cur.execute("""
            SELECT ie.product_id, ie.embedding, p.*
            FROM product_image_embeddings ie
            JOIN updated_essilor_products p ON ie.product_id = p."Product ID"
        """)
        
        rows = cur.fetchall()
        total = len(rows)
        
        if total == 0:
            print("   ⚠️  No image embeddings found in SQLite")
            return 0
        
        print(f"   Found {total} image embeddings to migrate")
        
        migrated = 0
        
        # Process in batches
        for i in range(0, total, batch_size):
            batch = rows[i:i+batch_size]
            
            ids = []
            embeddings = []
            metadatas = []
            
            for row in batch:
                product_dict = dict(row)
                product_id = product_dict['product_id']
                
                # Extract embedding from BLOB
                embedding_bytes = product_dict['embedding']
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                
                # Get metadata
                metadata = embedding_service._product_to_metadata(product_dict)
                
                ids.append(product_id)
                embeddings.append(embedding.tolist())
                metadatas.append(metadata)
            
            # Add batch to ChromaDB
            try:
                embedding_service.image_collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                migrated += len(batch)
                print(f"   ✓ Progress: {migrated}/{total} image embeddings")
            except Exception as e:
                print(f"   ❌ Error in batch {i//batch_size + 1}: {str(e)}")
                # Continue with next batch
        
        print(f"   ✅ Migrated {migrated}/{total} image embeddings")
        return migrated


def migrate_existing_embeddings(batch_size: int = 100) -> bool:
    """
    Main migration function: move embeddings from SQLite to ChromaDB.
    Returns True if successful, False otherwise.
    """
    print("="*70)
    print("🔄 MIGRATING EMBEDDINGS FROM SQLITE TO CHROMADB")
    print("="*70)
    
    # Check SQLite tables
    sqlite_info = check_sqlite_tables()
    
    if not sqlite_info['has_text_table'] and not sqlite_info['has_image_table']:
        print("\n❌ No existing embedding tables found in SQLite")
        print("   Tables checked: product_text_embeddings, product_image_embeddings")
        print("\n💡 Use --fresh flag to create new embeddings from scratch")
        return False
    
    print(f"\n📊 SQLite Status:")
    print(f"   Products: {sqlite_info['counts']['products']}")
    print(f"   Text embeddings: {sqlite_info['counts']['text']}")
    print(f"   Image embeddings: {sqlite_info['counts']['image']}")
    
    # Initialize ChromaDB
    print("\n🔧 Initializing ChromaDB...")
    embedding_service = get_embedding_service()
    
    # Check if ChromaDB already has data
    chroma_info = verify_chromadb_setup()
    if chroma_info['text_count'] > 0 or chroma_info['image_count'] > 0:
        print(f"\n⚠️  ChromaDB already contains data:")
        print(f"   Text embeddings: {chroma_info['text_count']}")
        print(f"   Image embeddings: {chroma_info['image_count']}")
        
        response = input("\n   Clear existing ChromaDB data? (yes/no): ")
        if response.lower() != 'yes':
            print("   Migration cancelled")
            return False
        
        print("   🗑️  Clearing ChromaDB collections...")
        embedding_service.chroma_client.delete_collection(settings.CHROMA_TEXT_COLLECTION)
        embedding_service.chroma_client.delete_collection(settings.CHROMA_IMAGE_COLLECTION)
        
        # Recreate collections
        embedding_service.text_collection = embedding_service.chroma_client.create_collection(
            name=settings.CHROMA_TEXT_COLLECTION,
            metadata={"description": "Product text embeddings using Sentence Transformers"}
        )
        embedding_service.image_collection = embedding_service.chroma_client.create_collection(
            name=settings.CHROMA_IMAGE_COLLECTION,
            metadata={"description": "Product image embeddings using CLIP"}
        )
    
    # Migrate embeddings
    text_migrated = 0
    image_migrated = 0
    
    if sqlite_info['has_text_table']:
        text_migrated = migrate_text_embeddings(embedding_service, batch_size)
    
    if sqlite_info['has_image_table']:
        image_migrated = migrate_image_embeddings(embedding_service, batch_size)
    
    # Verify migration
    print("\n3️⃣  Verifying migration...")
    final_stats = embedding_service.get_collection_stats()
    
    print(f"\n📊 Final ChromaDB Status:")
    print(f"   Text embeddings: {final_stats['text_collection']['count']}")
    print(f"   Image embeddings: {final_stats['image_collection']['count']}")
    print(f"   Storage location: {final_stats['persist_directory']}")
    
    success = (
        final_stats['text_collection']['count'] >= text_migrated and
        final_stats['image_collection']['count'] >= image_migrated
    )
    
    if success:
        print("\n" + "="*70)
        print("✅ MIGRATION COMPLETED SUCCESSFULLY")
        print("="*70)
        
        print("\n🎉 Next Steps:")
        print("   1. Start your API server: python main.py")
        print("   2. Test the search functionality")
        print("   3. (Optional) Clean up old SQLite tables")
        print("\n💡 To clean up SQLite tables, run:")
        print("   sqlite3 your_database.db")
        print("   DROP TABLE IF EXISTS product_text_embeddings;")
        print("   DROP TABLE IF EXISTS product_image_embeddings;")
        print("   VACUUM;")
        
        return True
    else:
        print("\n⚠️  Migration completed with warnings")
        print("   Please verify the data manually")
        return False


def create_fresh_embeddings(batch_size: int = 32) -> bool:
    """
    Create fresh embeddings from scratch (bypassing SQLite migration).
    Returns True if successful.
    """
    print("="*70)
    print("🆕 CREATING FRESH EMBEDDINGS IN CHROMADB")
    print("="*70)
    
    # Check products
    sqlite_info = check_sqlite_tables()
    print(f"\n📦 Found {sqlite_info['counts']['products']} products in database")
    
    if sqlite_info['counts']['products'] == 0:
        print("❌ No products found in database!")
        print("   Please ensure 'updated_essilor_products' table has data")
        return False
    
    # Initialize embedding service
    print("\n🔧 Loading embedding models and initializing ChromaDB...")
    embedding_service = get_embedding_service()
    
    # Check existing data
    chroma_info = verify_chromadb_setup()
    if chroma_info['text_count'] > 0 or chroma_info['image_count'] > 0:
        print(f"\n⚠️  ChromaDB already contains embeddings:")
        print(f"   Text: {chroma_info['text_count']}, Image: {chroma_info['image_count']}")
        
        response = input("\n   Clear and recreate all embeddings? (yes/no): ")
        if response.lower() != 'yes':
            print("   Operation cancelled")
            return False
    
    # Create embeddings
    print(f"\n🔄 Creating embeddings (batch size: {batch_size})...")
    print("   This may take several minutes depending on:")
    print("   - Number of products")
    print("   - Network speed (for downloading images)")
    print("   - CPU/GPU performance")
    print()
    
    try:
        embedding_service.embed_all_products(
            force_refresh=True,
            batch_size=batch_size
        )
        
        # Verify
        stats = embedding_service.get_collection_stats()
        
        print("\n" + "="*70)
        print("✅ FRESH EMBEDDINGS CREATED SUCCESSFULLY")
        print("="*70)
        
        print(f"\n📊 Final Statistics:")
        print(f"   Text embeddings: {stats['text_collection']['count']}")
        print(f"   Image embeddings: {stats['image_collection']['count']}")
        print(f"   Storage: {stats['persist_directory']}")
        
        print("\n🎉 Your system is ready!")
        print("   You can now start the API server: python main.py")
        
        return True
    
    except Exception as e:
        print(f"\n❌ Error creating embeddings: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def verify_only() -> bool:
    """
    Verify the current state without making changes.
    """
    print("="*70)
    print("🔍 VERIFICATION MODE - No changes will be made")
    print("="*70)
    
    # Check SQLite
    print("\n1️⃣  Checking SQLite database...")
    sqlite_info = check_sqlite_tables()
    
    print(f"\n   SQLite Status:")
    print(f"   ├── Products table: {'✅ Found' if sqlite_info['counts']['products'] > 0 else '❌ Empty'}")
    print(f"   ├── Products count: {sqlite_info['counts']['products']}")
    print(f"   ├── Text embeddings table: {'✅ Exists' if sqlite_info['has_text_table'] else '❌ Not found'}")
    print(f"   │   └── Count: {sqlite_info['counts']['text']}")
    print(f"   └── Image embeddings table: {'✅ Exists' if sqlite_info['has_image_table'] else '❌ Not found'}")
    print(f"       └── Count: {sqlite_info['counts']['image']}")
    
    # Check ChromaDB
    print("\n2️⃣  Checking ChromaDB...")
    chroma_info = verify_chromadb_setup()
    
    if chroma_info['initialized']:
        print(f"\n   ChromaDB Status:")
        print(f"   ├── Initialized: ✅")
        print(f"   ├── Location: {chroma_info['persist_dir']}")
        print(f"   ├── Text collection: {chroma_info['text_count']} embeddings")
        print(f"   └── Image collection: {chroma_info['image_count']} embeddings")
    else:
        print(f"\n   ChromaDB Status:")
        print(f"   ├── Initialized: ❌")
        print(f"   └── Error: {chroma_info.get('error', 'Unknown')}")
    
    # Recommendations
    print("\n3️⃣  Recommendations:")
    
    if not chroma_info['initialized']:
        print("   ❌ ChromaDB is not initialized")
        print("   → Run: python -m app.scripts.migrate_to_chromadb --fresh")
    
    elif chroma_info['text_count'] == 0 and chroma_info['image_count'] == 0:
        if sqlite_info['has_text_table'] or sqlite_info['has_image_table']:
            print("   💡 You have SQLite embeddings but ChromaDB is empty")
            print("   → Run: python -m app.scripts.migrate_to_chromadb")
        else:
            print("   💡 No embeddings found anywhere")
            print("   → Run: python -m app.scripts.migrate_to_chromadb --fresh")
    
    elif chroma_info['text_count'] < sqlite_info['counts']['products']:
        print(f"   ⚠️  ChromaDB has fewer embeddings than products")
        print(f"      ({chroma_info['text_count']} embeddings vs {sqlite_info['counts']['products']} products)")
        print("   → Consider running: python -m app.scripts.migrate_to_chromadb --fresh")
    
    else:
        print("   ✅ Everything looks good!")
        print("   → Your system is ready to use")
    
    print("\n" + "="*70)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Migrate product embeddings from SQLite to ChromaDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Migrate existing embeddings from SQLite
  python -m app.scripts.migrate_to_chromadb
  
  # Create fresh embeddings (recommended for first-time setup)
  python -m app.scripts.migrate_to_chromadb --fresh
  
  # Create fresh embeddings with smaller batches (if memory constrained)
  python -m app.scripts.migrate_to_chromadb --fresh --batch-size=16
  
  # Verify current state without making changes
  python -m app.scripts.migrate_to_chromadb --verify-only
        """
    )
    
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Skip migration and create fresh embeddings from scratch (recommended)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing products (default: 32, reduce if out of memory)"
    )
    
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify current state without making any changes"
    )
    
    args = parser.parse_args()
    
    try:
        if args.verify_only:
            success = verify_only()
        elif args.fresh:
            success = create_fresh_embeddings(batch_size=args.batch_size)
        else:
            success = migrate_existing_embeddings(batch_size=args.batch_size)
        
        return 0 if success else 1
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Migration interrupted by user")
        return 1
    
    except Exception as e:
        print("\n" + "="*70)
        print("❌ MIGRATION FAILED")
        print("="*70)
        print(f"\nError: {str(e)}")
        
        print("\n📝 Troubleshooting:")
        print("   1. Check database path in .env: DATABASE_PATH")
        print("   2. Ensure products table exists: updated_essilor_products")
        print("   3. Verify ChromaDB directory is writable: CHROMA_PERSIST_DIR")
        print("   4. Check network connection (for image downloads)")
        print("   5. Try with smaller batch size: --batch-size=16")
        
        import traceback
        print("\n🔍 Full traceback:")
        traceback.print_exc()
        
        return 1


if __name__ == "__main__":
    sys.exit(main())