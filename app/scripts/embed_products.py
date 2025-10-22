"""
One-time script to create embeddings for all products in the database.
Run this after setting up the database and before starting the API.

Usage:
    python -m app.scripts.embed_products
    
    # Or with force refresh:
    python -m app.scripts.embed_products --force-refresh
"""

import sys
import argparse
from app.core.database import init_db
from app.services.embedding_service import get_embedding_service


def main():
    parser = argparse.ArgumentParser(
        description="Create embeddings for all products in the database"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force re-embedding of all products (even if they already have embeddings)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing products (default: 32)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 PRODUCT EMBEDDING CREATION")
    print("="*60)
    
    # Initialize database tables
    print("\n1️⃣  Initializing database...")
    init_db()
    print("✅ Database initialized")
    
    # Get embedding service (loads models)
    print("\n2️⃣  Loading embedding models...")
    embedding_service = get_embedding_service()
    print("✅ Models loaded")
    
    # Create embeddings
    print("\n3️⃣  Creating product embeddings...")
    print(f"   Force refresh: {args.force_refresh}")
    print(f"   Batch size: {args.batch_size}")
    print()
    
    try:
        embedding_service.embed_all_products(
            force_refresh=args.force_refresh,
            batch_size=args.batch_size
        )
        
        print("\n" + "="*60)
        print("✅ EMBEDDING CREATION COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\n🎉 Your products are now searchable!")
        print("   You can now start the API server.")
        
        return 0
    
    except Exception as e:
        print("\n" + "="*60)
        print("❌ EMBEDDING CREATION FAILED")
        print("="*60)
        print(f"\nError: {str(e)}")
        print("\nPlease check:")
        print("  - Database path is correct")
        print("  - Products exist in 'updated_essilor_products' table")
        print("  - Image URLs are accessible")
        print("  - You have internet connection for model downloads")
        
        import traceback
        traceback.print_exc()
        
        return 1


if __name__ == "__main__":
    sys.exit(main())