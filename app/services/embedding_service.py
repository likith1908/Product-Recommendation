"""
Product embedding service using Sentence Transformers, CLIP, and ChromaDB.
Handles text and image embeddings for product recommendations.
"""

import sqlite3
import numpy as np
from typing import List, Dict, Optional
import requests
from io import BytesIO
from PIL import Image

import torch
from sentence_transformers import SentenceTransformer
import open_clip
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings
from app.core.database import get_db_connection


class ProductEmbeddingService:
    """
    Service for creating and searching product embeddings using:
    - Sentence Transformers for text embeddings (384D)
    - CLIP for image embeddings (512D)
    - ChromaDB for vector storage and similarity search
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern to avoid reloading models"""
        if cls._instance is None:
            cls._instance = super(ProductEmbeddingService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize embedding models and ChromaDB (only once)"""
        if not self._initialized:
            print("🔧 Initializing embedding models and ChromaDB...")
            
            # Load Sentence Transformer for text
            print(f"📝 Loading {settings.SENTENCE_TRANSFORMER_MODEL}...")
            self.text_model = SentenceTransformer(settings.SENTENCE_TRANSFORMER_MODEL)
            
            # Load CLIP for images
            print(f"🖼️  Loading CLIP {settings.CLIP_MODEL_NAME}...")
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                settings.CLIP_MODEL_NAME,
                pretrained=settings.CLIP_PRETRAINED
            )
            self.clip_tokenizer = open_clip.get_tokenizer(settings.CLIP_MODEL_NAME)
            
            # Set to evaluation mode
            self.clip_model.eval()
            
            # Move to GPU if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model = self.clip_model.to(self.device)
            
            # Initialize ChromaDB
            print(f"🗄️  Initializing ChromaDB at {settings.CHROMA_PERSIST_DIR}...")
            self.chroma_client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIR,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collections
            self.text_collection = self.chroma_client.get_or_create_collection(
                name=settings.CHROMA_TEXT_COLLECTION,
                metadata={"description": "Product text embeddings using Sentence Transformers"}
            )
            
            self.image_collection = self.chroma_client.get_or_create_collection(
                name=settings.CHROMA_IMAGE_COLLECTION,
                metadata={"description": "Product image embeddings using CLIP"}
            )
            
            self._initialized = True
            print(f"✅ Models and ChromaDB loaded successfully (device: {self.device})")
            print(f"   Text collection: {self.text_collection.count()} embeddings")
            print(f"   Image collection: {self.image_collection.count()} embeddings")
    
    def _get_product_text(self, product: Dict) -> str:
        """
        Create a rich text representation of the product for embedding.
        Combines all relevant textual fields.
        """
        text_parts = []
        
        # Core product info
        if product.get('Product Name'):
            text_parts.append(product['Product Name'])
        
        if product.get('Brand Name'):
            text_parts.append(f"by {product['Brand Name']}")
        
        # Type and style
        style_parts = []
        if product.get('Frame Type'):
            style_parts.append(product['Frame Type'])
        if product.get('Frame Shape'):
            style_parts.append(product['Frame Shape'])
        if product.get('Product Type'):
            style_parts.append(product['Product Type'])
        if style_parts:
            text_parts.append(" ".join(style_parts))
        
        # Physical attributes
        if product.get('Frame Colour'):
            text_parts.append(f"{product['Frame Colour']} color")
        
        if product.get('Frame Material'):
            text_parts.append(f"made from {product['Frame Material']}")
        
        if product.get('Frame Size'):
            text_parts.append(f"{product['Frame Size']} size")
        
        # Suitability
        if product.get('Face Shape'):
            text_parts.append(f"suitable for {product['Face Shape']} face")
        
        if product.get('Activity'):
            text_parts.append(f"ideal for {product['Activity']}")
        
        # Description
        if product.get('description'):
            text_parts.append(product['description'])
        
        # Price context
        if product.get('Price'):
            try:
                price = float(product['Price'])
                if price < 1000:
                    text_parts.append("affordable budget option")
                elif price < 3000:
                    text_parts.append("mid-range pricing")
                else:
                    text_parts.append("premium quality")
            except (ValueError, TypeError):
                pass
        
        return ". ".join(text_parts)
    
    def _product_to_metadata(self, product: Dict) -> Dict:
        """Convert product dict to ChromaDB metadata (strings, ints, floats only)"""
        metadata = {}
        
        # Include key fields that might be used for filtering
        safe_fields = [
            'Product Name', 'Brand Name', 'Frame Type', 'Frame Shape',
            'Frame Colour', 'Frame Material', 'Frame Size', 'Product Type',
            'Face Shape', 'Activity', 'Image URL', 'description'
        ]
        
        for field in safe_fields:
            value = product.get(field)
            if value is not None:
                # Convert to string to ensure compatibility
                metadata[field] = str(value)
        
        # Handle numeric fields
        if product.get('Price'):
            try:
                metadata['Price'] = float(product['Price'])
            except (ValueError, TypeError):
                pass
        
        if product.get('Rating'):
            try:
                metadata['Rating'] = float(product['Rating'])
            except (ValueError, TypeError):
                pass
        
        if product.get('Number of Reviews'):
            try:
                metadata['Number of Reviews'] = int(product['Number of Reviews'])
            except (ValueError, TypeError):
                pass
        
        return metadata
    
    def create_text_embedding(self, text: str) -> np.ndarray:
        """Create Sentence Transformer embedding for text (384D)"""
        embedding = self.text_model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)
    
    def create_image_embedding(self, image_url: str) -> Optional[np.ndarray]:
        """
        Create CLIP embedding for an image (512D).
        Downloads the image and processes it with CLIP.
        """
        try:
            # Download image
            response = requests.get(image_url, timeout=15)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            
            # Preprocess and encode
            image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_tensor)
                # Normalize
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy()[0].astype(np.float32)
        
        except Exception as e:
            print(f"❌ Error creating image embedding for {image_url}: {e}")
            return None
    
    def create_text_query_with_clip(self, query: str) -> np.ndarray:
        """
        Create CLIP embedding for a text query (512D).
        This allows text-to-image search.
        """
        text_tokens = self.clip_tokenizer([query]).to(self.device)
        
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            # Normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()[0].astype(np.float32)
    
    def embed_all_products(self, force_refresh: bool = False, batch_size: int = 32):
        """
        Embed all products in the database and store in ChromaDB.
        Creates both text and image embeddings.
        
        Args:
            force_refresh: If True, clear and re-embed all products
            batch_size: Number of products to process in each batch
        """
        if force_refresh:
            print("🗑️  Clearing existing embeddings...")
            self.chroma_client.delete_collection(settings.CHROMA_TEXT_COLLECTION)
            self.chroma_client.delete_collection(settings.CHROMA_IMAGE_COLLECTION)
            self.text_collection = self.chroma_client.create_collection(
                name=settings.CHROMA_TEXT_COLLECTION,
                metadata={"description": "Product text embeddings using Sentence Transformers"}
            )
            self.image_collection = self.chroma_client.create_collection(
                name=settings.CHROMA_IMAGE_COLLECTION,
                metadata={"description": "Product image embeddings using CLIP"}
            )
        
        with get_db_connection() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            
            # Get all products
            cur.execute('SELECT * FROM updated_essilor_products')
            products = cur.fetchall()
            
            print(f"\n📦 Found {len(products)} products to embed")
            
            # Get existing product IDs to skip
            existing_text_ids = set()
            existing_image_ids = set()
            
            if not force_refresh:
                try:
                    existing_text = self.text_collection.get(include=[])
                    existing_text_ids = set(existing_text['ids'])
                    
                    existing_image = self.image_collection.get(include=[])
                    existing_image_ids = set(existing_image['ids'])
                    
                    print(f"📊 Already embedded: {len(existing_text_ids)} text, {len(existing_image_ids)} image")
                except:
                    pass
            
            # Collect products to embed
            products_to_embed = []
            
            for idx, product in enumerate(products):
                product_dict = dict(product)
                product_id = product_dict.get('Product ID')
                
                if not product_id:
                    print(f"⚠️  Skipping product at index {idx} (no Product ID)")
                    continue
                
                # Check if already embedded
                if not force_refresh and product_id in existing_text_ids:
                    continue
                
                products_to_embed.append((idx, product_dict))
            
            if not products_to_embed:
                print("✅ All products already embedded!")
                return
            
            print(f"🔄 Embedding {len(products_to_embed)} products...")
            
            # Process in batches
            for i in range(0, len(products_to_embed), batch_size):
                batch = products_to_embed[i:i+batch_size]
                
                # Prepare data for batch
                batch_ids = []
                batch_texts = []
                batch_text_embeddings = []
                batch_metadata = []
                
                batch_image_ids = []
                batch_image_embeddings = []
                batch_image_metadata = []
                
                for idx, product_dict in batch:
                    product_id = product_dict['Product ID']
                    
                    # Text embedding
                    text = self._get_product_text(product_dict)
                    text_embedding = self.create_text_embedding(text)
                    
                    batch_ids.append(product_id)
                    batch_texts.append(text)
                    batch_text_embeddings.append(text_embedding.tolist())
                    batch_metadata.append(self._product_to_metadata(product_dict))
                    
                    # Image embedding
                    image_url = product_dict.get('Image URL')
                    if image_url:
                        image_embedding = self.create_image_embedding(image_url)
                        if image_embedding is not None:
                            batch_image_ids.append(product_id)
                            batch_image_embeddings.append(image_embedding.tolist())
                            batch_image_metadata.append(self._product_to_metadata(product_dict))
                    
                    print(f"  [{idx+1}/{len(products)}] ✓ {product_id}")
                
                # Add batch to ChromaDB
                if batch_ids:
                    self.text_collection.add(
                        ids=batch_ids,
                        embeddings=batch_text_embeddings,
                        documents=batch_texts,
                        metadatas=batch_metadata
                    )
                
                if batch_image_ids:
                    self.image_collection.add(
                        ids=batch_image_ids,
                        embeddings=batch_image_embeddings,
                        metadatas=batch_image_metadata
                    )
            
            print("✅ Embedding complete!")
            print(f"   Text embeddings: {self.text_collection.count()}")
            print(f"   Image embeddings: {self.image_collection.count()}")
    
    def search_by_text(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search products by text query using Sentence Transformer embeddings.
        
        Args:
            query: Text query from user
            top_k: Number of results to return
            filters: Optional filters like {"Brand Name": "John Jacobs"}
        """
        # Create query embedding
        query_embedding = self.create_text_embedding(query)
        
        # Build where clause for filters
        where = None
        if filters:
            where = {"$and": []}
            for key, value in filters.items():
                where["$and"].append({key: {"$eq": value}})
        
        # Query ChromaDB
        results = self.text_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where,
            include=["metadatas", "distances", "documents"]
        )
        
        if not results['ids'] or not results['ids'][0]:
            return []
        
        # Format results
        formatted_results = []
        for i, product_id in enumerate(results['ids'][0]):
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            
            # Convert distance to similarity (ChromaDB uses L2 distance by default)
            # For normalized vectors: similarity = 1 - (distance^2 / 2)
            similarity = 1 - (distance ** 2 / 2)
            
            result = {
                'Product ID': product_id,
                'similarity_score': max(0, min(1, similarity))  # Clamp to [0, 1]
            }
            result.update(metadata)
            formatted_results.append(result)
        
        return formatted_results
    
    def search_by_image_and_text(
        self,
        image_url: Optional[str] = None,
        text_query: Optional[str] = None,
        top_k: int = 5,
        text_weight: float = 0.5
    ) -> List[Dict]:
        """
        Multimodal search: combine image and text queries using CLIP.
        
        Args:
            image_url: URL of reference image
            text_query: Text description (e.g., "similar but in blue color")
            top_k: Number of results to return
            text_weight: Weight for text vs image (0-1, where 0.5 is equal)
        """
        if not image_url and not text_query:
            raise ValueError("Must provide either image_url or text_query")
        
        query_embedding = None
        
        # Create combined embedding
        if image_url and text_query:
            # Hybrid: combine image and text embeddings
            img_emb = self.create_image_embedding(image_url)
            txt_emb = self.create_text_query_with_clip(text_query)
            
            if img_emb is None:
                query_embedding = txt_emb
            else:
                # Weighted combination
                query_embedding = (1 - text_weight) * img_emb + text_weight * txt_emb
                # Re-normalize
                query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        elif image_url:
            # Image only
            query_embedding = self.create_image_embedding(image_url)
        
        else:
            # Text only (using CLIP text encoder)
            query_embedding = self.create_text_query_with_clip(text_query)
        
        if query_embedding is None:
            return []
        
        # Query ChromaDB image collection
        results = self.image_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["metadatas", "distances"]
        )
        
        if not results['ids'] or not results['ids'][0]:
            return []
        
        # Format results
        formatted_results = []
        for i, product_id in enumerate(results['ids'][0]):
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            
            # Convert distance to similarity
            similarity = 1 - (distance ** 2 / 2)
            
            result = {
                'Product ID': product_id,
                'similarity_score': max(0, min(1, similarity))
            }
            result.update(metadata)
            formatted_results.append(result)
        
        return formatted_results
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the ChromaDB collections"""
        return {
            "text_collection": {
                "name": settings.CHROMA_TEXT_COLLECTION,
                "count": self.text_collection.count()
            },
            "image_collection": {
                "name": settings.CHROMA_IMAGE_COLLECTION,
                "count": self.image_collection.count()
            },
            "persist_directory": settings.CHROMA_PERSIST_DIR
        }


# Singleton instance
_embedding_service_instance = None

def get_embedding_service() -> ProductEmbeddingService:
    """Get or create the singleton embedding service instance"""
    global _embedding_service_instance
    if _embedding_service_instance is None:
        _embedding_service_instance = ProductEmbeddingService()
    return _embedding_service_instance