"""
Product embedding service using Sentence Transformers and CLIP.
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

from app.core.config import settings
from app.core.database import get_db_connection


class ProductEmbeddingService:
    """
    Service for creating and searching product embeddings using:
    - Sentence Transformers for text embeddings (384D)
    - CLIP for image embeddings (512D)
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern to avoid reloading models"""
        if cls._instance is None:
            cls._instance = super(ProductEmbeddingService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize embedding models (only once)"""
        if not self._initialized:
            print("🔧 Initializing embedding models...")
            
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
            
            self._initialized = True
            print(f"✅ Models loaded successfully (device: {self.device})")
    
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
        Embed all products in the database.
        Creates both text and image embeddings.
        
        Args:
            force_refresh: If True, re-embed all products
            batch_size: Number of products to process in each batch
        """
        with get_db_connection() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            
            # Get all products
            cur.execute('SELECT * FROM updated_essilor_products')
            products = cur.fetchall()
            
            print(f"\n📦 Found {len(products)} products to embed")
            
            # Collect texts for batch processing
            products_to_embed = []
            
            for idx, product in enumerate(products):
                product_dict = dict(product)
                product_id = product_dict.get('Product ID')
                
                if not product_id:
                    print(f"⚠️  Skipping product at index {idx} (no Product ID)")
                    continue
                
                # Check if embeddings already exist
                if not force_refresh:
                    cur.execute(
                        "SELECT 1 FROM product_text_embeddings WHERE product_id = ?",
                        (product_id,)
                    )
                    if cur.fetchone():
                        continue
                
                products_to_embed.append((idx, product_dict))
            
            if not products_to_embed:
                print("✅ All products already embedded!")
                return
            
            print(f"🔄 Embedding {len(products_to_embed)} products...")
            
            # Process in batches for text embeddings
            for i in range(0, len(products_to_embed), batch_size):
                batch = products_to_embed[i:i+batch_size]
                
                # Prepare texts
                texts = [self._get_product_text(prod[1]) for prod in batch]
                
                # Batch encode texts
                text_embeddings = self.text_model.encode(
                    texts,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                
                # Save text embeddings and process images
                for j, (idx, product_dict) in enumerate(batch):
                    product_id = product_dict['Product ID']
                    
                    # Save text embedding
                    cur.execute("""
                        INSERT OR REPLACE INTO product_text_embeddings 
                        (product_id, embedding, embedding_model)
                        VALUES (?, ?, ?)
                    """, (
                        product_id,
                        text_embeddings[j].astype(np.float32).tobytes(),
                        settings.SENTENCE_TRANSFORMER_MODEL
                    ))
                    
                    # Process image embedding (individual, not batched due to downloads)
                    image_url = product_dict.get('Image URL')
                    if image_url:
                        image_embedding = self.create_image_embedding(image_url)
                        if image_embedding is not None:
                            cur.execute("""
                                INSERT OR REPLACE INTO product_image_embeddings 
                                (product_id, embedding, embedding_model)
                                VALUES (?, ?, ?)
                            """, (
                                product_id,
                                image_embedding.tobytes(),
                                f"{settings.CLIP_MODEL_NAME}-{settings.CLIP_PRETRAINED}"
                            ))
                    
                    print(f"  [{idx+1}/{len(products)}] ✓ {product_id}")
                
                conn.commit()
            
            print("✅ Embedding complete!")
    
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
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
        
        with get_db_connection() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            
            # Build SQL query with optional filters
            sql = """
                SELECT p.*, te.embedding
                FROM updated_essilor_products p
                JOIN product_text_embeddings te ON p."Product ID" = te.product_id
            """
            
            params = []
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(f'p."{key}" = ?')
                    params.append(value)
                sql += " WHERE " + " AND ".join(conditions)
            
            cur.execute(sql, params)
            products = cur.fetchall()
            
            if not products:
                return []
            
            # Calculate similarities
            results = []
            for product in products:
                product_dict = dict(product)
                product_embedding = np.frombuffer(
                    product_dict['embedding'],
                    dtype=np.float32
                )
                
                similarity = self.cosine_similarity(query_embedding, product_embedding)
                product_dict['similarity_score'] = similarity
                
                # Remove embedding from result
                del product_dict['embedding']
                results.append(product_dict)
            
            # Sort by similarity and return top_k
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            return results[:top_k]
    
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
        
        with get_db_connection() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            
            # Get all products with image embeddings
            cur.execute("""
                SELECT p.*, ie.embedding
                FROM updated_essilor_products p
                JOIN product_image_embeddings ie ON p."Product ID" = ie.product_id
            """)
            products = cur.fetchall()
            
            if not products:
                return []
            
            # Calculate similarities
            results = []
            for product in products:
                product_dict = dict(product)
                product_embedding = np.frombuffer(
                    product_dict['embedding'],
                    dtype=np.float32
                )
                
                similarity = self.cosine_similarity(query_embedding, product_embedding)
                product_dict['similarity_score'] = similarity
                
                # Remove embedding from result
                del product_dict['embedding']
                results.append(product_dict)
            
            # Sort by similarity and return top_k
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            return results[:top_k]


# Singleton instance
_embedding_service_instance = None

def get_embedding_service() -> ProductEmbeddingService:
    """Get or create the singleton embedding service instance"""
    global _embedding_service_instance
    if _embedding_service_instance is None:
        _embedding_service_instance = ProductEmbeddingService()
    return _embedding_service_instance