import os
from dotenv import load_dotenv

load_dotenv()
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Settings
    APP_NAME: str = "Product Recommendation API"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    
    # Security
    JWT_SECRET: str = os.environ.get("JWT_SECRET")
    ALGORITHM: str = "HS256"
    _access_token_expire = os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES")
    try:
        ACCESS_TOKEN_EXPIRE_MINUTES: int = int(_access_token_expire) if _access_token_expire is not None else 30
    except (ValueError, TypeError):
        ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Database (SQLite for user data only)
    DATABASE_PATH: str = os.environ.get("DATABASE_PATH")
    
    # ChromaDB Settings
    CHROMA_PERSIST_DIR: str = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
    CHROMA_TEXT_COLLECTION: str = "product_text_embeddings"
    CHROMA_IMAGE_COLLECTION: str = "product_image_embeddings"
    
    # CORS
    ALLOWED_ORIGINS: list = ["*"]
    
    # OpenAI
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
    
    # Embedding Models
    SENTENCE_TRANSFORMER_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CLIP_MODEL_NAME: str = "ViT-B-32"
    CLIP_PRETRAINED: str = "laion2b_s34b_b79k"
    
    # Embedding Settings
    TEXT_EMBEDDING_DIM: int = 384  # all-MiniLM-L6-v2 dimension
    IMAGE_EMBEDDING_DIM: int = 512  # CLIP ViT-B-32 dimension
    
    # Google Cloud Storage
    GCS_PROJECT_NAME: str = "prj-auropro-dev"
    GCS_BUCKET_NAME: str = "product-recommendation-chatbot"
    GCS_UPLOAD_FOLDER: str = "user_uploads"
    GOOGLE_APPLICATION_CREDENTIALS: str = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    
    # File Upload Settings
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_IMAGE_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".webp"}
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()