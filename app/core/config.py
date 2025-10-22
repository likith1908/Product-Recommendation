import os
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
    
    # Database
    DATABASE_PATH: str = os.environ.get("DATABASE_PATH")
    
    # CORS
    ALLOWED_ORIGINS: list = ["*"]
    
    # OpenAI
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY")
    
    # Embedding Models
    SENTENCE_TRANSFORMER_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CLIP_MODEL_NAME: str = "ViT-B-32"
    CLIP_PRETRAINED: str = "laion2b_s34b_b79k"
    
    # Embedding Settings
    TEXT_EMBEDDING_DIM: int = 384  # all-MiniLM-L6-v2 dimension
    IMAGE_EMBEDDING_DIM: int = 512  # CLIP ViT-B-32 dimension
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()