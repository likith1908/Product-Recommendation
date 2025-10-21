import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Settings
    APP_NAME: str = "Authentication API"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    
    # Security
    JWT_SECRET: str = os.environ.get("JWT_SECRET")
    ALGORITHM: str = "HS256"
    # Provide a safe default and conversion for the token expiry minutes.
    # If the environment variable is missing or invalid, fall back to 30 minutes.
    _access_token_expire = os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES")
    try:
        ACCESS_TOKEN_EXPIRE_MINUTES: int = int(_access_token_expire) if _access_token_expire is not None else 30
    except (ValueError, TypeError):
        ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Database
    DATABASE_PATH: str = os.environ.get("DATABASE_PATH")
    
    # CORS
    ALLOWED_ORIGINS: list = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()