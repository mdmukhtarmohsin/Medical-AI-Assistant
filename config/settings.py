"""
Configuration settings for the Medical AI Assistant application.
"""
import os
from pydantic import BaseModel
from typing import Optional


class Settings(BaseModel):
    """Application settings loaded from environment variables."""
    
    # Application settings
    APP_NAME: str = "Medical AI Assistant"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Server settings
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    
    # Google Gemini API settings
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-2.5-pro"
    GEMINI_TEMPERATURE: float = 0.1
    GEMINI_MAX_TOKENS: int = 4096
    
    # File upload settings
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: list = [".pdf"]
    UPLOAD_DIR: str = "uploads"
    
    # Vector store settings
    VECTOR_STORE_DIR: str = "vector_store"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Database settings (for future use if needed)
    DATABASE_URL: Optional[str] = None
    
    # Evaluation settings
    EVALUATION_DATASET_SIZE: int = 100
    SIMILARITY_THRESHOLD: float = 0.7
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()

# Ensure required directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.VECTOR_STORE_DIR, exist_ok=True) 