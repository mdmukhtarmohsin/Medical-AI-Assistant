"""
Configuration settings for the Medical AI Assistant application.
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application settings
    APP_NAME: str = "Medical AI Assistant"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Google Gemini API settings
    GEMINI_API_KEY: Optional[str] = None
    LLM_MODEL: str = "gemini-2.0-flash-exp"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 1024
    
    # File upload settings
    MAX_FILE_SIZE_MB: int = 50
    MAX_UPLOAD_SIZE: int = 52428800  # 50MB in bytes
    ALLOWED_FILE_TYPES: str = "pdf"
    UPLOAD_DIR: str = "uploads"
    
    # Vector store settings
    VECTOR_STORE_DIR: str = "data/vector_store"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    VECTOR_DIMENSION: int = 384
    SIMILARITY_THRESHOLD: float = 0.7
    
    # Document processing settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_CHUNKS_PER_QUERY: int = 5
    
    # Storage paths
    DOCUMENTS_DIR: str = "data/documents"
    FAISS_INDEX_DIR: str = "data/vector_store"
    LOGS_DIR: str = "logs"
    
    # RAGAS evaluation settings
    RAGAS_FAITHFULNESS_THRESHOLD: float = 0.90
    RAGAS_CONTEXT_PRECISION_THRESHOLD: float = 0.85
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()

# Validate required settings
if not settings.GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is required. Please set it in your .env file.")

# Ensure required directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.VECTOR_STORE_DIR, exist_ok=True)
os.makedirs(settings.DOCUMENTS_DIR, exist_ok=True)
os.makedirs(settings.LOGS_DIR, exist_ok=True) 