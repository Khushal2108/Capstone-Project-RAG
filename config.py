import os
from dotenv import load_dotenv
from typing import List, Optional
import time

load_dotenv()

class APIKeyManager:
    """Manages multiple Gemini API keys with automatic cycling on failure"""
    
    def __init__(self):
        self.api_keys: List[str] = self._load_api_keys()
        self.current_key_index = 0
        self.failed_keys = set()
        
        if not self.api_keys:
            raise ValueError("No API keys found in .env file. Please add at least one GEMINI_API_KEY_X")
    
    def _load_api_keys(self) -> List[str]:
        """Load all available API keys from environment"""
        keys = []
        for i in range(1, 11):  # Support up to 10 keys
            key = os.getenv(f"GEMINI_API_KEY_{i}")
            if key and key.strip():
                keys.append(key.strip())
        return keys
    
    def get_current_key(self) -> str:
        """Get the current active API key"""
        if len(self.failed_keys) >= len(self.api_keys):
            # All keys failed, reset and try again
            self.failed_keys.clear()
            time.sleep(2)  # Brief pause before retry
        
        return self.api_keys[self.current_key_index]
    
    def mark_key_failed(self):
        """Mark current key as failed and cycle to next"""
        self.failed_keys.add(self.current_key_index)
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        print(f"⚠️ Cycling to API key {self.current_key_index + 1}")
    
    def get_available_keys_count(self) -> int:
        """Get count of available (non-failed) keys"""
        return len(self.api_keys) - len(self.failed_keys)


class Config:
    """Application configuration"""
    
    # API Key Manager
    api_key_manager = APIKeyManager()
    
    # Paths
    DATA_DIR = "data"
    CHROMA_DB_DIR = "chroma_db"
    
    # ChromaDB Collections
    TEXT_COLLECTION = "document_texts"
    IMAGE_COLLECTION = "document_images"
    
    # Chunking Parameters
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Model Configuration
    # embeddings (local, no API limits!)
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Fast and efficient
    
    # Gemini only for vision understanding and LLM
    LLM_MODEL = "gemini-2.5-flash"  # Fast and efficient
    VISION_MODEL = "gemini-2.5-flash"  # For image understanding
    
    # Generation Parameters
    TEMPERATURE = 0.3
    MAX_OUTPUT_TOKENS = 2048
    TOP_K = 5  # Number of documents to retrieve
    
    # Image Processing
    IMAGE_QUALITY = 85
    MAX_IMAGE_SIZE = (1024, 1024)
    
    # Embedding dimensions for CLIP
    EMBEDDING_DIM = 384  # For all-MiniLM-L6-v2
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories"""
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.CHROMA_DB_DIR, exist_ok=True)
