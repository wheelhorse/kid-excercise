"""
Configuration settings for the Resume Retrieval System
"""
import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database Configurations
MARIADB_CONFIG = {
    "host": os.getenv("MARIADB_HOST", "localhost"),
    "port": int(os.getenv("MARIADB_PORT", "3306")),
    "user": os.getenv("MARIADB_USER", "root"),
    "password": os.getenv("MARIADB_PASSWORD", ""),
    "database": os.getenv("MARIADB_DATABASE", "cats_dev"),
    "charset": "utf8mb4"
}

QDRANT_CONFIG = {
    "url": os.getenv("QDRANT_URL", "http://localhost:6333"),
    "api_key": os.getenv("QDRANT_API_KEY", None),
    "timeout": 60.0,
    "prefer_grpc": True
}

# Model Settings
BGE_M3_MODEL = "BAAI/bge-m3"
COLLECTION_NAME = "resume_hybrid_search"
SEARCH_MODES = ["bge-m3", "hybrid"]
DEFAULT_TOP_K = 100
BATCH_SIZE = 32

# Vector Configuration
DENSE_VECTOR_SIZE = 1024  # BGE-M3 embedding dimension
DISTANCE_METRIC = "Cosine"

# Text Processing
MAX_TEXT_LENGTH = 8192  # Maximum text length for embedding
CHINESE_TOKENIZER = "jieba"
ENABLE_TRADITIONAL_CHINESE = True

# Sync Configuration
SYNC_BATCH_SIZE = 100
SYNC_INTERVAL_SECONDS = 300  # 5 minutes for auto-sync
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 5

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = "hybrid_search.log"

# Performance Settings
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")  # or "cuda"
ENABLE_CACHING = True
CACHE_SIZE = 1000

# Search Configuration
SIMILARITY_THRESHOLD = 0.3
RRF_K = 60  # Reciprocal Rank Fusion parameter
DENSE_WEIGHT = 0.7  # Weight for dense embeddings in hybrid search
SPARSE_WEIGHT = 0.3  # Weight for sparse embeddings in hybrid search

# Configuration Objects
SEARCH_CONFIG = {
    "collection_name": COLLECTION_NAME,
    "default_top_k": DEFAULT_TOP_K,
    "similarity_threshold": SIMILARITY_THRESHOLD,
    "dense_weight": DENSE_WEIGHT,
    "sparse_weight": SPARSE_WEIGHT,
    "rrf_k": RRF_K
}

SYNC_CONFIG = {
    "batch_size": SYNC_BATCH_SIZE,
    "interval_seconds": SYNC_INTERVAL_SECONDS,
    "max_retry_attempts": MAX_RETRY_ATTEMPTS,
    "retry_delay_seconds": RETRY_DELAY_SECONDS
}

def validate_config() -> bool:
    """Validate configuration settings"""
    required_env_vars = [
        "QDRANT_URL",
        "MARIADB_HOST", 
        "MARIADB_USER",
        "MARIADB_DATABASE"
    ]
    
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"Missing required environment variables: {missing_vars}")
        return False
    
    return True

def get_model_config() -> Dict[str, Any]:
    """Get model configuration"""
    return {
        "bge_m3_model": BGE_M3_MODEL,
        "dense_vector_size": DENSE_VECTOR_SIZE,
        "device": EMBEDDING_DEVICE,
        "max_text_length": MAX_TEXT_LENGTH
    }

def get_search_config() -> Dict[str, Any]:
    """Get search configuration"""
    return {
        "collection_name": COLLECTION_NAME,
        "default_top_k": DEFAULT_TOP_K,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "dense_weight": DENSE_WEIGHT,
        "sparse_weight": SPARSE_WEIGHT,
        "rrf_k": RRF_K
    }

def get_sync_config() -> Dict[str, Any]:
    """Get synchronization configuration"""
    return {
        "batch_size": SYNC_BATCH_SIZE,
        "interval_seconds": SYNC_INTERVAL_SECONDS,
        "max_retry_attempts": MAX_RETRY_ATTEMPTS,
        "retry_delay_seconds": RETRY_DELAY_SECONDS
    }
