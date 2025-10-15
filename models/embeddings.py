"""
Embedding models for BGE-M3 dense embeddings and BM25 sparse embeddings with caching support
"""
import numpy as np
import os
import pickle
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
from collections import Counter
import math

from utils.logger import Logger, log_performance
from utils.text_processor import text_processor
from utils.model_cache_manager import cache_manager
from utils.model_downloader import model_downloader
from config import BGE_M3_MODEL, EMBEDDING_DEVICE, MAX_TEXT_LENGTH, OFFLINE_MODE
from .text_preprocessing import preprocess_texts_consistent, preprocess_single_text
from .embeddings_optimized import OptimizedBM25Embedding

logger = Logger.get_logger("hybrid_search.embeddings")


class BGEEmbedding:
    """BGE-M3 dense embedding model with caching support"""
    
    def __init__(self, model_name: str = BGE_M3_MODEL, device: str = EMBEDDING_DEVICE, cache_dir: str = "./model_cache"):
        """Initialize BGE-M3 model with unified caching"""
        self.model_name = model_name
        self.device = device
        self.model = None
        
        # Ensure models are downloaded and cached
        self._ensure_model_cached()
        
        self._load_model()
    
    def _ensure_model_cached(self):
        """Ensure model is downloaded and cached"""
        if not cache_manager.is_dense_model_cached(self.model_name):
            logger.info(f"Dense model {self.model_name} not cached, downloading...")
            model_downloader.download_dense_model(self.model_name)
        else:
            logger.debug(f"Dense model {self.model_name} found in cache")
    
    def _load_model(self):
        """Load the BGE-M3 model from cache"""
        try:
            logger.info(f"Loading BGE-M3 model: {self.model_name} (offline_mode={OFFLINE_MODE})")
            
            # First, try to find the model in local cache
            local_model_path = self._get_local_model_path()
            
            if local_model_path and local_model_path.exists():
                logger.info(f"Using local cached model at: {local_model_path}")
                self.model = SentenceTransformer(str(local_model_path), device=self.device)
            elif OFFLINE_MODE:
                # In offline mode, refuse to download from HuggingFace
                raise RuntimeError(f"Model {self.model_name} not found in local cache and OFFLINE_MODE=true. "
                                 f"Please ensure the model is cached locally or set OFFLINE_MODE=false")
            else:
                # Fallback to HuggingFace with cache environment
                logger.info("Local model not found, using HuggingFace cache")
                os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(cache_manager.dense_cache_dir)
                self.model = SentenceTransformer(self.model_name, device=self.device)
                
            logger.info(f"BGE-M3 model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load BGE-M3 model: {str(e)}")
            raise
    
    def _get_local_model_path(self) -> Optional[Path]:
        """Get the local model path if it exists"""
        try:
            # Check HuggingFace cache format first
            hf_cache_name = f"models--{self.model_name.replace('/', '--')}"
            hf_cache_path = cache_manager.dense_cache_dir / hf_cache_name
            
            if hf_cache_path.exists():
                # Look for the latest snapshot directory with config.json
                snapshots_dir = hf_cache_path / "snapshots"
                if snapshots_dir.exists():
                    for snapshot_dir in snapshots_dir.iterdir():
                        if snapshot_dir.is_dir() and (snapshot_dir / "config.json").exists():
                            logger.debug(f"Found HuggingFace cache at: {snapshot_dir}")
                            return snapshot_dir
            
            # Check old format cache
            old_cache_path = cache_manager.get_dense_model_cache_path(self.model_name)
            if old_cache_path.exists() and (old_cache_path / "config.json").exists():
                logger.debug(f"Found old format cache at: {old_cache_path}")
                return old_cache_path
            
            return None
            
        except Exception as e:
            logger.warning(f"Error checking local model path: {e}")
            return None
    
    @log_performance
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """Encode texts to dense vectors"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Use common preprocessing
        processed_texts = preprocess_texts_consistent(texts, use_multiprocessing=False)
        
        try:
            embeddings = self.model.encode(
                processed_texts,
                batch_size=batch_size,
                show_progress_bar=len(processed_texts) > 10,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            
            logger.debug(f"Encoded {len(processed_texts)} texts to {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to encode texts: {str(e)}")
            raise
    
    def _encode_preprocessed(self, processed_texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode already preprocessed texts to dense vectors"""
        try:
            embeddings = self.model.encode(
                processed_texts,
                batch_size=batch_size,
                show_progress_bar=len(processed_texts) > 10,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            
            logger.debug(f"Encoded {len(processed_texts)} preprocessed texts to {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to encode preprocessed texts: {str(e)}")
            raise
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        if self.model is None:
            return 512  # BGE-M3 default
        return self.model.get_sentence_embedding_dimension()


# Use the optimized BM25 implementation with all its performance benefits
BM25Embedding = OptimizedBM25Embedding


class HybridEmbedding:
    """Hybrid embedding combining BGE-M3 dense and BM25 sparse"""
    
    def __init__(self, dense_weight: float = 0.7, sparse_weight: float = 0.3):
        """Initialize hybrid embedding"""
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
        self.bge_model = BGEEmbedding()
        self.bm25_model = BM25Embedding()
        self.is_fitted = False
        
        logger.info(f"Hybrid embedding initialized: dense_weight={dense_weight}, sparse_weight={sparse_weight}")
    
    def fit(self, texts: List[str]):
        """Fit both models on corpus"""
        logger.info("Fitting hybrid embedding models")
        
        # Fit BM25 on corpus
        self.bm25_model.fit(texts)
        self.is_fitted = True
        
        logger.info("Hybrid embedding models fitted successfully")
    
    @log_performance
    def encode(self, texts: Union[str, List[str]]) -> Dict[str, Any]:
        """Encode texts using both dense and sparse embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Apply consistent text preprocessing for both dense and sparse
        processed_texts = preprocess_texts_consistent(texts, use_multiprocessing=False)
        
        # Get dense embeddings using preprocessed texts
        dense_embeddings = self.bge_model._encode_preprocessed(processed_texts)
        
        # Get sparse embeddings using same preprocessed texts
        sparse_embeddings = self.bm25_model.encode(processed_texts)
        
        return {
            "dense": dense_embeddings,
            "sparse": sparse_embeddings
        }
    
    def encode_query(self, query: str) -> Dict[str, Any]:
        """Encode query for hybrid search"""
        # Apply consistent preprocessing
        processed_query = preprocess_single_text(query)
        
        # Dense embedding
        dense_embedding = self.bge_model._encode_preprocessed([processed_query])[0]
        
        # Sparse embedding
        sparse_embedding = self.bm25_model.encode_query(processed_query)
        
        return {
            "dense": dense_embedding,
            "sparse": sparse_embedding
        }
    
    def get_dense_dim(self) -> int:
        """Get dense embedding dimension"""
        return self.bge_model.get_embedding_dim()
    
    
    def get_sparse_dim(self) -> int:
        """Get sparse embedding dimension (vocabulary size)"""
        return self.bm25_model.get_vocab_size()


# Global instances - lazy initialization
_bge_model = None
_bm25_model = None
_hybrid_model = None


def get_bge_model() -> BGEEmbedding:
    """Get global BGE model instance with lazy initialization"""
    global _bge_model
    if _bge_model is None:
        _bge_model = BGEEmbedding()
    return _bge_model


def get_bm25_model() -> BM25Embedding:
    """Get global BM25 model instance with lazy initialization"""
    global _bm25_model
    if _bm25_model is None:
        _bm25_model = BM25Embedding()
    return _bm25_model


def get_hybrid_model() -> HybridEmbedding:
    """Get global hybrid model instance with lazy initialization"""
    global _hybrid_model
    if _hybrid_model is None:
        _hybrid_model = HybridEmbedding()
    return _hybrid_model
