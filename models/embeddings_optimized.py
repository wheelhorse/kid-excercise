"""
CPU-Optimized Embedding models for BGE-M3 dense embeddings and BM25 sparse embeddings
"""
import numpy as np
import os
import torch
import multiprocessing as mp
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
from collections import Counter
import math
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial

from utils.logger import Logger, log_performance
from utils.text_processor import text_processor
from utils.model_cache_manager import cache_manager
from utils.model_downloader import model_downloader
from config import BGE_M3_MODEL, EMBEDDING_DEVICE, MAX_TEXT_LENGTH
from .text_preprocessing import preprocess_texts_consistent, preprocess_single_text

logger = Logger.get_logger("hybrid_search.embeddings_optimized")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class OptimizedBGEEmbedding:
    """CPU-Optimized BGE-M3 dense embedding model"""
    
    def __init__(self, model_name: str = BGE_M3_MODEL, device: str = EMBEDDING_DEVICE, cache_dir: str = "./model_cache"):
        """Initialize optimized BGE-M3 model with unified caching"""
        self.model_name = model_name
        self.device = device
        self.model = None
        
        # CPU optimization settings
        self.cpu_count = os.cpu_count() or 1
        self.optimal_threads = min(self.cpu_count, 128)  # Cap at 128 for memory efficiency
        self.optimal_batch_size = self._calculate_optimal_batch_size()
        
        # Set PyTorch threading for optimal CPU usage (only if not already set)
        self._set_thread_settings()
        
        # Ensure models are downloaded and cached
        self._ensure_model_cached()
        
        self._load_model()
        
        logger.info(f"Optimized BGE model initialized: threads={self.optimal_threads}, "
                   f"batch_size={self.optimal_batch_size}, cpu_count={self.cpu_count}")
    
    def _ensure_model_cached(self):
        """Ensure model is downloaded and cached"""
        if not cache_manager.is_dense_model_cached(self.model_name):
            logger.info(f"Dense model {self.model_name} not cached, downloading...")
            model_downloader.download_dense_model(self.model_name)
        else:
            logger.debug(f"Dense model {self.model_name} found in cache")
    
    def _set_thread_settings(self):
        """Set thread settings safely"""
        try:
            # Only set if not already configured
            if torch.get_num_threads() == 1:  # Default PyTorch value
                torch.set_num_threads(self.optimal_threads)
            if torch.get_num_interop_threads() == 1:  # Default PyTorch value
                torch.set_num_interop_threads(self.optimal_threads)
        except RuntimeError as e:
            logger.warning(f"Could not set PyTorch thread settings: {e}")
        
        # Set environment variables (these are safe to set)
        os.environ['OMP_NUM_THREADS'] = str(self.optimal_threads)
        os.environ['MKL_NUM_THREADS'] = str(self.optimal_threads)
        os.environ['NUMBA_NUM_THREADS'] = str(self.optimal_threads)
    
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on CPU cores"""
        # Rule of thumb: batch_size should be multiple of CPU cores for better parallelization
        base_batch_size = max(32, self.cpu_count // 4)  # At least 32, scale with cores
        return min(base_batch_size, 256)  # Cap at 256 to avoid memory issues
    
    def _load_model(self):
        """Load the BGE-M3 model with CPU optimizations from unified cache"""
        try:
            logger.info(f"Loading optimized BGE-M3 model: {self.model_name}")
            
            # Configure cache environment for SentenceTransformers
            os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(cache_manager.dense_cache_dir)
            
            # Load model with CPU-specific optimizations
            self.model = SentenceTransformer(
                self.model_name, 
                device=self.device,
                # Use unified cache
                cache_folder=str(cache_manager.dense_cache_dir)
            )
            
            # Apply CPU-specific optimizations to the model
            if self.device == 'cpu':
                # Enable CPU optimizations for transformers
                self._optimize_for_cpu()
            
            logger.info(f"Optimized BGE-M3 model loaded successfully from cache on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load optimized BGE-M3 model: {str(e)}")
            raise
    
    def _optimize_for_cpu(self):
        """Apply CPU-specific optimizations"""
        try:
            # Set model to evaluation mode and optimize for inference
            self.model.eval()
            
            # Enable CPU-specific optimizations if available
            for module in self.model.modules():
                if hasattr(module, 'cpu'):
                    module.cpu()
            
            logger.debug("Applied CPU-specific optimizations")
            
        except Exception as e:
            logger.warning(f"Could not apply all CPU optimizations: {str(e)}")
    
    @log_performance
    def encode(self, texts: Union[str, List[str]], 
               batch_size: Optional[int] = None,
               num_workers: Optional[int] = None,
               use_multiprocessing: bool = False) -> np.ndarray:
        """
        Encode texts to dense vectors with CPU optimizations
        
        Args:
            texts: Text or list of texts to encode
            batch_size: Batch size for encoding (auto-calculated if None)
            num_workers: Number of worker threads/processes (auto-calculated if None)
            use_multiprocessing: Use multiprocessing instead of threading for large batches
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Use optimal parameters if not specified
        if batch_size is None:
            batch_size = self.optimal_batch_size
        if num_workers is None:
            num_workers = min(self.optimal_threads, len(texts) // 10 + 1)
        
        # Preprocess texts using common function
        processed_texts = preprocess_texts_consistent(texts, use_multiprocessing=use_multiprocessing)
        
        try:
            # For large batches, consider parallel processing
            if len(processed_texts) > 1000 and use_multiprocessing:
                embeddings = self._encode_with_multiprocessing(
                    processed_texts, batch_size, num_workers
                )
            else:
                # Standard optimized encoding
                embeddings = self._encode_standard(processed_texts, batch_size)
            
            logger.debug(f"Encoded {len(processed_texts)} texts to {embeddings.shape} "
                        f"using batch_size={batch_size}, workers={num_workers}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to encode texts: {str(e)}")
            raise
    
    
    def _encode_standard(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Standard optimized encoding"""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 50,
            convert_to_numpy=True,
            normalize_embeddings=True,
            # CPU-specific optimizations
            device=self.device
        )
    
    def _encode_with_multiprocessing(self, texts: List[str], 
                                   batch_size: int, num_workers: int) -> np.ndarray:
        """Encode using multiprocessing for very large batches"""
        # Split texts into chunks for parallel processing
        chunk_size = max(len(texts) // num_workers, batch_size)
        text_chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
        
        # Create partial function for worker processes
        encode_chunk_func = partial(self._encode_chunk, batch_size=batch_size)
        
        # Use process pool for CPU-intensive work
        with ProcessPoolExecutor(max_workers=min(num_workers, len(text_chunks))) as executor:
            chunk_embeddings = list(executor.map(encode_chunk_func, text_chunks))
        
        # Concatenate results
        return np.vstack(chunk_embeddings)
    
    def _encode_chunk(self, text_chunk: List[str], batch_size: int) -> np.ndarray:
        """Encode a chunk of texts (for multiprocessing)"""
        # Create a temporary model instance for this process
        temp_model = SentenceTransformer(self.model_name, device='cpu')
        temp_model.eval()
        
        return temp_model.encode(
            text_chunk,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        if self.model is None:
            return 1024  # BGE-M3 default
        return self.model.get_sentence_embedding_dimension()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance configuration stats"""
        return {
            "cpu_count": self.cpu_count,
            "optimal_threads": self.optimal_threads,
            "optimal_batch_size": self.optimal_batch_size,
            "pytorch_threads": torch.get_num_threads(),
            "device": self.device
        }


class OptimizedBM25Embedding:
    """CPU-Optimized BM25 sparse embedding with parallel processing"""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75, cache_dir: str = "./model_cache"):
        """Initialize optimized BM25 parameters"""
        self.k1 = k1
        self.b = b
        self.corpus_tokens = []
        self.doc_freqs = Counter()
        self.idf_values = {}
        self.corpus_size = 0
        self.avg_doc_length = 0.0
        self.vocabulary = {}
        self.vocab_size = 0
        
        # Use unified cache manager for BM25 cache path
        self.bm25_cache_path = cache_manager.get_bm25_cache_path()
        
        # CPU optimization settings
        self.cpu_count = os.cpu_count() or 1
        self.optimal_workers = min(self.cpu_count, 64)  # Cap workers for memory efficiency
        
        # Try to load existing model first
        self._try_load_cached_model()
        
        logger.info(f"Optimized BM25 initialized with k1={k1}, b={b}, workers={self.optimal_workers}")
    
    def fit(self, texts: List[str], use_parallel: bool = True):
        """Fit BM25 on corpus with parallel processing"""
        logger.info(f"Fitting optimized BM25 on {len(texts)} documents")
        
        if use_parallel and len(texts) > 100:
            self._fit_parallel(texts)
        else:
            self._fit_sequential(texts)
        
        # Create vocabulary mapping
        self.vocabulary = {token: idx for idx, token in enumerate(self.doc_freqs.keys())}
        self.vocab_size = len(self.vocabulary)
        
        # Calculate IDF values
        self._calculate_idf()
        
        # Save fitted model to cache
        self._save_model()
        
        logger.info(f"Optimized BM25 fitted: vocab_size={self.vocab_size}, "
                   f"avg_doc_length={self.avg_doc_length:.2f}")
    
    def _fit_parallel(self, texts: List[str]):
        """Parallel fitting using multiprocessing"""
        # Split texts into chunks for parallel processing
        chunk_size = max(len(texts) // self.optimal_workers, 100)
        text_chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=len(text_chunks)) as executor:
            chunk_results = list(executor.map(self._process_text_chunk, text_chunks))
        
        # Combine results
        self.corpus_tokens = []
        total_length = 0
        self.doc_freqs = Counter()
        
        for chunk_tokens, chunk_length, chunk_freqs in chunk_results:
            self.corpus_tokens.extend(chunk_tokens)
            total_length += chunk_length
            self.doc_freqs.update(chunk_freqs)
        
        self.corpus_size = len(texts)
        self.avg_doc_length = total_length / self.corpus_size if self.corpus_size > 0 else 0
    
    def _fit_sequential(self, texts: List[str]):
        """Sequential fitting (fallback)"""
        self.corpus_tokens = []
        self.doc_freqs = Counter()
        
        total_length = 0
        for text in texts:
            tokens = text_processor.create_bm25_tokens(text)
            self.corpus_tokens.append(tokens)
            total_length += len(tokens)
            
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1
        
        self.corpus_size = len(texts)
        self.avg_doc_length = total_length / self.corpus_size if self.corpus_size > 0 else 0
    
    def _process_text_chunk(self, text_chunk: List[str]) -> tuple:
        """Process a chunk of texts for parallel fitting"""
        chunk_tokens = []
        chunk_freqs = Counter()
        chunk_length = 0
        
        for text in text_chunk:
            tokens = text_processor.create_bm25_tokens(text)
            chunk_tokens.append(tokens)
            chunk_length += len(tokens)
            
            unique_tokens = set(tokens)
            for token in unique_tokens:
                chunk_freqs[token] += 1
        
        return chunk_tokens, chunk_length, chunk_freqs
    
    def _calculate_idf(self):
        """Calculate IDF values for all terms"""
        self.idf_values = {}
        for term, doc_freq in self.doc_freqs.items():
            idf = math.log((self.corpus_size - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
            self.idf_values[term] = max(idf, 0.01)
    
    @log_performance
    def encode(self, texts: Union[str, List[str]], use_parallel: bool = True) -> List[Dict[str, Any]]:
        """Encode texts to sparse vectors with parallel processing"""
        if isinstance(texts, str):
            texts = [texts]
        
        if use_parallel and len(texts) > 50:
            return self._encode_parallel(texts)
        else:
            return self._encode_sequential(texts)
    
    def _encode_parallel(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Parallel encoding using threading"""
        with ThreadPoolExecutor(max_workers=self.optimal_workers) as executor:
            sparse_vectors = list(executor.map(self._encode_single, texts))
        
        logger.debug(f"Parallel encoded {len(texts)} texts to sparse vectors")
        return sparse_vectors
    
    def _encode_sequential(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Sequential encoding (fallback)"""
        sparse_vectors = [self._encode_single(text) for text in texts]
        logger.debug(f"Sequential encoded {len(texts)} texts to sparse vectors")
        return sparse_vectors
    
    def _encode_single(self, text: str) -> Dict[str, Any]:
        """Encode single text to sparse vector"""
        tokens = text_processor.create_bm25_tokens(text)
        token_counts = Counter(tokens)
        doc_length = len(tokens)
        
        indices = []
        values = []
        
        for token, count in token_counts.items():
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                idf = self.idf_values[token]
                
                tf = count
                norm_factor = self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                score = idf * (tf * (self.k1 + 1)) / (tf + norm_factor)
                
                if score > 0:
                    indices.append(idx)
                    values.append(score)
        
        return {
            "indices": indices,
            "values": values
        }
    
    def encode_query(self, query: str) -> Dict[str, Any]:
        """Encode query for search"""
        return self._encode_single(query)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return self.vocab_size
    
    def _try_load_cached_model(self):
        """Try to load cached BM25 model using unified cache system"""
        if self.bm25_cache_path.exists():
            try:
                logger.info(f"Loading cached BM25 model from {self.bm25_cache_path}")
                with open(self.bm25_cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Restore model state
                self.doc_freqs = cached_data['doc_freqs']
                self.idf_values = cached_data['idf_values']
                self.corpus_size = cached_data['corpus_size']
                self.avg_doc_length = cached_data['avg_doc_length']
                self.vocabulary = cached_data['vocabulary']
                self.vocab_size = cached_data['vocab_size']
                self.k1 = cached_data.get('k1', self.k1)
                self.b = cached_data.get('b', self.b)
                
                # Register with unified cache manager
                cache_manager.register_model(
                    model_type="sparse",
                    model_name="bm25",
                    cache_path=self.bm25_cache_path,
                    metadata={
                        "vocab_size": self.vocab_size,
                        "corpus_size": self.corpus_size,
                        "k1": self.k1,
                        "b": self.b,
                        "avg_doc_length": self.avg_doc_length
                    }
                )
                
                logger.info(f"Cached BM25 model loaded: vocab_size={self.vocab_size}, "
                           f"corpus_size={self.corpus_size}")
                return True
                
            except Exception as e:
                logger.warning(f"Failed to load cached BM25 model: {e}")
                # Clear corrupted cache
                try:
                    self.bm25_cache_path.unlink()
                except:
                    pass
        
        return False
    
    def _save_model(self):
        """Save current BM25 model state to cache with unified cache registration"""
        try:
            cache_data = {
                'doc_freqs': self.doc_freqs,
                'idf_values': self.idf_values,
                'corpus_size': self.corpus_size,
                'avg_doc_length': self.avg_doc_length,
                'vocabulary': self.vocabulary,
                'vocab_size': self.vocab_size,
                'k1': self.k1,
                'b': self.b,
                'saved_at': str(Path.cwd())  # For debugging
            }
            
            # Atomic write using temporary file
            temp_path = self.bm25_cache_path.with_suffix('.tmp')
            with open(temp_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Atomic move
            temp_path.replace(self.bm25_cache_path)
            
            # Register with unified cache manager
            cache_manager.register_model(
                model_type="sparse",
                model_name="bm25",
                cache_path=self.bm25_cache_path,
                metadata={
                    "vocab_size": self.vocab_size,
                    "corpus_size": self.corpus_size,
                    "k1": self.k1,
                    "b": self.b,
                    "avg_doc_length": self.avg_doc_length
                }
            )
            
            logger.info(f"BM25 model saved to cache: {self.bm25_cache_path}")
            
        except Exception as e:
            logger.error(f"Failed to save BM25 model to cache: {e}")


class OptimizedHybridEmbedding:
    """CPU-Optimized hybrid embedding combining BGE-M3 dense and BM25 sparse"""
    
    def __init__(self, dense_weight: float = 0.7, sparse_weight: float = 0.3, cache_dir: str = "./model_cache"):
        """Initialize optimized hybrid embedding"""
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.cache_dir = cache_dir
        
        # Initialize models with shared cache directory
        self.bge_model = OptimizedBGEEmbedding(cache_dir=cache_dir)
        self.bm25_model = OptimizedBM25Embedding(cache_dir=cache_dir)
        self.is_fitted = False
        
        logger.info(f"Optimized hybrid embedding initialized: "
                   f"dense_weight={dense_weight}, sparse_weight={sparse_weight}, cache_dir={cache_dir}")
    
    def fit(self, texts: List[str], use_parallel: bool = True):
        """Fit both models on corpus with parallel processing"""
        logger.info("Fitting optimized hybrid embedding models")
        
        # Fit BM25 on corpus with parallel processing
        self.bm25_model.fit(texts, use_parallel=use_parallel)
        self.is_fitted = True
        
        logger.info("Optimized hybrid embedding models fitted successfully")
    
    @log_performance
    def encode(self, texts: Union[str, List[str]], 
               batch_size: Optional[int] = None,
               use_parallel: bool = True) -> Dict[str, Any]:
        """Encode texts using both dense and sparse embeddings with optimizations"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Apply consistent text preprocessing for both dense and sparse
        processed_texts = preprocess_texts_consistent(texts, use_multiprocessing=True)
        
        # Get dense embeddings with optimizations using preprocessed texts
        dense_embeddings = self.bge_model._encode_standard(processed_texts, 
                                                          batch_size or self.bge_model.optimal_batch_size)
        
        # Get sparse embeddings with parallel processing using same preprocessed texts
        sparse_embeddings = self.bm25_model.encode(processed_texts, use_parallel=use_parallel)
        
        return {
            "dense": dense_embeddings,
            "sparse": sparse_embeddings
        }
    
    def encode_query(self, query: str) -> Dict[str, Any]:
        """Encode query for hybrid search"""
        # Apply consistent preprocessing
        processed_query = preprocess_single_text(query)
        
        # Dense embedding
        dense_embedding = self.bge_model._encode_standard([processed_query], 
                                                         self.bge_model.optimal_batch_size)[0]
        
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
    
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance stats"""
        return {
            "bge_stats": self.bge_model.get_performance_stats(),
            "bm25_workers": self.bm25_model.optimal_workers,
            "preprocessing_consistency": "Applied unified preprocessing for both dense and sparse",
            "cpu_count": os.cpu_count()
        }


# Global optimized instances
optimized_bge_model = OptimizedBGEEmbedding()
optimized_bm25_model = OptimizedBM25Embedding()
optimized_hybrid_model = OptimizedHybridEmbedding()
