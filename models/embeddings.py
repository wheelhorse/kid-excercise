"""
Embedding models for BGE-M3 dense embeddings and BM25 sparse embeddings
"""
import numpy as np
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
from collections import Counter
import math

from utils.logger import Logger, log_performance
from utils.text_processor import text_processor
from config import BGE_M3_MODEL, EMBEDDING_DEVICE, MAX_TEXT_LENGTH
from .text_preprocessing import preprocess_texts_consistent, preprocess_single_text

logger = Logger.get_logger("hybrid_search.embeddings")


class BGEEmbedding:
    """BGE-M3 dense embedding model"""
    
    def __init__(self, model_name: str = BGE_M3_MODEL, device: str = EMBEDDING_DEVICE):
        """Initialize BGE-M3 model"""
        self.model_name = model_name
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the BGE-M3 model"""
        try:
            logger.info(f"Loading BGE-M3 model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"BGE-M3 model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load BGE-M3 model: {str(e)}")
            raise
    
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


class BM25Embedding:
    """BM25 sparse embedding using jieba tokenization"""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        """Initialize BM25 parameters"""
        self.k1 = k1
        self.b = b
        self.corpus_tokens = []
        self.doc_freqs = Counter()
        self.idf_values = {}
        self.corpus_size = 0
        self.avg_doc_length = 0.0
        self.vocabulary = {}
        self.vocab_size = 0
        
        logger.info(f"BM25 initialized with k1={k1}, b={b}")
    
    def fit(self, texts: List[str]):
        """Fit BM25 on corpus"""
        logger.info(f"Fitting BM25 on {len(texts)} documents")
        
        self.corpus_tokens = []
        self.doc_freqs = Counter()
        
        # Tokenize all documents
        total_length = 0
        for text in texts:
            tokens = text_processor.create_bm25_tokens(text)
            self.corpus_tokens.append(tokens)
            total_length += len(tokens)
            
            # Count document frequencies
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1
        
        self.corpus_size = len(texts)
        self.avg_doc_length = total_length / self.corpus_size if self.corpus_size > 0 else 0
        
        # Create vocabulary mapping
        self.vocabulary = {token: idx for idx, token in enumerate(self.doc_freqs.keys())}
        self.vocab_size = len(self.vocabulary)
        
        # Calculate IDF values
        self._calculate_idf()
        
        logger.info(f"BM25 fitted: vocab_size={self.vocab_size}, avg_doc_length={self.avg_doc_length:.2f}")
    
    def _calculate_idf(self):
        """Calculate IDF values for all terms"""
        self.idf_values = {}
        for term, doc_freq in self.doc_freqs.items():
            idf = math.log((self.corpus_size - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
            self.idf_values[term] = max(idf, 0.01)  # Minimum IDF to avoid zero
    
    @log_performance
    def encode(self, texts: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """Encode texts to sparse vectors"""
        if isinstance(texts, str):
            texts = [texts]
        
        sparse_vectors = []
        for text in texts:
            sparse_vector = self._encode_single(text)
            sparse_vectors.append(sparse_vector)
        
        logger.debug(f"Encoded {len(texts)} texts to sparse vectors")
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
                
                # BM25 score calculation
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


# Global instances
bge_model = BGEEmbedding()
bm25_model = BM25Embedding()
hybrid_model = HybridEmbedding()
