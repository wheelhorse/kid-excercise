"""
Intel-optimized Embedding models using Intel PyTorch/IPEX acceleration with caching support
Focus on Intel Extension for PyTorch (IPEX) and Intel MKL optimizations
"""
import numpy as np
import os
import pickle
import tempfile
import torch
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
from collections import Counter
import math
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Intel-specific high-performance libraries
try:
    import openvino as ov
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False

from utils.logger import Logger, log_performance
from utils.text_processor import text_processor
from config import BGE_M3_MODEL, EMBEDDING_DEVICE, MAX_TEXT_LENGTH
from .text_preprocessing import preprocess_texts_consistent, preprocess_single_text

logger = Logger.get_logger("hybrid_search.embeddings_intel_optimized")


class IntelOptimizedBGEEmbedding:
    """Intel-optimized BGE-M3 dense embedding model with Intel hardware acceleration and caching"""
    
    def __init__(self, model_name: str = BGE_M3_MODEL, device: str = EMBEDDING_DEVICE, cache_dir: str = "./model_cache"):
        """Initialize Intel-optimized BGE-M3 model with caching"""
        self.model_name = model_name
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.model = None
        self.openvino_model = None
        self.optimization_type = "standard"
        self.is_intel_cpu = self._detect_intel_cpu()
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # CPU optimization settings
        self.cpu_count = os.cpu_count() or 1
        self.optimal_threads = min(self.cpu_count, 128)
        self.optimal_batch_size = self._calculate_optimal_batch_size()
        
        # Set Intel-specific CPU optimizations
        self._set_intel_cpu_optimizations()
        
        # Load Intel-optimized PyTorch/IPEX model
        self._load_intel_optimized_model()
        
        logger.info(f"Intel-optimized BGE model initialized: {self.optimization_type}, "
                   f"Intel CPU: {self.is_intel_cpu}, threads={self.optimal_threads}, "
                   f"batch_size={self.optimal_batch_size}, cache_dir={cache_dir}")
    
    def _detect_intel_cpu(self) -> bool:
        """Detect if running on Intel CPU"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpu_info = f.read().lower()
            return 'intel' in cpu_info or 'genuine intel' in cpu_info
        except:
            # Fallback detection method
            try:
                import platform
                return 'intel' in platform.processor().lower()
            except:
                return False
    
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size for Intel CPUs"""
        if self.is_intel_cpu:
            # Intel CPUs benefit from larger batch sizes with MKL
            base_batch_size = max(32, self.cpu_count)
            return min(base_batch_size, 256)
        else:
            # Conservative for non-Intel CPUs
            base_batch_size = max(16, self.cpu_count // 2)
            return min(base_batch_size, 128)
    
    def _set_intel_cpu_optimizations(self):
        """Set comprehensive Intel CPU optimizations"""
        # PyTorch optimizations (only if not already set)
        try:
            if torch.get_num_threads() == 1:  # Default PyTorch value
                torch.set_num_threads(self.optimal_threads)
            if torch.get_num_interop_threads() == 1:  # Default PyTorch value
                torch.set_num_interop_threads(self.optimal_threads)
        except RuntimeError as e:
            logger.warning(f"Could not set PyTorch thread settings: {e}")
        
        # Intel MKL optimizations (most important for Intel CPUs)
        os.environ['OMP_NUM_THREADS'] = str(self.optimal_threads)
        os.environ['MKL_NUM_THREADS'] = str(self.optimal_threads)
        os.environ['NUMBA_NUM_THREADS'] = str(self.optimal_threads)
        
        if self.is_intel_cpu:
            # Intel-specific optimizations
            os.environ['KMP_BLOCKTIME'] = '0'  # Reduce thread spin-wait time
            os.environ['KMP_AFFINITY'] = 'granularity=fine,verbose,compact,1,0'
            os.environ['KMP_SETTINGS'] = '1'  # Display KMP settings
            
            # Intel MKL specific settings
            os.environ['MKL_DYNAMIC'] = 'FALSE'  # Disable dynamic adjustment
            os.environ['MKL_ENABLE_INSTRUCTIONS'] = 'AVX512'  # Use Intel AVX-512 if available
            
            logger.info("Applied Intel CPU-specific optimizations")
        else:
            logger.warning("Non-Intel CPU detected - some optimizations may not be effective")
    
    def _load_intel_optimized_model(self):
        """Load Intel-optimized PyTorch/IPEX model"""
        if self.is_intel_cpu:
            # Try Intel Extension for PyTorch first (best for Intel CPUs)
            if self._try_load_ipex_model():
                return
        
        # Use standard optimized PyTorch as fallback
        self._load_standard_optimized_model()
        
        # Mark OpenVINO as available for future use (if needed)
        self.openvino_available = OPENVINO_AVAILABLE and self.is_intel_cpu
    
    def _try_load_openvino_model(self) -> bool:
        """Try to load OpenVINO optimized model (Intel CPUs only)"""
        try:
            if not OPENVINO_AVAILABLE or not self.is_intel_cpu:
                return False
            
            logger.info("Attempting to load OpenVINO optimized model for Intel CPU...")
            
            # OpenVINO optimization would require model conversion
            # This is a simplified implementation - full OpenVINO integration would be more complex
            logger.info("OpenVINO optimization placeholder - full implementation pending")
            
            # For now, return False to fall back to other methods
            return False
            
        except Exception as e:
            logger.warning(f"OpenVINO optimization failed: {e}")
            return False
    
    def _try_load_ipex_model(self) -> bool:
        """Try to load Intel Extension for PyTorch optimized model"""
        try:
            if not IPEX_AVAILABLE:
                return False
            
            logger.info("Attempting to load Intel Extension for PyTorch...")
            
            # Load standard model first
            self.model = SentenceTransformer(self.model_name, device='cpu')
            self.model.eval()
            
            # Apply Intel optimizations
            self.model = ipex.optimize(self.model)
            self.optimization_type = "Intel PyTorch Extension"
            
            logger.info("Successfully loaded Intel PyTorch Extension optimized model")
            return True
            
        except Exception as e:
            logger.warning(f"Intel PyTorch Extension optimization failed: {e}")
            return False
    
    
    def _load_standard_optimized_model(self):
        """Load standard optimized PyTorch model with Intel optimizations"""
        try:
            logger.info("Loading Intel-optimized PyTorch model...")
            
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.model.eval()
            
            # Apply Intel CPU optimizations
            if self.device == 'cpu':
                # Enable Intel MKL-DNN optimizations
                torch.backends.mkldnn.enabled = True
                torch.backends.mkl.enabled = True
                
                if self.is_intel_cpu:
                    # Fast warmup for Intel CPUs (don't do heavy JIT compilation)
                    torch.backends.mkldnn.verbose = 0  # Disable verbose output
                    
                    with torch.no_grad():
                        dummy_input = ["Intel optimization warmup"]
                        try:
                            # Quick warmup without heavy JIT
                            self.model.encode(dummy_input, convert_to_numpy=True, normalize_embeddings=True)
                            self.optimization_type = "PyTorch (Intel-optimized)"
                            logger.info("Intel-optimized PyTorch model ready")
                        except Exception as e:
                            logger.warning(f"Intel warmup failed: {e}")
                            self.optimization_type = "PyTorch (Standard)"
                else:
                    self.optimization_type = "PyTorch (Non-Intel CPU)"
            
            logger.info("Intel-optimized model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load standard model: {e}")
            raise
    
    @log_performance
    def encode(self, texts: Union[str, List[str]], 
               batch_size: Optional[int] = None,
               num_workers: Optional[int] = None) -> np.ndarray:
        """
        Intel-optimized encoding with PyTorch/IPEX acceleration
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Use Intel-optimized parameters if not specified
        if batch_size is None:
            batch_size = self.optimal_batch_size
        if num_workers is None:
            # Intel CPUs benefit from more workers due to hyperthreading
            num_workers = min(self.optimal_threads, len(texts) // 3 + 1) if self.is_intel_cpu else min(self.optimal_threads, len(texts) // 5 + 1)
        
        # Preprocess texts using common function
        processed_texts = preprocess_texts_consistent(texts, use_multiprocessing=True)
        
        try:
            # Use Intel-optimized PyTorch/IPEX for all dataset sizes
            embeddings = self._encode_best_available(processed_texts, batch_size)
            
            logger.debug(f"Intel-optimized encoded {len(processed_texts)} texts to {embeddings.shape} "
                        f"using {self.optimization_type}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to encode texts: {str(e)}")
            raise
    
    
    
    def _encode_ipex(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Intel Extension for PyTorch optimized encoding"""
        with torch.no_grad(), torch.cpu.amp.autocast():
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 50,
                convert_to_numpy=True,
                normalize_embeddings=True,
                device=self.device
            )
        return embeddings
    
    def _encode_best_available(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Use the best available encoding method for Intel CPUs"""
        if self.optimization_type == "Intel PyTorch Extension":
            return self._encode_ipex(texts, batch_size)
        else:
            return self._encode_intel_optimized_standard(texts, batch_size)
    
    def _encode_intel_optimized_standard(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Standard optimized encoding with Intel CPU-specific optimizations"""
        # Enable Intel CPU optimizations
        with torch.no_grad():
            # Use mixed precision for faster computation on Intel CPUs
            if hasattr(torch, 'cpu') and hasattr(torch.cpu, 'amp'):
                with torch.cpu.amp.autocast():
                    embeddings = self.model.encode(
                        texts,
                        batch_size=batch_size,
                        show_progress_bar=len(texts) > 50,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        device=self.device
                    )
            else:
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=len(texts) > 50,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    device=self.device
                )
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        if self.model is not None:
            return self.model.get_sentence_embedding_dimension()
        else:
            return 1024  # BGE-M3 default
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive Intel CPU performance stats"""
        return {
            "optimization_type": self.optimization_type,
            "is_intel_cpu": self.is_intel_cpu,
            "cpu_count": self.cpu_count,
            "optimal_threads": self.optimal_threads,
            "optimal_batch_size": self.optimal_batch_size,
            "pytorch_threads": torch.get_num_threads(),
            "device": self.device,
            "intel_libraries": {
                "openvino": OPENVINO_AVAILABLE,
                "intel_extension_pytorch": IPEX_AVAILABLE
            },
            "mkl_settings": {
                "mkldnn_enabled": torch.backends.mkldnn.enabled,
                "mkl_enabled": torch.backends.mkl.enabled
            }
        }


class IntelOptimizedHybridEmbedding:
    """Intel-optimized hybrid embedding with Intel-specific acceleration"""
    
    def __init__(self, dense_weight: float = 0.7, sparse_weight: float = 0.3):
        """Initialize Intel-optimized hybrid embedding"""
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
        self.bge_model = IntelOptimizedBGEEmbedding()
        
        # Use optimized BM25 from previous implementation
        from .embeddings_optimized import OptimizedBM25Embedding
        self.bm25_model = OptimizedBM25Embedding()
        
        self.is_fitted = False
        
        logger.info(f"Intel-optimized hybrid embedding initialized with {self.bge_model.optimization_type}")
    
    def fit(self, texts: List[str], use_parallel: bool = True):
        """Fit models with Intel optimizations"""
        logger.info("Fitting Intel-optimized hybrid embedding models")
        
        # Fit BM25 with Intel-optimized parallel processing
        self.bm25_model.fit(texts, use_parallel=use_parallel)
        self.is_fitted = True
        
        logger.info("Intel-optimized hybrid embedding models fitted successfully")
    
    @log_performance
    def encode(self, texts: Union[str, List[str]], 
               batch_size: Optional[int] = None,
               use_parallel: bool = True) -> Dict[str, Any]:
        """Intel-optimized encoding"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Apply consistent text preprocessing for both dense and sparse
        processed_texts = preprocess_texts_consistent(texts, use_multiprocessing=True)
        
        # Get Intel-optimized dense embeddings
        dense_embeddings = self.bge_model._encode_best_available(processed_texts, 
                                                                batch_size or self.bge_model.optimal_batch_size)
        
        # Get optimized sparse embeddings with same preprocessing
        sparse_embeddings = self.bm25_model.encode(processed_texts, use_parallel=use_parallel)
        
        return {
            "dense": dense_embeddings,
            "sparse": sparse_embeddings
        }
    
    def encode_query(self, query: str) -> Dict[str, Any]:
        """Intel-optimized query encoding"""
        # Apply consistent preprocessing
        processed_query = preprocess_single_text(query)
        
        # Dense embedding
        dense_embedding = self.bge_model._encode_best_available([processed_query], 
                                                               self.bge_model.optimal_batch_size)[0]
        
        # Sparse embedding
        sparse_embedding = self.bm25_model.encode_query(processed_query)
        
        return {
            "dense": dense_embedding,
            "sparse": sparse_embedding
        }
    
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive Intel performance statistics"""
        return {
            "bge_stats": self.bge_model.get_performance_stats(),
            "bm25_workers": self.bm25_model.optimal_workers,
            "preprocessing_consistency": "Applied unified preprocessing for both dense and sparse",
            "intel_optimization_active": self.bge_model.is_intel_cpu
        }


# Global Intel-optimized instances
intel_optimized_bge_model = IntelOptimizedBGEEmbedding()
intel_optimized_hybrid_model = IntelOptimizedHybridEmbedding()


def install_intel_optimization_libraries():
    """Instructions and script to install Intel-specific optimization libraries"""
    install_commands = [
        "# Install Intel-optimized libraries for best performance on Intel CPUs",
        "",
        "# Install Intel Extension for PyTorch (Intel CPUs - PRIMARY RECOMMENDATION)",
        "pip install intel_extension_for_pytorch",
        "",
        "# Install OpenVINO (Intel CPUs - advanced optimization)",
        "pip install openvino",
        "",
        "# Install Intel MKL and OpenMP for maximum performance",
        "pip install mkl",
        "pip install intel-openmp",
        "",
        "# Additional Intel optimization tools (optional)",
        "pip install intel-tensorflow  # If using TensorFlow components",
        "pip install scikit-learn-intelex  # Intel-optimized scikit-learn",
        "",
        "# Note: ONNX Runtime has been removed from Intel optimization",
        "# as PyTorch/IPEX provides better performance for all dataset sizes",
    ]
    
    return "\n".join(install_commands)


def check_intel_cpu_compatibility():
    """Check if system is compatible with Intel optimizations"""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpu_info = f.read().lower()
        
        is_intel = 'intel' in cpu_info or 'genuine intel' in cpu_info
        
        print("=== Intel CPU Compatibility Check ===")
        print(f"Intel CPU detected: {is_intel}")
        
        if is_intel:
            print("✓ System is compatible with Intel-specific optimizations")
            print("✓ Recommended to install Intel Extension for PyTorch and OpenVINO")
        else:
            print("⚠ Non-Intel CPU detected")
            print("⚠ Intel-specific optimizations may not provide benefits")
            print("⚠ Consider using generic optimizations instead")
        
        return is_intel
        
    except Exception as e:
        print(f"Could not detect CPU type: {e}")
        return False


if __name__ == "__main__":
    print("=== Intel-Optimized Embedding Libraries (PyTorch/IPEX Focus) ===")
    is_intel = check_intel_cpu_compatibility()
    print(f"\nLibrary availability:")
    print(f"OpenVINO Available: {OPENVINO_AVAILABLE}")
    print(f"Intel Extension for PyTorch Available: {IPEX_AVAILABLE}")
    
    if is_intel:
        print("\n=== Intel CPU Detected - Recommended Installation ===")
        print(install_intel_optimization_libraries())
    else:
        print("\n=== Non-Intel CPU Detected ===")
        print("Consider using embeddings_optimized.py instead")
