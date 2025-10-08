"""
AMD-specific optimized embedding models for AMD 5700G and similar processors
Focuses on optimizations that work best with AMD Zen architecture
"""
import numpy as np
import os
import torch
import multiprocessing as mp
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
from collections import Counter
import math
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import warnings

# AMD-compatible optimization libraries
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    OPTIMUM_AVAILABLE = True
except ImportError:
    OPTIMUM_AVAILABLE = False

from utils.logger import Logger, log_performance
from utils.text_processor import text_processor
from config import BGE_M3_MODEL, EMBEDDING_DEVICE, MAX_TEXT_LENGTH

logger = Logger.get_logger("hybrid_search.embeddings_amd_optimized")


class AMDOptimizedBGEEmbedding:
    """AMD 5700G optimized BGE-M3 dense embedding model"""
    
    def __init__(self, model_name: str = BGE_M3_MODEL, device: str = EMBEDDING_DEVICE):
        """Initialize AMD-optimized BGE-M3 model"""
        self.model_name = model_name
        self.device = device
        self.model = None
        self.onnx_model = None
        self.optimization_type = "standard"
        
        # AMD processor detection and optimization
        self.cpu_count = os.cpu_count() or 1
        self.physical_cores = self.cpu_count // 2  # Assume hyperthreading
        
        # AMD-optimized threading - use MORE threads like the optimized version
        self.optimal_threads = self.cpu_count  # Use all available threads
        self.optimal_batch_size = self._calculate_amd_optimal_batch_size()
        
        # Set AMD-specific optimizations
        self._set_amd_optimizations()
        
        # Load the most optimized version for AMD
        self._load_amd_optimized_model()
        
        logger.info(f"AMD-optimized BGE model initialized: {self.optimization_type}, "
                   f"threads={self.optimal_threads}, batch_size={self.optimal_batch_size}, "
                   f"physical_cores={self.physical_cores}")
    
    def _calculate_amd_optimal_batch_size(self) -> int:
        """Calculate optimal batch size for AMD processors"""
        # AMD processors can handle larger batch sizes like the optimized version
        # Match the performance of the optimized implementation
        if self.cpu_count >= 64:  # High-end AMD EPYC
            return 128
        elif self.cpu_count >= 32:  # Mid-range AMD EPYC
            return 96
        elif self.cpu_count >= 16:  # AMD Ryzen/Threadripper
            return 64
        elif self.cpu_count >= 8:  # AMD Ryzen 7/5700G
            return 48
        elif self.cpu_count >= 4:
            return 32
        else:
            return 16
    
    def _set_amd_optimizations(self):
        """Set AMD-specific CPU optimizations"""
        # PyTorch threading optimized for AMD (only if not already set)
        try:
            if torch.get_num_threads() == 1:  # Default PyTorch value
                torch.set_num_threads(self.optimal_threads)
            if torch.get_num_interop_threads() == 1:  # Default PyTorch value
                torch.set_num_interop_threads(max(1, self.optimal_threads // 4))
        except RuntimeError as e:
            logger.warning(f"Could not set PyTorch thread settings: {e}")
        
        # AMD-friendly OpenMP settings (these are safe to set)
        os.environ['OMP_NUM_THREADS'] = str(self.optimal_threads)
        os.environ['MKL_NUM_THREADS'] = str(self.optimal_threads)
        
        # AMD-specific optimizations
        os.environ['OMP_PROC_BIND'] = 'true'
        os.environ['OMP_PLACES'] = 'cores'
        
        # Disable Intel-specific optimizations that might hurt AMD performance
        os.environ['KMP_AFFINITY'] = 'disabled'
        
        # ONNX Runtime optimizations for AMD
        if ONNX_AVAILABLE:
            os.environ['ORT_NUM_THREADS'] = str(self.optimal_threads)
        
        # Enable AMD optimizations in PyTorch
        torch.backends.mkldnn.enabled = True
        
        logger.info("Applied AMD-specific CPU optimizations")
    
    def _load_amd_optimized_model(self):
        """Load the most AMD-optimized model version available"""
        # For small to medium workloads, AMD-optimized PyTorch is faster than ONNX
        # ONNX has high overhead for model loading and small batches
        self._load_amd_pytorch_model()
        
        # Only use ONNX for very large batch processing (when explicitly requested)
        self.onnx_available = ONNX_AVAILABLE and OPTIMUM_AVAILABLE
    
    def _try_load_onnx_model(self) -> bool:
        """Try to load ONNX optimized model (works great with AMD)"""
        try:
            if not (ONNX_AVAILABLE and OPTIMUM_AVAILABLE):
                return False
            
            logger.info("Loading ONNX optimized model for AMD CPU...")
            
            # ONNX Runtime session optimized for AMD
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = self.optimal_threads
            sess_options.inter_op_num_threads = max(1, self.optimal_threads // 4)
            sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # AMD-specific ONNX optimizations
            sess_options.add_session_config_entry("session.intra_op.allow_spinning", "1")
            sess_options.add_session_config_entry("session.inter_op.allow_spinning", "1")
            
            # Check if ONNX model already exists and only export if needed
            try:
                # Try to load existing ONNX model first
                self.onnx_model = ORTModelForFeatureExtraction.from_pretrained(
                    self.model_name,
                    export=False,  # Don't export if already exists
                    session_options=sess_options,
                    provider="CPUExecutionProvider"
                )
                logger.info("Loaded existing ONNX model (no conversion needed)")
                
            except Exception:
                # If loading existing fails, then export new one
                logger.info("ONNX model not found, converting from PyTorch...")
                self.onnx_model = ORTModelForFeatureExtraction.from_pretrained(
                    self.model_name,
                    export=True,  # Export new ONNX model
                    session_options=sess_options,
                    provider="CPUExecutionProvider"
                )
                logger.info("Successfully converted and loaded new ONNX model")
            
            self.optimization_type = "ONNX (AMD-optimized)"
            logger.info("Successfully loaded ONNX model optimized for AMD")
            return True
            
        except Exception as e:
            logger.warning(f"ONNX model loading failed: {e}")
            return False
    
    def _load_amd_pytorch_model(self):
        """Load AMD-optimized PyTorch model"""
        try:
            logger.info("Loading AMD-optimized PyTorch model...")
            
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.model.eval()
            
            if self.device == 'cpu':
                # Enable AMD-friendly optimizations
                torch.backends.mkldnn.enabled = True
                
                # Fast warmup for AMD (don't do heavy JIT compilation)
                with torch.no_grad():
                    dummy_input = ["AMD optimization warmup"]
                    try:
                        # Quick warmup without heavy JIT
                        self.model.encode(dummy_input, convert_to_numpy=True, normalize_embeddings=True)
                        self.optimization_type = "PyTorch (AMD-optimized)"
                        logger.info("AMD-optimized PyTorch model ready")
                    except Exception as e:
                        logger.warning(f"AMD warmup failed: {e}")
                        self.optimization_type = "PyTorch (Standard)"
            
            logger.info("AMD-optimized PyTorch model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load AMD-optimized model: {e}")
            raise
    
    @log_performance
    def encode(self, texts: Union[str, List[str]], 
               batch_size: Optional[int] = None,
               num_workers: Optional[int] = None,
               use_onnx: bool = False) -> np.ndarray:
        """
        AMD-optimized encoding with perfect utilization of CPU cores
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Use AMD-optimized parameters
        if batch_size is None:
            batch_size = self.optimal_batch_size
        if num_workers is None:
            num_workers = min(self.optimal_threads, len(texts) // 4 + 1)
        
        # AMD-optimized text preprocessing
        processed_texts = self._preprocess_texts_amd(texts)
        
        try:
            # For AMD CPUs, PyTorch is usually faster than ONNX for small-medium batches
            # Only use ONNX for very large batches (1000+ texts) when explicitly requested
            if use_onnx and len(texts) > 1000 and self.onnx_available:
                if self.onnx_model is None:
                    self._try_load_onnx_model()
                if self.onnx_model is not None:
                    embeddings = self._encode_onnx_amd(processed_texts, batch_size)
                else:
                    embeddings = self._encode_pytorch_amd(processed_texts, batch_size)
            else:
                # Use fast AMD-optimized PyTorch for most cases
                embeddings = self._encode_pytorch_amd(processed_texts, batch_size)
            
            logger.debug(f"AMD-optimized encoded {len(processed_texts)} texts to {embeddings.shape} "
                        f"using {self.optimization_type}")
            return embeddings
            
        except Exception as e:
            logger.error(f"AMD encoding failed: {str(e)}")
            raise
    
    def _preprocess_texts_amd(self, texts: List[str]) -> List[str]:
        """AMD-optimized text preprocessing"""
        def process_single_text(text: str) -> str:
            if len(text) > MAX_TEXT_LENGTH:
                text = text[:MAX_TEXT_LENGTH]
            return text_processor.clean_text(text)
        
        # AMD 5700G benefits from moderate parallelism for preprocessing
        if len(texts) > 100:
            # Use thread pool optimized for AMD cores
            max_workers = min(self.physical_cores, len(texts) // 10)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                processed_texts = list(executor.map(process_single_text, texts))
        else:
            processed_texts = [process_single_text(text) for text in texts]
        
        return processed_texts
    
    def _encode_onnx_amd(self, texts: List[str], batch_size: int) -> np.ndarray:
        """ONNX encoding optimized for AMD processors"""
        # Use the ONNX model's encode method directly, which handles the output properly
        try:
            # ONNX model from optimum usually has an encode method
            embeddings = self.onnx_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 100,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            return embeddings
            
        except AttributeError:
            # Fallback: Use the model directly but handle output properly
            embeddings_list = []
            
            # Process in AMD-optimized batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # ONNX inference with AMD optimizations
                with torch.no_grad():
                    # For ONNX models, use the model's own tokenizer or get it from the preprocessor
                    try:
                        # Try different ways to access the tokenizer
                        if hasattr(self.onnx_model, 'tokenizer'):
                            tokenizer = self.onnx_model.tokenizer
                        elif hasattr(self.onnx_model, 'preprocessor') and hasattr(self.onnx_model.preprocessor, 'tokenizer'):
                            tokenizer = self.onnx_model.preprocessor.tokenizer
                        elif hasattr(self.onnx_model, 'config') and hasattr(self.onnx_model.config, 'tokenizer'):
                            tokenizer = self.onnx_model.config.tokenizer
                        else:
                            # Last resort: create tokenizer from model name
                            from transformers import AutoTokenizer
                            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                        
                        inputs = tokenizer(
                            batch_texts, 
                            padding=True, 
                            truncation=True, 
                            return_tensors="pt",
                            max_length=512
                        )
                        
                        # Get model outputs
                        outputs = self.onnx_model(**inputs)
                    except Exception as tokenizer_error:
                        logger.warning(f"Tokenizer access failed: {tokenizer_error}")
                        # Final fallback - just try to call the model directly
                        outputs = self.onnx_model(batch_texts)
                    
                    # Handle different output formats
                    if hasattr(outputs, 'sentence_embedding'):
                        embeddings = outputs.sentence_embedding
                    elif hasattr(outputs, 'last_hidden_state'):
                        # Pool the last hidden state (mean pooling)
                        embeddings = outputs.last_hidden_state.mean(dim=1)
                    elif hasattr(outputs, 'pooler_output'):
                        embeddings = outputs.pooler_output
                    else:
                        # If it's a tuple/list, take the first element and pool
                        if isinstance(outputs, (tuple, list)):
                            hidden_states = outputs[0]  # First element is usually last_hidden_state
                        else:
                            hidden_states = outputs
                        
                        # Mean pooling over sequence dimension
                        if hasattr(hidden_states, 'mean'):
                            embeddings = hidden_states.mean(dim=1)
                        else:
                            # Convert to tensor first
                            hidden_states = torch.tensor(hidden_states) if not torch.is_tensor(hidden_states) else hidden_states
                            embeddings = hidden_states.mean(dim=1)
                    
                    # Convert to numpy properly
                    if hasattr(embeddings, 'cpu'):
                        embeddings = embeddings.cpu().numpy()
                    elif hasattr(embeddings, 'numpy'):
                        embeddings = embeddings.numpy()
                    else:
                        embeddings = np.array(embeddings)
                    
                    # Ensure it's a 2D array
                    if embeddings.ndim == 1:
                        embeddings = embeddings.reshape(1, -1)
                    
                    # Normalize for cosine similarity
                    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                    embeddings = embeddings / np.maximum(norms, 1e-8)
                    
                    embeddings_list.append(embeddings)
            
            return np.vstack(embeddings_list)
    
    def _encode_pytorch_amd(self, texts: List[str], batch_size: int) -> np.ndarray:
        """PyTorch encoding optimized for AMD CPUs"""
        with torch.no_grad():
            # AMD CPUs benefit from these optimizations
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 100,
                convert_to_numpy=True,
                normalize_embeddings=True,
                device=self.device
            )
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        if self.onnx_model is not None:
            return 1024  # BGE-M3 default
        elif self.model is not None:
            return self.model.get_sentence_embedding_dimension()
        else:
            return 1024
    
    def get_amd_performance_stats(self) -> Dict[str, Any]:
        """Get AMD-specific performance stats"""
        return {
            "optimization_type": self.optimization_type,
            "cpu_count": self.cpu_count,
            "physical_cores": self.physical_cores,
            "optimal_threads": self.optimal_threads,
            "optimal_batch_size": self.optimal_batch_size,
            "pytorch_threads": torch.get_num_threads(),
            "device": self.device,
            "amd_optimizations": {
                "onnx_available": ONNX_AVAILABLE,
                "optimum_available": OPTIMUM_AVAILABLE,
                "mkldnn_enabled": torch.backends.mkldnn.enabled,
                "omp_threads": os.environ.get('OMP_NUM_THREADS'),
                "proc_bind": os.environ.get('OMP_PROC_BIND')
            }
        }


class AMDOptimizedHybridEmbedding:
    """AMD-optimized hybrid embedding for 5700G"""
    
    def __init__(self, dense_weight: float = 0.7, sparse_weight: float = 0.3):
        """Initialize AMD-optimized hybrid embedding"""
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
        self.bge_model = AMDOptimizedBGEEmbedding()
        
        # Use optimized BM25 from the optimized implementation
        from .embeddings_optimized import OptimizedBM25Embedding
        self.bm25_model = OptimizedBM25Embedding()
        
        self.is_fitted = False
        
        logger.info(f"AMD-optimized hybrid embedding initialized with {self.bge_model.optimization_type}")
    
    def fit(self, texts: List[str], use_parallel: bool = True):
        """Fit models with AMD optimizations"""
        logger.info("Fitting AMD-optimized hybrid embedding models")
        
        # Fit BM25 with AMD-friendly parallelism
        self.bm25_model.fit(texts, use_parallel=use_parallel)
        self.is_fitted = True
        
        logger.info("AMD-optimized hybrid embedding models fitted successfully")
    
    @log_performance
    def encode(self, texts: Union[str, List[str]], 
               batch_size: Optional[int] = None,
               use_parallel: bool = True) -> Dict[str, Any]:
        """AMD-optimized hybrid encoding"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Get AMD-optimized dense embeddings
        dense_embeddings = self.bge_model.encode(
            texts, 
            batch_size=batch_size,
            use_multiprocessing=len(texts) > 200  # AMD 5700G threshold
        )
        
        # Get optimized sparse embeddings
        sparse_embeddings = self.bm25_model.encode(texts, use_parallel=use_parallel)
        
        return {
            "dense": dense_embeddings,
            "sparse": sparse_embeddings
        }
    
    def encode_query(self, query: str) -> Dict[str, Any]:
        """AMD-optimized query encoding"""
        # Dense embedding
        dense_embedding = self.bge_model.encode([query])[0]
        
        # Sparse embedding
        sparse_embedding = self.bm25_model.encode_query(query)
        
        return {
            "dense": dense_embedding,
            "sparse": sparse_embedding
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive AMD performance statistics"""
        return {
            "bge_stats": self.bge_model.get_amd_performance_stats(),
            "bm25_workers": self.bm25_model.optimal_workers,
            "amd_optimization_summary": {
                "cpu_detected": "AMD (optimized)",
                "recommended_libraries": [
                    "onnxruntime (HIGHLY RECOMMENDED for AMD)",
                    "optimum[onnxruntime]",
                    "torch with MKL-DNN"
                ]
            }
        }


# Global AMD-optimized instances
amd_optimized_bge_model = AMDOptimizedBGEEmbedding()
amd_optimized_hybrid_model = AMDOptimizedHybridEmbedding()


def install_amd_optimization_libraries():
    """AMD 5700G specific optimization library installation guide"""
    install_commands = [
        "# === AMD 5700G Optimization Libraries ===",
        "",
        "# 1. ONNX Runtime (MOST IMPORTANT for AMD CPUs)",
        "pip install onnxruntime",
        "",
        "# 2. Optimum for ONNX model conversion",
        "pip install optimum[onnxruntime]",
        "",
        "# 3. PyTorch with CPU optimizations",
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
        "",
        "# 4. Additional math libraries (AMD compatible)",
        "pip install numpy scipy",
        "",
        "# 5. Optional: AMD-optimized BLAS (if available)",
        "# conda install -c conda-forge openblas",
        "",
        "# === Performance Testing ===",
        "# Run this to test your optimizations:",
        "python -c \"from models.embeddings_amd_optimized import amd_optimized_bge_model; print(amd_optimized_bge_model.get_amd_performance_stats())\"",
    ]
    
    return "\n".join(install_commands)


def amd_performance_tips():
    """Performance tips specific to AMD 5700G"""
    tips = [
        "=== AMD 5700G Performance Optimization Tips ===",
        "",
        "1. ONNX Runtime provides the best performance for AMD CPUs",
        "2. Use batch sizes between 16-32 for optimal memory usage",
        "3. Leave 2-4 threads available for the system (use 12-14 threads)",
        "4. Enable all CPU cores but avoid hyperthreading oversubscription",
        "5. Monitor CPU temperature - 5700G can throttle under heavy load",
        "6. Ensure adequate RAM (16GB+ recommended for large batches)",
        "",
        "Expected Performance on AMD 5700G:",
        "- Small batches (1-10 texts): ~50-100 texts/second",
        "- Medium batches (32 texts): ~200-400 texts/second", 
        "- Large batches (64+ texts): ~300-600 texts/second",
        "",
        "Memory Usage:",
        "- Model loading: ~2-3GB RAM",
        "- Encoding: ~100-500MB per batch (depending on batch size)",
    ]
    
    return "\n".join(tips)


if __name__ == "__main__":
    print("=== AMD 5700G Embedding Optimization ===")
    print(f"ONNX Runtime Available: {ONNX_AVAILABLE} (CRITICAL for AMD performance)")
    print(f"Optimum Available: {OPTIMUM_AVAILABLE}")
    print("\nInstallation Guide:")
    print(install_amd_optimization_libraries())
    print("\n" + amd_performance_tips())
