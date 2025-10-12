"""
Model downloading utilities with caching support
"""
import os
import shutil
import tempfile
import jieba
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from sentence_transformers import SentenceTransformer

from utils.logger import Logger
from utils.model_cache_manager import cache_manager
from config import BGE_M3_MODEL

logger = Logger.get_logger("hybrid_search.model_downloader")


class ModelDownloader:
    """Handles downloading and caching of all model types"""
    
    def __init__(self, cache_dir: str = "./model_cache"):
        """Initialize model downloader"""
        self.cache_manager = cache_manager
        self.cache_dir = Path(cache_dir)
        
        logger.info(f"Model downloader initialized with cache: {self.cache_dir}")
    
    def download_dense_model(self, model_name: str = BGE_M3_MODEL, force_redownload: bool = False) -> bool:
        """
        Download and cache dense embedding model (BGE-M3)
        
        Args:
            model_name: HuggingFace model name (e.g., 'BAAI/bge-small-zh-v1.5')
            force_redownload: Force redownload even if cached
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Downloading dense model: {model_name}")
        
        # Check if already cached and valid
        if not force_redownload and self.cache_manager.is_dense_model_cached(model_name):
            logger.info(f"Dense model {model_name} already cached, skipping download")
            return True
        
        try:
            # Get cache path
            cache_path = self.cache_manager.get_dense_model_cache_path(model_name)
            
            # Remove existing cache if force redownload
            if force_redownload and cache_path.exists():
                logger.info(f"Force redownload: removing existing cache at {cache_path}")
                shutil.rmtree(cache_path, ignore_errors=True)
            
            # Ensure cache directory exists
            cache_path.mkdir(parents=True, exist_ok=True)
            
            # Download model using SentenceTransformers (it handles caching internally)
            logger.info(f"Downloading {model_name} to {cache_path}...")
            
            # Set environment to use our cache directory
            original_cache = os.environ.get('SENTENCE_TRANSFORMERS_HOME')
            os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(self.cache_manager.dense_cache_dir)
            
            try:
                # This will download and cache the model
                model = SentenceTransformer(model_name, device='cpu')
                
                # Verify model was downloaded correctly
                embedding_dim = model.get_sentence_embedding_dimension()
                
                # Test a simple encoding to ensure model works
                test_embedding = model.encode(["测试文本 test text"], convert_to_numpy=True)
                
                logger.info(f"Dense model downloaded successfully: {model_name}")
                logger.info(f"Model dimensions: {embedding_dim}, test encoding shape: {test_embedding.shape}")
                
                # Register in cache manager
                model_metadata = {
                    "embedding_dim": embedding_dim,
                    "model_type": "sentence_transformer",
                    "device": "cpu",
                    "test_encoding_shape": test_embedding.shape
                }
                
                # Find actual cache path (SentenceTransformers creates subdirectory)
                actual_cache_path = self._find_sentence_transformer_cache_path(model_name)
                if actual_cache_path:
                    self.cache_manager.register_model("dense", model_name, actual_cache_path, model_metadata)
                else:
                    logger.warning(f"Could not find actual cache path for {model_name}")
                
                return True
                
            finally:
                # Restore original cache environment
                if original_cache:
                    os.environ['SENTENCE_TRANSFORMERS_HOME'] = original_cache
                elif 'SENTENCE_TRANSFORMERS_HOME' in os.environ:
                    del os.environ['SENTENCE_TRANSFORMERS_HOME']
        
        except Exception as e:
            logger.error(f"Failed to download dense model {model_name}: {str(e)}")
            
            # Clean up partial download
            if cache_path.exists():
                try:
                    shutil.rmtree(cache_path, ignore_errors=True)
                except:
                    pass
            
            return False
    
    def _find_sentence_transformer_cache_path(self, model_name: str) -> Optional[Path]:
        """Find the actual cache path created by SentenceTransformers"""
        cache_base = self.cache_manager.dense_cache_dir
        
        # HuggingFace cache format: models--BAAI--bge-small-zh-v1.5
        hf_cache_name = f"models--{model_name.replace('/', '--')}"
        hf_cache_path = cache_base / hf_cache_name
        
        if hf_cache_path.exists():
            # Find the snapshot directory with config.json
            for snapshot_path in hf_cache_path.rglob("config.json"):
                if "snapshots" in str(snapshot_path):
                    return snapshot_path.parent
        
        # Fallback: look for the old format directory
        clean_name = model_name.replace('/', '_')
        old_format_path = cache_base / clean_name
        if old_format_path.exists() and (old_format_path / "config.json").exists():
            return old_format_path
        
        # Final fallback: any directory with config.json that matches model name
        for path in cache_base.rglob("config.json"):
            if model_name.replace('/', '_') in str(path.parent) or model_name.replace('/', '--') in str(path.parent):
                return path.parent
        
        return None
    
    def download_jieba_model(self, force_redownload: bool = False) -> bool:
        """
        Download and cache jieba dictionaries
        
        Args:
            force_redownload: Force redownload even if cached
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Setting up jieba model cache")
        
        # Check if already cached
        if not force_redownload and self.cache_manager.is_jieba_cached():
            logger.info("Jieba dictionaries already cached, skipping download")
            return True
        
        try:
            jieba_cache_path = self.cache_manager.get_jieba_cache_path()
            
            # Remove existing cache if force redownload
            if force_redownload and jieba_cache_path.exists():
                logger.info(f"Force redownload: removing existing jieba cache at {jieba_cache_path}")
                shutil.rmtree(jieba_cache_path, ignore_errors=True)
            
            # Ensure cache directory exists
            jieba_cache_path.mkdir(parents=True, exist_ok=True)
            
            # Configure jieba to use our cache directory
            original_jieba_cache = getattr(jieba, 'cache_file', None)
            
            # Set jieba cache to our directory
            jieba.dt.cache_file = str(jieba_cache_path / "jieba.cache")
            
            # Initialize jieba (this will download dictionaries if needed)
            logger.info("Initializing jieba and downloading dictionaries...")
            jieba.initialize()
            
            # Force jieba to load and cache dictionaries by doing some operations
            test_texts = [
                "中文分词测试",
                "自然语言处理技术",
                "机器学习算法",
                "软件工程师招聘"
            ]
            
            for text in test_texts:
                list(jieba.cut(text))
                list(jieba.cut_for_search(text))
            
            # Copy jieba's default dictionary to our cache
            try:
                jieba_module_path = Path(jieba.__file__).parent
                default_dict = jieba_module_path / "dict.txt"
                
                if default_dict.exists():
                    cache_dict = jieba_cache_path / "dict.txt"
                    shutil.copy2(default_dict, cache_dict)
                    logger.info(f"Copied jieba dictionary to cache: {cache_dict}")
                
                # Also copy user dictionary if it exists
                user_dict = jieba_module_path / "userdict.txt"
                if user_dict.exists():
                    cache_user_dict = jieba_cache_path / "userdict.txt"
                    shutil.copy2(user_dict, cache_user_dict)
                    logger.info(f"Copied jieba user dictionary to cache: {cache_user_dict}")
                    
            except Exception as e:
                logger.warning(f"Failed to copy jieba dictionaries: {e}")
            
            # Create a custom setup script for jieba caching
            setup_script = jieba_cache_path / "setup_jieba_cache.py"
            setup_content = f'''"""
Jieba cache setup script
Generated on: {datetime.now().isoformat()}
"""
import jieba
import os
from pathlib import Path

def setup_jieba_cache():
    """Configure jieba to use project cache directory"""
    cache_dir = Path(__file__).parent
    
    # Set cache file location
    jieba.dt.cache_file = str(cache_dir / "jieba.cache")
    
    # Initialize jieba
    jieba.initialize()
    
    # Load custom dictionaries if they exist
    user_dict = cache_dir / "userdict.txt"
    if user_dict.exists():
        jieba.load_userdict(str(user_dict))
    
    print(f"Jieba cache configured: {{cache_dir}}")

if __name__ == "__main__":
    setup_jieba_cache()
'''
            
            with open(setup_script, 'w', encoding='utf-8') as f:
                f.write(setup_content)
            
            logger.info("Created jieba setup script")
            
            # Register in cache manager
            jieba_metadata = {
                "dictionary_files": list(jieba_cache_path.glob("*.txt")),
                "cache_files": list(jieba_cache_path.glob("*.cache")),
                "setup_script": str(setup_script)
            }
            
            self.cache_manager.register_model("jieba", "default", jieba_cache_path, jieba_metadata)
            
            logger.info(f"Jieba model cached successfully at {jieba_cache_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup jieba model cache: {str(e)}")
            return False
    
    def download_all_models(self, force_redownload: bool = False) -> Dict[str, bool]:
        """
        Download all required models
        
        Args:
            force_redownload: Force redownload even if cached
            
        Returns:
            Dict with download results for each model type
        """
        logger.info("Starting download of all required models")
        
        results = {
            "dense_model": False,
            "jieba_model": False
        }
        
        # Download dense model
        try:
            results["dense_model"] = self.download_dense_model(BGE_M3_MODEL, force_redownload)
        except Exception as e:
            logger.error(f"Dense model download failed: {e}")
        
        # Download jieba model
        try:
            results["jieba_model"] = self.download_jieba_model(force_redownload)
        except Exception as e:
            logger.error(f"Jieba model download failed: {e}")
        
        # Summary
        successful = sum(results.values())
        total = len(results)
        
        logger.info(f"Model download completed: {successful}/{total} successful")
        
        if successful == total:
            logger.info("All models downloaded successfully!")
        else:
            logger.warning(f"Some models failed to download: {results}")
        
        return results
    
    def verify_cached_models(self) -> Dict[str, Any]:
        """Verify all cached models are working correctly"""
        logger.info("Verifying cached models...")
        
        verification_results = {
            "dense_model": {"cached": False, "working": False, "error": None},
            "jieba_model": {"cached": False, "working": False, "error": None},
            "bm25_model": {"cached": False, "working": False, "error": None}
        }
        
        # Verify dense model
        try:
            if self.cache_manager.is_dense_model_cached(BGE_M3_MODEL):
                verification_results["dense_model"]["cached"] = True
                
                # Test loading and encoding
                os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(self.cache_manager.dense_cache_dir)
                model = SentenceTransformer(BGE_M3_MODEL, device='cpu')
                test_embedding = model.encode(["test text"], convert_to_numpy=True)
                
                if test_embedding.shape[0] == 1 and test_embedding.shape[1] > 0:
                    verification_results["dense_model"]["working"] = True
                else:
                    verification_results["dense_model"]["error"] = f"Invalid embedding shape: {test_embedding.shape}"
                    
        except Exception as e:
            verification_results["dense_model"]["error"] = str(e)
        
        # Verify jieba model
        try:
            if self.cache_manager.is_jieba_cached():
                verification_results["jieba_model"]["cached"] = True
                
                # Test jieba functionality
                jieba_cache_path = self.cache_manager.get_jieba_cache_path()
                jieba.dt.cache_file = str(jieba_cache_path / "jieba.cache")
                
                test_result = list(jieba.cut("测试中文分词"))
                if len(test_result) > 0:
                    verification_results["jieba_model"]["working"] = True
                else:
                    verification_results["jieba_model"]["error"] = "Jieba returned empty result"
                    
        except Exception as e:
            verification_results["jieba_model"]["error"] = str(e)
        
        # Verify BM25 model (existing)
        try:
            bm25_path = self.cache_manager.get_bm25_cache_path()
            if bm25_path.exists():
                verification_results["bm25_model"]["cached"] = True
                
                # Try to load the pickle file
                import pickle
                with open(bm25_path, 'rb') as f:
                    bm25_data = pickle.load(f)
                
                if isinstance(bm25_data, dict) and 'doc_freqs' in bm25_data:
                    verification_results["bm25_model"]["working"] = True
                else:
                    verification_results["bm25_model"]["error"] = "Invalid BM25 cache structure"
                    
        except Exception as e:
            verification_results["bm25_model"]["error"] = str(e)
        
        # Summary
        working_models = sum(1 for result in verification_results.values() if result["working"])
        cached_models = sum(1 for result in verification_results.values() if result["cached"])
        
        logger.info(f"Model verification completed: {cached_models} cached, {working_models} working")
        
        return verification_results


# Global model downloader instance
model_downloader = ModelDownloader()
