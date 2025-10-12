"""
Unified model cache management system for hybrid search models
"""
import os
import json
import pickle
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import tempfile

from utils.logger import Logger

logger = Logger.get_logger("hybrid_search.model_cache_manager")


class ModelCacheManager:
    """Centralized cache management for all model types"""
    
    def __init__(self, cache_dir: str = "./model_cache"):
        """Initialize cache manager"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache metadata file
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
        
        # Subdirectories for different model types
        self.dense_cache_dir = self.cache_dir / "sentence_transformers"
        self.jieba_cache_dir = self.cache_dir / "jieba"
        self.bm25_cache_dir = self.cache_dir  # Keep BM25 in root for compatibility
        
        # Ensure subdirectories exist
        self.dense_cache_dir.mkdir(parents=True, exist_ok=True)
        self.jieba_cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Model cache manager initialized: {self.cache_dir}")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                logger.debug(f"Loaded cache metadata: {len(metadata)} entries")
                return metadata
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        
        # Return default metadata
        return {
            "created_at": datetime.now().isoformat(),
            "models": {},
            "stats": {
                "total_models": 0,
                "total_size_mb": 0,
                "last_cleanup": None
            }
        }
    
    def _save_metadata(self):
        """Save cache metadata atomically"""
        try:
            # Update stats
            self.metadata["stats"]["last_updated"] = datetime.now().isoformat()
            self.metadata["stats"]["total_models"] = len(self.metadata["models"])
            
            # Atomic write using temporary file
            temp_path = self.metadata_file.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            
            # Atomic move
            temp_path.replace(self.metadata_file)
            
            logger.debug(f"Cache metadata saved: {len(self.metadata['models'])} models")
            
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
    
    def get_dense_model_cache_path(self, model_name: str) -> Path:
        """Get cache path for dense model"""
        # Clean model name for filesystem
        clean_name = model_name.replace('/', '_').replace('\\', '_')
        return self.dense_cache_dir / clean_name
    
    def get_jieba_cache_path(self) -> Path:
        """Get cache path for jieba dictionaries"""
        return self.jieba_cache_dir
    
    def get_bm25_cache_path(self) -> Path:
        """Get cache path for BM25 model (maintain compatibility)"""
        return self.cache_dir / "bm25_latest.pkl"
    
    def is_dense_model_cached(self, model_name: str) -> bool:
        """Check if dense model is cached"""
        # Check HuggingFace cache format first
        hf_cache_name = f"models--{model_name.replace('/', '--')}"
        hf_cache_path = self.dense_cache_dir / hf_cache_name
        
        if hf_cache_path.exists():
            # Look for config.json in snapshots directory
            for config_path in hf_cache_path.rglob("config.json"):
                if "snapshots" in str(config_path):
                    return True
        
        # Fallback to old format
        cache_path = self.get_dense_model_cache_path(model_name)
        return cache_path.exists() and (cache_path / "config.json").exists()
    
    def is_jieba_cached(self) -> bool:
        """Check if jieba dictionaries are cached"""
        jieba_cache = self.get_jieba_cache_path()
        return jieba_cache.exists() and len(list(jieba_cache.glob("*.txt"))) > 0
    
    def register_model(self, model_type: str, model_name: str, 
                      cache_path: Path, metadata: Optional[Dict] = None):
        """Register a cached model in metadata"""
        model_key = f"{model_type}:{model_name}"
        
        model_info = {
            "type": model_type,
            "name": model_name,
            "cache_path": str(cache_path),
            "cached_at": datetime.now().isoformat(),
            "size_mb": self._calculate_directory_size(cache_path) if cache_path.exists() else 0
        }
        
        if metadata:
            # Convert any Path objects to strings for JSON serialization
            safe_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, Path):
                    safe_metadata[key] = str(value)
                elif isinstance(value, list) and value and isinstance(value[0], Path):
                    safe_metadata[key] = [str(p) for p in value]
                else:
                    safe_metadata[key] = value
            model_info.update(safe_metadata)
        
        self.metadata["models"][model_key] = model_info
        self._save_metadata()
        
        logger.info(f"Registered cached model: {model_key}")
    
    def get_model_info(self, model_type: str, model_name: str) -> Optional[Dict]:
        """Get cached model information"""
        model_key = f"{model_type}:{model_name}"
        return self.metadata["models"].get(model_key)
    
    def _calculate_directory_size(self, directory: Path) -> float:
        """Calculate directory size in MB"""
        if not directory.exists():
            return 0.0
        
        total_size = 0
        try:
            for path in directory.rglob('*'):
                if path.is_file():
                    total_size += path.stat().st_size
        except Exception as e:
            logger.warning(f"Failed to calculate size for {directory}: {e}")
            return 0.0
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    def cleanup_cache(self, max_age_days: int = 30, max_size_mb: int = 5000):
        """Clean up old or excessive cache entries"""
        logger.info(f"Starting cache cleanup: max_age={max_age_days}days, max_size={max_size_mb}MB")
        
        current_time = datetime.now()
        total_size = 0
        models_to_remove = []
        
        # Calculate current cache size and identify old models
        for model_key, model_info in self.metadata["models"].items():
            try:
                cache_path = Path(model_info["cache_path"])
                if cache_path.exists():
                    size = self._calculate_directory_size(cache_path)
                    total_size += size
                    
                    # Check age
                    cached_at = datetime.fromisoformat(model_info["cached_at"])
                    age_days = (current_time - cached_at).days
                    
                    if age_days > max_age_days:
                        models_to_remove.append((model_key, cache_path, f"age: {age_days} days"))
                else:
                    # Cache path doesn't exist, remove from metadata
                    models_to_remove.append((model_key, cache_path, "path not found"))
                    
            except Exception as e:
                logger.warning(f"Error checking model {model_key}: {e}")
                models_to_remove.append((model_key, None, f"error: {e}"))
        
        # Remove old models
        for model_key, cache_path, reason in models_to_remove:
            logger.info(f"Removing cached model {model_key}: {reason}")
            if cache_path and cache_path.exists():
                try:
                    shutil.rmtree(cache_path)
                except Exception as e:
                    logger.error(f"Failed to remove {cache_path}: {e}")
            
            del self.metadata["models"][model_key]
        
        # Update metadata
        self.metadata["stats"]["last_cleanup"] = current_time.isoformat()
        self.metadata["stats"]["total_size_mb"] = total_size
        self._save_metadata()
        
        logger.info(f"Cache cleanup completed: removed {len(models_to_remove)} models, "
                   f"total size: {total_size:.1f}MB")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        stats = {
            "cache_dir": str(self.cache_dir),
            "total_models": len(self.metadata["models"]),
            "total_size_mb": 0,
            "models_by_type": {},
            "disk_usage": {}
        }
        
        # Calculate current sizes and categorize by type
        for model_key, model_info in self.metadata["models"].items():
            model_type = model_info["type"]
            cache_path = Path(model_info["cache_path"])
            
            if cache_path.exists():
                size = self._calculate_directory_size(cache_path)
                stats["total_size_mb"] += size
                
                if model_type not in stats["models_by_type"]:
                    stats["models_by_type"][model_type] = {"count": 0, "size_mb": 0}
                
                stats["models_by_type"][model_type]["count"] += 1
                stats["models_by_type"][model_type]["size_mb"] += size
        
        # Get disk usage for cache directory
        try:
            disk_usage = shutil.disk_usage(self.cache_dir)
            stats["disk_usage"] = {
                "total_gb": disk_usage.total / (1024**3),
                "used_gb": disk_usage.used / (1024**3),
                "free_gb": disk_usage.free / (1024**3)
            }
        except Exception as e:
            logger.warning(f"Failed to get disk usage: {e}")
            stats["disk_usage"] = {"error": str(e)}
        
        return stats
    
    def validate_cache_integrity(self) -> Dict[str, Any]:
        """Validate cache integrity and report issues"""
        results = {
            "valid_models": [],
            "invalid_models": [],
            "missing_models": [],
            "orphaned_paths": []
        }
        
        # Check registered models
        for model_key, model_info in self.metadata["models"].items():
            cache_path = Path(model_info["cache_path"])
            
            if not cache_path.exists():
                results["missing_models"].append({
                    "key": model_key,
                    "path": str(cache_path),
                    "reason": "Path does not exist"
                })
            elif model_info["type"] == "dense" and not (cache_path / "config.json").exists():
                results["invalid_models"].append({
                    "key": model_key,
                    "path": str(cache_path),
                    "reason": "Missing config.json for dense model"
                })
            else:
                results["valid_models"].append(model_key)
        
        # Check for orphaned directories
        for subdir in [self.dense_cache_dir, self.jieba_cache_dir]:
            if subdir.exists():
                for path in subdir.iterdir():
                    if path.is_dir():
                        # Check if this path is registered
                        found = False
                        for model_info in self.metadata["models"].values():
                            if Path(model_info["cache_path"]) == path:
                                found = True
                                break
                        
                        if not found:
                            results["orphaned_paths"].append(str(path))
        
        logger.info(f"Cache validation: {len(results['valid_models'])} valid, "
                   f"{len(results['invalid_models'])} invalid, "
                   f"{len(results['missing_models'])} missing, "
                   f"{len(results['orphaned_paths'])} orphaned")
        
        return results


# Global cache manager instance
cache_manager = ModelCacheManager()
