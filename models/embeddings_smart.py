"""
Smart Embedding Factory with Automatic CPU Detection and Optimization Selection
Automatically selects the best embedding implementation based on detected CPU characteristics
"""
import os
import warnings
from typing import Dict, Any, Optional, Union, List
from utils.logger import Logger
from utils.cpu_detector import cpu_detector, CPUInfo

logger = Logger.get_logger("hybrid_search.embeddings_smart")


class SmartEmbeddingFactory:
    """Factory that automatically selects optimal embedding implementation"""
    
    def __init__(self, force_optimization: Optional[str] = None):
        """
        Initialize smart factory
        
        Args:
            force_optimization: Override automatic detection ('standard', 'optimized', 
                              'intel_optimized')
        """
        self.cpu_info = cpu_detector.get_cpu_info()
        self.force_optimization = force_optimization
        self.selected_optimization = self._select_optimization()
        
        # Cache for loaded models
        self._bge_model = None
        self._hybrid_model = None
        
        logger.info(f"Smart factory initialized: {self.selected_optimization} "
                   f"for {self.cpu_info.vendor} {self.cpu_info.model}")
    
    def _select_optimization(self) -> str:
        """Select the best optimization based on CPU and availability"""
        if self.force_optimization:
            logger.info(f"Using forced optimization: {self.force_optimization}")
            return self.force_optimization
        
        recommended = self.cpu_info.recommended_optimization
        
        # Check if recommended optimization is available
        if self._is_optimization_available(recommended):
            logger.info(f"Using recommended optimization: {recommended}")
            return recommended
        
        # Fallback hierarchy - removed amd_optimized, use optimized for all non-Intel CPUs
        fallback_order = ['intel_optimized', 'optimized', 'standard']
        
        for optimization in fallback_order:
            if self._is_optimization_available(optimization):
                logger.info(f"Using fallback optimization: {optimization}")
                return optimization
        
        # Final fallback
        logger.warning("All optimizations failed, using standard implementation")
        return 'standard'
    
    def _is_optimization_available(self, optimization: str) -> bool:
        """Check if optimization implementation is available and functional"""
        try:
            if optimization == 'standard':
                from . import embeddings
                return True
            elif optimization == 'optimized':
                from . import embeddings_optimized
                return True
            elif optimization == 'intel_optimized':
                from . import embeddings_intel_optimized
                return True
            return False
        except ImportError as e:
            logger.warning(f"Optimization {optimization} not available: {e}")
            return False
        except Exception as e:
            logger.warning(f"Error checking optimization {optimization}: {e}")
            return False
    
    def get_bge_model(self):
        """Get optimized BGE model instance"""
        if self._bge_model is None:
            self._bge_model = self._load_bge_model()
        return self._bge_model
    
    def get_hybrid_model(self):
        """Get optimized hybrid model instance"""
        if self._hybrid_model is None:
            self._hybrid_model = self._load_hybrid_model()
        return self._hybrid_model
    
    def _load_bge_model(self):
        """Load the appropriate BGE model based on selected optimization"""
        try:
            if self.selected_optimization == 'intel_optimized':
                from .embeddings_intel_optimized import IntelOptimizedBGEEmbedding
                model = IntelOptimizedBGEEmbedding()
                logger.info("Loaded Intel-optimized BGE model")
                return model
            
            elif self.selected_optimization == 'optimized':
                from .embeddings_optimized import OptimizedBGEEmbedding
                model = OptimizedBGEEmbedding()
                logger.info("Loaded optimized BGE model")
                return model
            
            else:  # standard
                from .embeddings import BGEEmbedding
                model = BGEEmbedding()
                logger.info("Loaded standard BGE model")
                return model
                
        except Exception as e:
            logger.error(f"Failed to load {self.selected_optimization} BGE model: {e}")
            # Fallback to standard
            from .embeddings import BGEEmbedding
            return BGEEmbedding()
    
    def _load_hybrid_model(self):
        """Load the appropriate hybrid model based on selected optimization"""
        try:
            if self.selected_optimization == 'intel_optimized':
                from .embeddings_intel_optimized import IntelOptimizedHybridEmbedding
                model = IntelOptimizedHybridEmbedding()
                logger.info("Loaded Intel-optimized hybrid model")
                return model
            
            elif self.selected_optimization == 'optimized':
                from .embeddings_optimized import OptimizedHybridEmbedding
                model = OptimizedHybridEmbedding()
                logger.info("Loaded optimized hybrid model")
                return model
            
            else:  # standard
                from .embeddings import HybridEmbedding
                model = HybridEmbedding()
                logger.info("Loaded standard hybrid model")
                return model
                
        except Exception as e:
            logger.error(f"Failed to load {self.selected_optimization} hybrid model: {e}")
            # Fallback to standard
            from .embeddings import HybridEmbedding
            return HybridEmbedding()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        bge_model = self.get_bge_model()
        
        # Get model-specific stats if available
        model_stats = {}
        if hasattr(bge_model, 'get_amd_performance_stats'):
            model_stats = bge_model.get_amd_performance_stats()
        elif hasattr(bge_model, 'get_performance_stats'):
            model_stats = bge_model.get_performance_stats()
        
        return {
            'smart_factory': {
                'selected_optimization': self.selected_optimization,
                'forced_optimization': self.force_optimization,
                'cpu_recommendation': self.cpu_info.recommended_optimization,
            },
            'cpu_info': cpu_detector.get_performance_summary(),
            'model_stats': model_stats
        }
    
    def benchmark_all_available(self, test_texts: List[str], 
                               iterations: int = 3) -> Dict[str, Dict[str, Any]]:
        """Benchmark all available optimizations for comparison"""
        logger.info(f"Benchmarking all available optimizations with {len(test_texts)} texts")
        
        optimizations = ['standard', 'optimized', 'intel_optimized']
        results = {}
        
        for opt in optimizations:
            if self._is_optimization_available(opt):
                try:
                    logger.info(f"Benchmarking {opt} optimization...")
                    factory = SmartEmbeddingFactory(force_optimization=opt)
                    model = factory.get_bge_model()
                    
                    # Warm up
                    model.encode(test_texts[:2])
                    
                    # Benchmark
                    import time
                    times = []
                    for i in range(iterations):
                        start_time = time.time()
                        embeddings = model.encode(test_texts)
                        end_time = time.time()
                        times.append(end_time - start_time)
                    
                    avg_time = sum(times) / len(times)
                    texts_per_second = len(test_texts) / avg_time
                    
                    results[opt] = {
                        'avg_time_seconds': avg_time,
                        'texts_per_second': texts_per_second,
                        'embedding_shape': embeddings.shape if hasattr(embeddings, 'shape') else None,
                        'optimization_type': getattr(model, 'optimization_type', 'unknown')
                    }
                    
                    logger.info(f"{opt}: {texts_per_second:.1f} texts/second")
                    
                except Exception as e:
                    logger.error(f"Benchmark failed for {opt}: {e}")
                    results[opt] = {'error': str(e)}
        
        return results
    
    def print_selection_report(self):
        """Print detailed selection and performance report"""
        print("=== Smart Embedding Factory Report ===")
        print(f"Selected Optimization: {self.selected_optimization}")
        print(f"Forced Optimization: {self.force_optimization or 'None (auto-detected)'}")
        print(f"CPU Recommendation: {self.cpu_info.recommended_optimization}")
        
        print(f"\nCPU Information:")
        print(f"  Vendor: {self.cpu_info.vendor}")
        print(f"  Model: {self.cpu_info.model}")
        print(f"  Architecture: {self.cpu_info.architecture}")
        print(f"  Physical Cores: {self.cpu_info.physical_cores}")
        print(f"  Logical Cores: {self.cpu_info.logical_cores}")
        print(f"  Optimal Threads: {self.cpu_info.optimal_threads}")
        print(f"  Optimal Batch Size: {self.cpu_info.optimal_batch_size}")
        
        # Check availability of all optimizations
        print(f"\nOptimization Availability:")
        optimizations = ['standard', 'optimized', 'intel_optimized']
        for opt in optimizations:
            available = self._is_optimization_available(opt)
            status = "✓ Available" if available else "✗ Not Available"
            marker = " ← SELECTED" if opt == self.selected_optimization else ""
            print(f"  {opt}: {status}{marker}")


# Global smart factory instance (default auto-detection)
smart_factory = SmartEmbeddingFactory()

# Convenient global instances
smart_bge_model = smart_factory.get_bge_model()
smart_hybrid_model = smart_factory.get_hybrid_model()


def get_smart_bge_model(force_optimization: Optional[str] = None):
    """Get smart BGE model with optional optimization override"""
    if force_optimization:
        factory = SmartEmbeddingFactory(force_optimization=force_optimization)
        return factory.get_bge_model()
    return smart_bge_model


def get_smart_hybrid_model(force_optimization: Optional[str] = None):
    """Get smart hybrid model with optional optimization override"""
    if force_optimization:
        factory = SmartEmbeddingFactory(force_optimization=force_optimization)
        return factory.get_hybrid_model()
    return smart_hybrid_model


def benchmark_optimizations(test_texts: Optional[List[str]] = None, 
                          iterations: int = 3) -> Dict[str, Any]:
    """Benchmark all available optimizations"""
    if test_texts is None:
        test_texts = [
            f"This is test sentence number {i} for benchmarking embedding performance."
            for i in range(50)
        ]
    
    return smart_factory.benchmark_all_available(test_texts, iterations)


def print_smart_report():
    """Print comprehensive smart factory report"""
    smart_factory.print_selection_report()


# Configuration utilities
def force_optimization(optimization: str):
    """Force a specific optimization globally"""
    global smart_factory, smart_bge_model, smart_hybrid_model
    
    logger.info(f"Forcing optimization to: {optimization}")
    smart_factory = SmartEmbeddingFactory(force_optimization=optimization)
    smart_bge_model = smart_factory.get_bge_model()
    smart_hybrid_model = smart_factory.get_hybrid_model()


def reset_to_auto():
    """Reset to automatic optimization detection"""
    global smart_factory, smart_bge_model, smart_hybrid_model
    
    logger.info("Resetting to automatic optimization detection")
    smart_factory = SmartEmbeddingFactory()
    smart_bge_model = smart_factory.get_bge_model()
    smart_hybrid_model = smart_factory.get_hybrid_model()


# Environment variable support
def _check_environment_override():
    """Check for environment variable optimization override"""
    env_optimization = os.environ.get('EMBEDDING_OPTIMIZATION')
    if env_optimization:
        logger.info(f"Environment variable override: EMBEDDING_OPTIMIZATION={env_optimization}")
        force_optimization(env_optimization.lower())


# Initialize with environment override if present
_check_environment_override()


if __name__ == "__main__":
    print("=== Smart Embedding Factory Test ===")
    
    # Print selection report
    print_smart_report()
    
    # Test encoding
    print("\n=== Testing Encoding ===")
    test_texts = ["Hello world", "This is a test", "Smart embedding selection"]
    
    try:
        embeddings = smart_bge_model.encode(test_texts)
        print(f"Successfully encoded {len(test_texts)} texts to shape: {embeddings.shape}")
        
        hybrid_result = smart_hybrid_model.encode(test_texts)
        print(f"Hybrid encoding successful: dense={hybrid_result['dense'].shape}, "
              f"sparse={len(hybrid_result['sparse'])} vectors")
        
    except Exception as e:
        print(f"Encoding test failed: {e}")
    
    # Quick benchmark
    print("\n=== Quick Benchmark ===")
    try:
        benchmark_results = benchmark_optimizations(test_texts, iterations=1)
        for opt, results in benchmark_results.items():
            if 'error' not in results:
                print(f"{opt}: {results['texts_per_second']:.1f} texts/second")
            else:
                print(f"{opt}: ERROR - {results['error']}")
    except Exception as e:
        print(f"Benchmark failed: {e}")
