"""
Comprehensive Embedding Performance Benchmark and Validation Script
Tests and compares all available embedding optimizations
"""
import time
import sys
import os
import json
import psutil
from typing import List, Dict, Any, Optional
from statistics import mean, stdev

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logger import Logger
from utils.cpu_detector import cpu_detector
from models.embeddings_smart import (
    SmartEmbeddingFactory, 
    benchmark_optimizations,
    print_smart_report
)

logger = Logger.get_logger("hybrid_search.benchmark")


class EmbeddingBenchmark:
    """Comprehensive embedding performance benchmark"""
    
    def __init__(self):
        self.cpu_info = cpu_detector.get_cpu_info()
        self.test_datasets = self._create_test_datasets()
        self.results = {}
        
    def _create_test_datasets(self) -> Dict[str, List[str]]:
        """Create various test datasets of different sizes and characteristics"""
        datasets = {}
        
        # Small dataset (quick test)
        datasets['small'] = [
            f"This is a short test sentence number {i}."
            for i in range(10)
        ]
        
        # Medium dataset (typical batch)
        datasets['medium'] = [
            f"This is test sentence number {i} for evaluating embedding performance. "
            f"It contains multiple words and represents typical text length for search applications."
            for i in range(50)
        ]
        
        # Large dataset (stress test)
        datasets['large'] = [
            f"This is a comprehensive test sentence number {i} designed to evaluate the performance "
            f"of different embedding optimizations under various workloads. The sentence includes "
            f"multiple clauses, various vocabulary, and represents realistic text processing scenarios "
            f"that might be encountered in production search systems."
            for i in range(200)
        ]
        
        # Variable length dataset
        short_texts = [f"Short {i}" for i in range(25)]
        medium_texts = [f"Medium length sentence number {i} with more words" for i in range(25)]
        long_texts = [
            f"This is a longer text example number {i} that contains significantly more content "
            f"and tests how well the embedding models handle varying text lengths in a single batch."
            for i in range(25)
        ]
        datasets['variable'] = short_texts + medium_texts + long_texts
        
        return datasets
    
    def run_comprehensive_benchmark(self, iterations: int = 3) -> Dict[str, Any]:
        """Run comprehensive benchmark across all optimizations and datasets"""
        logger.info("Starting comprehensive embedding benchmark")
        logger.info(f"CPU: {self.cpu_info.vendor} {self.cpu_info.model}")
        logger.info(f"Cores: {self.cpu_info.physical_cores} physical, {self.cpu_info.logical_cores} logical")
        
        results = {
            'system_info': self._get_system_info(),
            'cpu_info': cpu_detector.get_performance_summary(),
            'benchmarks': {}
        }
        
        # Test each dataset
        for dataset_name, texts in self.test_datasets.items():
            logger.info(f"Benchmarking dataset: {dataset_name} ({len(texts)} texts)")
            results['benchmarks'][dataset_name] = self._benchmark_dataset(texts, iterations)
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _benchmark_dataset(self, texts: List[str], iterations: int) -> Dict[str, Any]:
        """Benchmark all optimizations on a specific dataset"""
        optimizations = ['standard', 'optimized', 'intel_optimized', 'amd_optimized']
        results = {}
        
        for opt in optimizations:
            logger.info(f"  Testing {opt} optimization...")
            results[opt] = self._benchmark_single_optimization(opt, texts, iterations)
            
            # Memory cleanup between tests
            import gc
            gc.collect()
            time.sleep(1)
        
        return results
    
    def _benchmark_single_optimization(self, optimization: str, texts: List[str], 
                                     iterations: int) -> Dict[str, Any]:
        """Benchmark a single optimization"""
        try:
            factory = SmartEmbeddingFactory(force_optimization=optimization)
            
            # Check if optimization is available
            if not factory._is_optimization_available(optimization):
                return {'error': f'Optimization {optimization} not available'}
            
            model = factory.get_bge_model()
            
            # Get optimization info
            optimization_info = {
                'optimization_type': getattr(model, 'optimization_type', 'unknown'),
                'device': getattr(model, 'device', 'unknown'),
                'model_name': getattr(model, 'model_name', 'unknown')
            }
            
            # Warmup (important for accurate benchmarking)
            logger.debug(f"    Warming up {optimization}...")
            try:
                warmup_texts = texts[:min(5, len(texts))]
                model.encode(warmup_texts)
            except Exception as e:
                logger.warning(f"    Warmup failed for {optimization}: {e}")
            
            # Benchmark iterations
            times = []
            memory_usage = []
            
            for i in range(iterations):
                # Measure memory before
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
                
                # Time the encoding
                start_time = time.perf_counter()
                embeddings = model.encode(texts)
                end_time = time.perf_counter()
                
                # Measure memory after
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                
                iteration_time = end_time - start_time
                times.append(iteration_time)
                memory_usage.append(memory_after - memory_before)
                
                logger.debug(f"    Iteration {i+1}: {iteration_time:.3f}s")
            
            # Calculate statistics
            avg_time = mean(times)
            time_std = stdev(times) if len(times) > 1 else 0
            texts_per_second = len(texts) / avg_time
            avg_memory = mean(memory_usage)
            
            return {
                'success': True,
                'optimization_info': optimization_info,
                'performance': {
                    'avg_time_seconds': avg_time,
                    'time_std_dev': time_std,
                    'texts_per_second': texts_per_second,
                    'avg_memory_mb': avg_memory,
                    'times_all_iterations': times
                },
                'output_info': {
                    'embedding_shape': list(embeddings.shape) if hasattr(embeddings, 'shape') else None,
                    'embedding_dtype': str(embeddings.dtype) if hasattr(embeddings, 'dtype') else None
                }
            }
            
        except Exception as e:
            logger.error(f"Benchmark failed for {optimization}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        import platform
        import torch
        
        try:
            memory_info = psutil.virtual_memory()
            cpu_info = psutil.cpu_freq()
            
            return {
                'platform': {
                    'system': platform.system(),
                    'release': platform.release(),
                    'version': platform.version(),
                    'machine': platform.machine(),
                    'processor': platform.processor()
                },
                'memory': {
                    'total_gb': memory_info.total / (1024**3),
                    'available_gb': memory_info.available / (1024**3),
                    'percent_used': memory_info.percent
                },
                'cpu': {
                    'current_freq_mhz': cpu_info.current if cpu_info else None,
                    'max_freq_mhz': cpu_info.max if cpu_info else None,
                    'min_freq_mhz': cpu_info.min if cpu_info else None
                },
                'pytorch': {
                    'version': torch.__version__,
                    'num_threads': torch.get_num_threads(),
                    'mkldnn_enabled': torch.backends.mkldnn.enabled
                }
            }
        except Exception as e:
            logger.warning(f"Failed to get system info: {e}")
            return {'error': str(e)}
    
    def _save_results(self, results: Dict[str, Any]):
        """Save benchmark results to file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"embedding_benchmark_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Benchmark results saved to: {filename}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def print_results_summary(self, results: Dict[str, Any]):
        """Print formatted summary of benchmark results"""
        print("\n" + "="*80)
        print("EMBEDDING PERFORMANCE BENCHMARK RESULTS")
        print("="*80)
        
        # System info
        system = results.get('system_info', {})
        cpu_summary = results.get('cpu_info', {}).get('cpu_summary', {})
        
        print(f"\nSystem Information:")
        print(f"  CPU: {cpu_summary.get('vendor', 'Unknown')} {cpu_summary.get('model', 'Unknown')}")
        print(f"  Architecture: {cpu_summary.get('architecture', 'Unknown')}")
        print(f"  Cores: {cpu_summary.get('physical_cores', 'Unknown')} physical, "
              f"{cpu_summary.get('logical_cores', 'Unknown')} logical")
        
        if 'memory' in system:
            memory = system['memory']
            print(f"  Memory: {memory.get('total_gb', 0):.1f}GB total, "
                  f"{memory.get('available_gb', 0):.1f}GB available")
        
        # Performance results
        print(f"\nPerformance Results:")
        print("-" * 80)
        
        for dataset_name, dataset_results in results.get('benchmarks', {}).items():
            print(f"\nDataset: {dataset_name.upper()}")
            print(f"{'Optimization':<20} {'Texts/sec':<12} {'Avg Time':<12} {'Memory':<10} {'Status'}")
            print("-" * 70)
            
            # Sort by performance (texts per second)
            sorted_opts = []
            for opt_name, opt_results in dataset_results.items():
                if opt_results.get('success', False):
                    tps = opt_results['performance']['texts_per_second']
                    sorted_opts.append((opt_name, tps, opt_results))
                else:
                    sorted_opts.append((opt_name, 0, opt_results))
            
            sorted_opts.sort(key=lambda x: x[1], reverse=True)
            
            for opt_name, tps, opt_results in sorted_opts:
                if opt_results.get('success', False):
                    perf = opt_results['performance']
                    print(f"{opt_name:<20} {perf['texts_per_second']:<12.1f} "
                          f"{perf['avg_time_seconds']:<12.3f} "
                          f"{perf['avg_memory_mb']:<10.1f} {'✓'}")
                else:
                    error = opt_results.get('error', 'Unknown error')
                    print(f"{opt_name:<20} {'N/A':<12} {'N/A':<12} {'N/A':<10} {'✗ ' + error[:30]}")
        
        # Recommendations
        print(f"\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        
        best_optimization = self._get_best_optimization(results)
        if best_optimization:
            print(f"Best Overall Performance: {best_optimization}")
        
        auto_selected = results.get('cpu_info', {}).get('optimization_settings', {}).get('recommended_optimization')
        if auto_selected:
            print(f"Auto-Detected Recommendation: {auto_selected}")
        
        print(f"\nTo use the best optimization automatically:")
        print(f"  from models.embeddings_smart import smart_bge_model")
        print(f"  embeddings = smart_bge_model.encode(texts)")
    
    def _get_best_optimization(self, results: Dict[str, Any]) -> Optional[str]:
        """Determine the best overall optimization from results"""
        optimization_scores = {}
        
        for dataset_name, dataset_results in results.get('benchmarks', {}).items():
            for opt_name, opt_results in dataset_results.items():
                if opt_results.get('success', False):
                    tps = opt_results['performance']['texts_per_second']
                    if opt_name not in optimization_scores:
                        optimization_scores[opt_name] = []
                    optimization_scores[opt_name].append(tps)
        
        # Calculate average performance across all datasets
        avg_scores = {}
        for opt_name, scores in optimization_scores.items():
            if scores:
                avg_scores[opt_name] = mean(scores)
        
        if avg_scores:
            return max(avg_scores.items(), key=lambda x: x[1])[0]
        
        return None


def run_quick_benchmark():
    """Run a quick benchmark for immediate feedback"""
    print("Running quick embedding benchmark...")
    
    benchmark = EmbeddingBenchmark()
    
    # Test with small dataset only
    texts = benchmark.test_datasets['small']
    results = {}
    
    optimizations = ['standard', 'optimized', 'intel_optimized', 'amd_optimized']
    
    for opt in optimizations:
        print(f"Testing {opt}...")
        try:
            factory = SmartEmbeddingFactory(force_optimization=opt)
            if factory._is_optimization_available(opt):
                model = factory.get_bge_model()
                
                # Quick test
                start_time = time.perf_counter()
                embeddings = model.encode(texts)
                end_time = time.perf_counter()
                
                elapsed_time = end_time - start_time
                tps = len(texts) / elapsed_time
                
                results[opt] = {
                    'texts_per_second': tps,
                    'time_seconds': elapsed_time,
                    'optimization_type': getattr(model, 'optimization_type', 'unknown')
                }
                print(f"  {opt}: {tps:.1f} texts/second ({elapsed_time:.3f}s)")
            else:
                print(f"  {opt}: Not available")
                results[opt] = {'error': 'Not available'}
                
        except Exception as e:
            print(f"  {opt}: Error - {e}")
            results[opt] = {'error': str(e)}
    
    # Show best
    best_opt = None
    best_tps = 0
    for opt, result in results.items():
        if 'texts_per_second' in result and result['texts_per_second'] > best_tps:
            best_tps = result['texts_per_second']
            best_opt = opt
    
    if best_opt:
        print(f"\nBest performance: {best_opt} ({best_tps:.1f} texts/second)")
    
    return results


def main():
    """Main benchmark execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Embedding Performance Benchmark")
    parser.add_argument('--quick', action='store_true', help='Run quick benchmark only')
    parser.add_argument('--iterations', type=int, default=3, help='Number of benchmark iterations')
    parser.add_argument('--report-only', action='store_true', help='Only show CPU detection report')
    
    args = parser.parse_args()
    
    if args.report_only:
        print_smart_report()
        return
    
    if args.quick:
        run_quick_benchmark()
        return
    
    # Full benchmark
    benchmark = EmbeddingBenchmark()
    results = benchmark.run_comprehensive_benchmark(iterations=args.iterations)
    benchmark.print_results_summary(results)
    
    print(f"\nBenchmark completed. Results saved to embedding_benchmark_*.json")


if __name__ == "__main__":
    main()
