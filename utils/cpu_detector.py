"""
CPU Detection and Optimization Selection Utility
Automatically detects CPU characteristics and recommends optimal embedding configuration
"""
import os
import platform
import re
import subprocess
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from utils.logger import Logger

logger = Logger.get_logger("hybrid_search.cpu_detector")


@dataclass
class CPUInfo:
    """CPU information and optimization recommendations"""
    vendor: str
    model: str
    architecture: str
    physical_cores: int
    logical_cores: int
    optimal_threads: int
    optimal_batch_size: int
    recommended_optimization: str
    supports_avx2: bool
    supports_avx512: bool
    cache_size_l3: Optional[int] = None
    base_frequency: Optional[float] = None


class CPUDetector:
    """Intelligent CPU detection and optimization recommendation"""
    
    def __init__(self):
        self.cpu_info = self._detect_cpu()
        logger.info(f"Detected CPU: {self.cpu_info.vendor} {self.cpu_info.model}")
    
    def _detect_cpu(self) -> CPUInfo:
        """Detect CPU characteristics and determine optimal configuration"""
        try:
            # Get basic CPU info
            logical_cores = os.cpu_count() or 1
            
            if platform.system() == "Linux":
                return self._detect_linux_cpu(logical_cores)
            elif platform.system() == "Windows":
                return self._detect_windows_cpu(logical_cores)
            elif platform.system() == "Darwin":  # macOS
                return self._detect_macos_cpu(logical_cores)
            else:
                return self._create_fallback_cpu_info(logical_cores)
                
        except Exception as e:
            logger.warning(f"CPU detection failed: {e}")
            return self._create_fallback_cpu_info(logical_cores)
    
    def _detect_linux_cpu(self, logical_cores: int) -> CPUInfo:
        """Detect CPU on Linux systems"""
        cpu_info = {}
        
        try:
            with open('/proc/cpuinfo', 'r') as f:
                content = f.read()
            
            # Extract CPU information
            cpu_info['vendor'] = self._extract_cpu_vendor(content)
            cpu_info['model'] = self._extract_cpu_model(content)
            cpu_info['physical_cores'] = self._count_physical_cores(content)
            cpu_info['flags'] = self._extract_cpu_flags(content)
            
            # Get cache information
            cpu_info['cache_l3'] = self._get_l3_cache_size()
            
            # Get frequency information
            cpu_info['base_freq'] = self._get_base_frequency()
            
        except Exception as e:
            logger.warning(f"Failed to read /proc/cpuinfo: {e}")
            cpu_info = self._get_basic_cpu_info()
        
        return self._build_cpu_info(cpu_info, logical_cores)
    
    def _detect_windows_cpu(self, logical_cores: int) -> CPUInfo:
        """Detect CPU on Windows systems"""
        try:
            # Use wmic to get CPU information
            result = subprocess.run(
                ['wmic', 'cpu', 'get', 'Name,Manufacturer,NumberOfCores,NumberOfLogicalProcessors'],
                capture_output=True, text=True, timeout=10
            )
            
            cpu_info = self._parse_windows_cpu_info(result.stdout)
            cpu_info['physical_cores'] = cpu_info.get('cores', logical_cores // 2)
            
        except Exception as e:
            logger.warning(f"Windows CPU detection failed: {e}")
            cpu_info = self._get_basic_cpu_info()
        
        return self._build_cpu_info(cpu_info, logical_cores)
    
    def _detect_macos_cpu(self, logical_cores: int) -> CPUInfo:
        """Detect CPU on macOS systems"""
        try:
            # Use sysctl to get CPU information
            brand_result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True, text=True, timeout=5
            )
            
            cores_result = subprocess.run(
                ['sysctl', '-n', 'hw.physicalcpu'],
                capture_output=True, text=True, timeout=5
            )
            
            cpu_info = {
                'model': brand_result.stdout.strip(),
                'vendor': 'Apple' if 'Apple' in brand_result.stdout else 'Intel',
                'physical_cores': int(cores_result.stdout.strip())
            }
            
        except Exception as e:
            logger.warning(f"macOS CPU detection failed: {e}")
            cpu_info = self._get_basic_cpu_info()
        
        return self._build_cpu_info(cpu_info, logical_cores)
    
    def _extract_cpu_vendor(self, cpuinfo_content: str) -> str:
        """Extract CPU vendor from /proc/cpuinfo"""
        patterns = [
            r'vendor_id\s*:\s*(.+)',
            r'vendor\s*:\s*(.+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, cpuinfo_content, re.IGNORECASE)
            if match:
                vendor = match.group(1).strip()
                if 'AMD' in vendor.upper():
                    return 'AMD'
                elif 'INTEL' in vendor.upper() or 'GenuineIntel' in vendor:
                    return 'Intel'
                else:
                    return vendor
        
        return 'Unknown'
    
    def _extract_cpu_model(self, cpuinfo_content: str) -> str:
        """Extract CPU model from /proc/cpuinfo"""
        patterns = [
            r'model name\s*:\s*(.+)',
            r'cpu model\s*:\s*(.+)',
            r'processor\s*:\s*(.+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, cpuinfo_content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return 'Unknown Model'
    
    def _count_physical_cores(self, cpuinfo_content: str) -> int:
        """Count physical CPU cores from /proc/cpuinfo"""
        try:
            # Count unique physical IDs and core IDs
            physical_ids = set()
            core_ids = set()
            
            for line in cpuinfo_content.split('\n'):
                if line.startswith('physical id'):
                    physical_ids.add(line.split(':')[1].strip())
                elif line.startswith('core id'):
                    core_ids.add(line.split(':')[1].strip())
            
            if physical_ids and core_ids:
                return len(physical_ids) * len(core_ids)
            
            # Fallback: count cpu cores field
            cores_match = re.search(r'cpu cores\s*:\s*(\d+)', cpuinfo_content)
            if cores_match:
                return int(cores_match.group(1))
            
        except Exception:
            pass
        
        # Final fallback
        return os.cpu_count() // 2 if os.cpu_count() else 1
    
    def _extract_cpu_flags(self, cpuinfo_content: str) -> list:
        """Extract CPU flags/features from /proc/cpuinfo"""
        flags_match = re.search(r'flags\s*:\s*(.+)', cpuinfo_content)
        if flags_match:
            return flags_match.group(1).strip().split()
        return []
    
    def _get_l3_cache_size(self) -> Optional[int]:
        """Get L3 cache size in KB"""
        try:
            cache_files = [
                '/sys/devices/system/cpu/cpu0/cache/index3/size',
                '/proc/cpuinfo'  # Fallback
            ]
            
            for cache_file in cache_files:
                if os.path.exists(cache_file):
                    with open(cache_file, 'r') as f:
                        content = f.read().strip()
                        if 'K' in content:
                            return int(content.replace('K', ''))
                        elif 'cache size' in content:
                            match = re.search(r'cache size\s*:\s*(\d+)\s*KB', content)
                            if match:
                                return int(match.group(1))
        except Exception:
            pass
        
        return None
    
    def _get_base_frequency(self) -> Optional[float]:
        """Get base CPU frequency in GHz"""
        try:
            freq_files = [
                '/proc/cpuinfo',
                '/sys/devices/system/cpu/cpu0/cpufreq/base_frequency'
            ]
            
            for freq_file in freq_files:
                if os.path.exists(freq_file):
                    with open(freq_file, 'r') as f:
                        content = f.read()
                        
                        # Look for MHz in cpuinfo
                        freq_match = re.search(r'cpu MHz\s*:\s*(\d+\.?\d*)', content)
                        if freq_match:
                            return float(freq_match.group(1)) / 1000  # Convert MHz to GHz
        except Exception:
            pass
        
        return None
    
    def _get_basic_cpu_info(self) -> Dict[str, Any]:
        """Get basic CPU info as fallback"""
        return {
            'vendor': 'Unknown',
            'model': platform.processor() or 'Unknown Model',
            'physical_cores': os.cpu_count() // 2 if os.cpu_count() else 1,
            'flags': []
        }
    
    def _parse_windows_cpu_info(self, wmic_output: str) -> Dict[str, Any]:
        """Parse Windows wmic CPU output"""
        cpu_info = {}
        lines = wmic_output.strip().split('\n')
        
        for line in lines[1:]:  # Skip header
            if line.strip():
                parts = line.split()
                if len(parts) >= 4:
                    cpu_info['vendor'] = parts[0] if 'AMD' in parts[0] or 'Intel' in parts[0] else 'Unknown'
                    cpu_info['model'] = ' '.join(parts[:-2])
                    cpu_info['cores'] = int(parts[-2])
        
        return cpu_info
    
    def _build_cpu_info(self, cpu_data: Dict[str, Any], logical_cores: int) -> CPUInfo:
        """Build CPUInfo object from detected data"""
        vendor = cpu_data.get('vendor', 'Unknown')
        model = cpu_data.get('model', 'Unknown Model')
        physical_cores = cpu_data.get('physical_cores', logical_cores // 2)
        flags = cpu_data.get('flags', [])
        
        # Detect architecture
        architecture = self._detect_architecture(vendor, model)
        
        # Check for instruction set support
        supports_avx2 = 'avx2' in flags
        supports_avx512 = any('avx512' in flag for flag in flags)
        
        # Calculate optimal settings
        optimal_threads, optimal_batch_size, recommended_optimization = self._calculate_optimal_settings(
            vendor, model, physical_cores, logical_cores, architecture
        )
        
        return CPUInfo(
            vendor=vendor,
            model=model,
            architecture=architecture,
            physical_cores=physical_cores,
            logical_cores=logical_cores,
            optimal_threads=optimal_threads,
            optimal_batch_size=optimal_batch_size,
            recommended_optimization=recommended_optimization,
            supports_avx2=supports_avx2,
            supports_avx512=supports_avx512,
            cache_size_l3=cpu_data.get('cache_l3'),
            base_frequency=cpu_data.get('base_freq')
        )
    
    def _detect_architecture(self, vendor: str, model: str) -> str:
        """Detect CPU architecture"""
        model_lower = model.lower()
        
        if vendor == 'AMD':
            if any(gen in model_lower for gen in ['5700g', '5600g', '5800x', '5900x', '5950x']):
                return 'Zen 3'
            elif any(gen in model_lower for gen in ['3700x', '3800x', '3900x', '3950x']):
                return 'Zen 2'
            elif any(gen in model_lower for gen in ['2700x', '2800x']):
                return 'Zen+'
            elif 'ryzen' in model_lower:
                return 'Zen'
            else:
                return 'AMD Unknown'
        
        elif vendor == 'Intel':
            if any(gen in model_lower for gen in ['i3-12', 'i5-12', 'i7-12', 'i9-12']):
                return 'Alder Lake'
            elif any(gen in model_lower for gen in ['i3-11', 'i5-11', 'i7-11', 'i9-11']):
                return 'Tiger Lake'
            elif any(gen in model_lower for gen in ['i3-10', 'i5-10', 'i7-10', 'i9-10']):
                return 'Comet Lake'
            else:
                return 'Intel Unknown'
        
        elif vendor == 'Apple':
            if 'm1' in model_lower or 'm2' in model_lower:
                return 'Apple Silicon'
        
        return 'Unknown'
    
    def _calculate_optimal_settings(self, vendor: str, model: str, physical_cores: int, 
                                  logical_cores: int, architecture: str) -> Tuple[int, int, str]:
        """Calculate optimal threading, batch size, and recommended optimization"""
        
        # AMD-specific optimizations
        if vendor == 'AMD':
            if architecture == 'Zen 3' and physical_cores >= 8:  # AMD 5700G, 5800X, etc.
                optimal_threads = min(14, logical_cores - 2)  # Leave 2-4 threads for system
                optimal_batch_size = 32
                recommended_optimization = 'amd_optimized'
            elif physical_cores >= 6:
                optimal_threads = min(logical_cores - 2, 12)
                optimal_batch_size = 24
                recommended_optimization = 'amd_optimized'
            elif physical_cores >= 4:
                optimal_threads = min(logical_cores - 1, 8)
                optimal_batch_size = 16
                recommended_optimization = 'amd_optimized'
            else:
                optimal_threads = logical_cores
                optimal_batch_size = 8
                recommended_optimization = 'optimized'
        
        # Intel-specific optimizations
        elif vendor == 'Intel':
            if physical_cores >= 8:
                optimal_threads = min(logical_cores - 2, 16)
                optimal_batch_size = 48
                recommended_optimization = 'intel_optimized'  # Has Intel-specific libs
            elif physical_cores >= 6:
                optimal_threads = min(logical_cores - 1, 12)
                optimal_batch_size = 32
                recommended_optimization = 'intel_optimized'
            elif physical_cores >= 4:
                optimal_threads = min(logical_cores, 8)
                optimal_batch_size = 24
                recommended_optimization = 'optimized'
            else:
                optimal_threads = logical_cores
                optimal_batch_size = 16
                recommended_optimization = 'optimized'
        
        # Apple Silicon
        elif vendor == 'Apple':
            # Apple Silicon has excellent single-thread performance
            optimal_threads = min(logical_cores, 8)
            optimal_batch_size = 24
            recommended_optimization = 'optimized'
        
        # Unknown/Generic
        else:
            optimal_threads = min(logical_cores - 1, 8)
            optimal_batch_size = 16
            recommended_optimization = 'optimized'
        
        return optimal_threads, optimal_batch_size, recommended_optimization
    
    def _create_fallback_cpu_info(self, logical_cores: int) -> CPUInfo:
        """Create fallback CPU info when detection fails"""
        return CPUInfo(
            vendor='Unknown',
            model='Unknown Model',
            architecture='Unknown',
            physical_cores=logical_cores // 2,
            logical_cores=logical_cores,
            optimal_threads=min(logical_cores - 1, 8),
            optimal_batch_size=16,
            recommended_optimization='optimized',
            supports_avx2=False,
            supports_avx512=False
        )
    
    def get_cpu_info(self) -> CPUInfo:
        """Get detected CPU information"""
        return self.cpu_info
    
    def get_optimization_recommendation(self) -> str:
        """Get recommended optimization type"""
        return self.cpu_info.recommended_optimization
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return {
            'cpu_summary': {
                'vendor': self.cpu_info.vendor,
                'model': self.cpu_info.model,
                'architecture': self.cpu_info.architecture,
                'physical_cores': self.cpu_info.physical_cores,
                'logical_cores': self.cpu_info.logical_cores,
            },
            'optimization_settings': {
                'recommended_optimization': self.cpu_info.recommended_optimization,
                'optimal_threads': self.cpu_info.optimal_threads,
                'optimal_batch_size': self.cpu_info.optimal_batch_size,
            },
            'capabilities': {
                'supports_avx2': self.cpu_info.supports_avx2,
                'supports_avx512': self.cpu_info.supports_avx512,
                'cache_size_l3_kb': self.cpu_info.cache_size_l3,
                'base_frequency_ghz': self.cpu_info.base_frequency,
            }
        }
    
    def print_cpu_report(self):
        """Print detailed CPU detection report"""
        info = self.cpu_info
        print("=== CPU Detection Report ===")
        print(f"Vendor: {info.vendor}")
        print(f"Model: {info.model}")
        print(f"Architecture: {info.architecture}")
        print(f"Physical Cores: {info.physical_cores}")
        print(f"Logical Cores: {info.logical_cores}")
        print(f"L3 Cache: {info.cache_size_l3}KB" if info.cache_size_l3 else "L3 Cache: Unknown")
        print(f"Base Frequency: {info.base_frequency}GHz" if info.base_frequency else "Base Frequency: Unknown")
        print(f"AVX2 Support: {info.supports_avx2}")
        print(f"AVX512 Support: {info.supports_avx512}")
        print("\n=== Optimization Recommendations ===")
        print(f"Recommended Optimization: {info.recommended_optimization}")
        print(f"Optimal Threads: {info.optimal_threads}")
        print(f"Optimal Batch Size: {info.optimal_batch_size}")


# Global CPU detector instance
cpu_detector = CPUDetector()


if __name__ == "__main__":
    detector = CPUDetector()
    detector.print_cpu_report()
