# Technology Context: Resume Retrieval System

## Technology Stack Overview

The system leverages a modern Python-based stack optimized for CPU-efficient vector search and hybrid retrieval.

### Core Technologies

| Component | Technology | Version/Model | Purpose |
|-----------|------------|---------------|---------|
| **Vector Database** | Qdrant | Latest | Dense & sparse vector storage/search |
| **Embedding Model** | BAAI/bge-small-zh-v1.5 | Latest | Chinese/English semantic embeddings |
| **Sparse Search** | jieba + BM25 | Latest | Chinese tokenization + keyword search |
| **Primary Database** | MariaDB | 10.x+ | Source resume data storage |
| **Runtime** | Python | 3.8+ | Application runtime environment |
| **Optimization** | Intel MKL/OpenMP | Auto-detected | CPU-specific performance optimization |

### Key Dependencies

```python
# Core ML/Search
sentence-transformers>=2.2.0    # BGE model loading
qdrant-client>=1.6.0           # Vector database client
jieba>=0.42.1                  # Chinese text segmentation
rank-bm25>=0.2.2               # BM25 implementation

# Database
PyMySQL>=1.0.2                 # MariaDB connector
python-dotenv>=1.0.0           # Environment configuration

# Performance
numpy>=1.21.0                  # Efficient array operations
torch>=2.0.0                   # Neural network inference
optimum[intel]                 # Intel CPU optimizations (optional)

# Utilities
psutil>=5.9.0                  # System/CPU detection
```

## Embedding Technology Deep Dive

### BGE-M3 Model Characteristics

**Model**: `BAAI/bge-small-zh-v1.5`
- **Dimensions**: 512 (optimized for balance of quality vs performance)
- **Languages**: Chinese, English, multilingual support
- **Context Length**: 512 tokens
- **Inference**: CPU-optimized, sub-100ms encoding
- **Memory**: ~400MB model size

**Why BGE over alternatives**:
- Superior Chinese-English bilingual performance
- Optimized for CPU inference
- Smaller model size vs BGE-large
- Active maintenance and updates
- Strong performance on similarity tasks

### CPU Optimization Stack

#### Intel Optimization Path
```python
# Intel-specific optimizations
import intel_extension_for_pytorch as ipex  # Optional
from optimum.intel import IntelTextEncoder   # Optional

# MKL-DNN optimizations
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4
```

#### AMD/General Optimization Path
```python
# OpenMP optimizations
export OMP_NUM_THREADS=8
export OMP_SCHEDULE=static

# PyTorch optimizations
torch.set_num_threads(8)
torch.set_num_interop_threads(4)
```

#### Fallback (Standard) Path
```python
# Basic optimizations
torch.set_num_threads(cpu_count // 2)
# No special CPU instructions
# Compatible with all hardware
```

## Database Architecture

### MariaDB Schema (Read-Only Access)

**Target Tables**:
```sql
-- Candidate information
CREATE TABLE candidate (
    candidate_id INT PRIMARY KEY,
    first_name VARCHAR(255),
    last_name VARCHAR(255),
    email1 VARCHAR(255),
    key_skills TEXT,
    notes TEXT,
    date_modified DATETIME
);

-- Resume attachments with extracted text
CREATE TABLE attachment (
    attachment_id INT PRIMARY KEY,
    data_item_id INT,           -- FK to candidate_id
    data_item_type INT,         -- 100 for candidates
    text LONGTEXT,              -- Extracted resume text
    resume TINYINT,             -- 1 for resume files
    date_modified DATETIME
);
```

**Query Pattern**:
```sql
-- Sync query for resume data
SELECT 
    c.candidate_id, c.first_name, c.last_name, c.email1,
    c.key_skills, c.notes, a.text as resume_text,
    GREATEST(c.date_modified, a.date_modified) as last_modified
FROM candidate c 
LEFT JOIN attachment a ON c.candidate_id = a.data_item_id 
WHERE a.resume = 1 AND a.data_item_type = 100
  AND GREATEST(c.date_modified, a.date_modified) > ?
ORDER BY last_modified ASC;
```

### Qdrant Configuration

**Collection Setup**:
```python
vectors_config = {
    "bge-dense": VectorParams(
        size=512,                    # BGE-small embedding size
        distance=Distance.COSINE,    # Cosine similarity
        on_disk=True                 # Optional: disk storage for large datasets
    )
}

# Optional: Sparse vectors for hybrid search
sparse_vectors_config = {
    "bm25-sparse": SparseVectorParams(
        index=SparseIndexParams(
            on_disk=False            # Keep sparse index in memory
        )
    )
}
```

**Payload Schema**:
```python
payload = {
    "candidate_id": int,           # MariaDB reference
    "first_name": str,
    "last_name": str,
    "email1": str,
    "key_skills": str,
    "notes": str,
    "search_text": str,            # Combined searchable text
    "date_modified": str,          # ISO format timestamp
    "source": "mariadb"            # Data provenance
}
```

## Text Processing Pipeline

### Chinese Text Handling

**Jieba Configuration**:
```python
import jieba
import jieba.analyse

# Initialize with custom dictionary if needed
jieba.load_userdict("custom_tech_terms.txt")

# Segmentation for BM25
def segment_chinese(text: str) -> List[str]:
    # Full segmentation for maximum recall
    return jieba.lcut(text, cut_all=False)

# Keyword extraction for enhancement
def extract_keywords(text: str, topK: int = 20) -> List[str]:
    return jieba.analyse.extract_tags(text, topK=topK)
```

### English Text Processing

**Standard NLP Pipeline**:
```python
import re
from typing import List

def process_english(text: str) -> str:
    # Basic cleaning
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)       # Normalize whitespace
    text = text.lower().strip()
    return text

def tokenize_english(text: str) -> List[str]:
    # Simple whitespace tokenization
    # BM25 will handle stemming internally
    return text.split()
```

### Hybrid Processing Strategy

```python
def process_mixed_text(text: str) -> Dict[str, Any]:
    # Language detection
    chinese_ratio = len(re.findall(r'[\u4e00-\u9fff]', text)) / len(text)
    
    if chinese_ratio > 0.3:
        # Chinese-dominant processing
        tokens = jieba.lcut(text)
        processed_text = ' '.join(tokens)
    else:
        # English-dominant processing
        processed_text = process_english(text)
        tokens = tokenize_english(processed_text)
    
    return {
        'processed_text': processed_text,
        'tokens': tokens,
        'language_hint': 'chinese' if chinese_ratio > 0.3 else 'english'
    }
```

## Performance Optimization Technologies

### CPU Detection and Optimization

**Hardware Detection**:
```python
import psutil
import platform
import subprocess

class CPUDetector:
    def get_cpu_info(self) -> CPUInfo:
        # Detect vendor (Intel, AMD, ARM, etc.)
        # Count physical/logical cores
        # Detect instruction sets (AVX, AVX2, etc.)
        # Recommend optimization strategy
        
    def get_optimal_thread_count(self) -> int:
        # Physical cores for CPU-bound tasks
        # Account for hyperthreading efficiency
        # Consider system load
```

**Optimization Selection Logic**:
```python
def select_optimization_strategy(cpu_info: CPUInfo) -> str:
    if cpu_info.vendor == 'Intel' and cpu_info.has_avx2:
        return 'intel_optimized'
    elif cpu_info.vendor == 'AMD' and cpu_info.physical_cores >= 4:
        return 'optimized'
    else:
        return 'standard'
```

### Memory Management

**Batch Processing Configuration**:
```python
# Dynamic batch sizing based on available memory
def calculate_batch_size() -> int:
    available_memory = psutil.virtual_memory().available
    model_memory = 400 * 1024 * 1024  # ~400MB for BGE-small
    
    # Conservative: 50% of available memory for processing
    processing_memory = available_memory * 0.5
    
    # Estimate per-text memory usage
    avg_text_size = 2048  # Average resume text length
    text_memory = avg_text_size * 4  # Rough estimate
    
    batch_size = int(processing_memory / text_memory)
    return min(max(batch_size, 8), 128)  # Clamp between 8-128
```

## Development Environment

### Local Development Setup

**Prerequisites**:
```bash
# Python environment
python3.8+ with pip
virtualenv or conda

# System dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install python3-dev build-essential

# Optional: Intel optimizations
pip install intel-extension-for-pytorch
```

**Environment Configuration**:
```bash
# Required environment variables
export MARIADB_HOST="localhost"
export MARIADB_USER="cats_user"
export MARIADB_PASSWORD="secure_password"
export MARIADB_DATABASE="cats_dev"

export QDRANT_URL="http://localhost:6333"
export QDRANT_API_KEY=""  # Optional for local

# Optional optimizations
export EMBEDDING_OPTIMIZATION="auto"  # or "intel_optimized", "optimized", "standard"
export BATCH_SIZE="32"
export LOG_LEVEL="INFO"
```

### Testing Technologies

**Testing Stack**:
```python
# Unit testing
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0

# Mocking and fixtures
pytest-mock>=3.10.0
responses>=0.23.0

# Performance testing
pytest-benchmark>=4.0.0
memory-profiler>=0.60.0
```

**Benchmarking Framework**:
```python
# Custom benchmarking for embedding optimizations
def benchmark_embedding_performance():
    test_texts = generate_test_corpus()
    
    for optimization in ['standard', 'optimized', 'intel_optimized']:
        factory = SmartEmbeddingFactory(force_optimization=optimization)
        model = factory.get_bge_model()
        
        # Measure throughput and latency
        results = time_embedding_operations(model, test_texts)
        
        # Memory profiling
        memory_usage = profile_memory_usage(model, test_texts)
        
        save_benchmark_results(optimization, results, memory_usage)
```

## Deployment Technologies

### Configuration Management

**Environment-Based Config**:
```python
# Hierarchical configuration loading
def load_config():
    # 1. Default values in config.py
    # 2. Environment variables override defaults
    # 3. Runtime parameters override environment
    # 4. Validation of final configuration
```

### Logging and Monitoring

**Structured Logging**:
```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    def info(self, message: str, **kwargs):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': 'INFO',
            'message': message,
            'component': self.component_name,
            **kwargs
        }
        print(json.dumps(log_entry))
```

**Performance Metrics**:
```python
# Key metrics to track
metrics = {
    'search_latency_ms': timer.elapsed() * 1000,
    'search_results_count': len(results),
    'embedding_time_ms': embedding_time * 1000,
    'qdrant_query_time_ms': query_time * 1000,
    'optimization_used': factory.selected_optimization,
    'cpu_usage_percent': psutil.cpu_percent(),
    'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024
}
```

### Error Handling Technologies

**Exception Hierarchy**:
```python
class HybridSearchError(Exception):
    """Base exception for hybrid search operations"""

class EmbeddingError(HybridSearchError):
    """Embedding generation failures"""

class DatabaseConnectionError(HybridSearchError):
    """Database connectivity issues"""

class SearchServiceError(HybridSearchError):
    """Search service operational errors"""
```

This technology stack provides a robust foundation for high-performance, CPU-optimized hybrid search while maintaining compatibility across different hardware configurations and deployment environments.
