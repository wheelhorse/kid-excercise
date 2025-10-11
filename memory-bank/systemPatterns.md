# System Patterns: Resume Retrieval System Architecture

## Architectural Overview

The system follows a **layered, modular architecture** with clear separation of concerns and intelligent optimization patterns.

```
┌─────────────────────────────────────────────────────────────┐
│                    Terminal Interface                        │
├─────────────────────────────────────────────────────────────┤
│                   Search Service Layer                       │
├─────────────────────────────────────────────────────────────┤
│     Smart Embedding Factory    │      Sync Manager          │
├─────────────────────────────────────────────────────────────┤
│   Qdrant Client    │   MariaDB Client   │   Text Processing │
├─────────────────────────────────────────────────────────────┤
│              Utilities (Logger, CPU Detection)              │
└─────────────────────────────────────────────────────────────┘
```

## Core Design Patterns

### 1. Smart Factory Pattern (Key Innovation)

**Problem**: Different CPU architectures (Intel, AMD) require different optimization strategies for embedding models.

**Solution**: `SmartEmbeddingFactory` automatically detects hardware and selects optimal implementation.

```python
# Pattern Implementation
class SmartEmbeddingFactory:
    def __init__(self, force_optimization: Optional[str] = None):
        self.cpu_info = cpu_detector.get_cpu_info()
        self.selected_optimization = self._select_optimization()
    
    def _select_optimization(self) -> str:
        # Auto-detection with fallback hierarchy
        if intel_cpu_detected:
            return 'intel_optimized'
        elif amd_cpu_detected:
            return 'optimized'
        else:
            return 'standard'
```

**Benefits**:
- Automatic performance optimization
- Graceful fallback for unsupported hardware
- Transparent to consuming code
- Easy testing with forced optimizations

### 2. Service Facade Pattern

**Problem**: Complex interactions between multiple subsystems (database, search, sync).

**Solution**: `SearchService` provides unified interface hiding internal complexity.

```python
# Unified Interface
class SearchService:
    def search_candidates(self, job_description: str, **kwargs) -> Dict[str, Any]:
        # Orchestrates: embedding → search → enrichment → ranking
        pass
    
    def initialize(self, force_sync: bool = False) -> bool:
        # Coordinates: sync_manager → qdrant_setup → validation
        pass
```

**Benefits**:
- Single point of interaction for complex operations
- Internal complexity hidden from clients
- Easier testing and mocking
- Clear error handling boundaries

### 3. Strategy Pattern for Embedding Optimization

**Problem**: Multiple embedding implementations with different performance characteristics.

**Solution**: Interchangeable embedding strategies selected by Smart Factory.

```python
# Strategy Interface (implicit)
class BGEEmbedding:           # Standard implementation
class OptimizedBGEEmbedding:  # General optimizations
class IntelOptimizedBGEEmbedding:  # Intel-specific optimizations

# Selection Logic
factory.get_bge_model()  # Returns appropriate strategy
```

**Benefits**:
- Runtime optimization selection
- Easy addition of new optimizations
- Consistent interface across implementations
- Performance testing and comparison

### 4. Manager Pattern for Complex Operations

**Problem**: Data synchronization involves multiple steps, error handling, and state management.

**Solution**: `SyncManager` encapsulates all sync-related complexity.

```python
# Manager Responsibilities
class SyncManager:
    - Connection management (MariaDB + Qdrant)
    - Incremental vs full sync logic
    - Error recovery and retry mechanisms
    - Progress tracking and logging
    - State persistence
```

**Benefits**:
- Centralized sync logic
- Consistent error handling
- State management abstraction
- Testable sync operations

### 5. Hybrid Search Composition Pattern

**Problem**: Combine dense (semantic) and sparse (keyword) search results optimally.

**Solution**: RRF (Reciprocal Rank Fusion) composition with configurable weights.

```python
# Composition Logic
def hybrid_search(query, dense_weight=0.7, sparse_weight=0.3):
    dense_results = dense_search(query)
    sparse_results = sparse_search(query)
    
    # RRF fusion
    combined_scores = reciprocal_rank_fusion(
        dense_results, sparse_results, 
        dense_weight, sparse_weight
    )
    
    return ranked_results(combined_scores)
```

**Benefits**:
- Best of both search approaches
- Configurable balance between semantic and keyword matching
- Explainable ranking (separate scores visible)
- Consistent top-k results

## Data Flow Patterns

### 1. Pipeline Pattern for Text Processing

**Text Processing Pipeline**:
```
Raw Text → Cleaning → Tokenization → Embedding → Vector Storage
         ↓
    Language Detection → Appropriate Tokenizer (jieba/standard)
```

**Implementation**:
- Each stage is independent and testable
- Language-specific processing automatically selected
- Error handling at each stage
- Performance monitoring throughout pipeline

### 2. Event-Driven Sync Pattern

**Incremental Sync Flow**:
```
MariaDB Change Detection → Timestamp Comparison → Batch Processing → Qdrant Update
                                                        ↓
                                              Progress Tracking → Status Update
```

**Key Characteristics**:
- Timestamp-based change detection
- Batch processing for efficiency
- Atomic operations where possible
- Comprehensive logging for audit trail

### 3. Lazy Loading Pattern for Models

**Model Loading Strategy**:
```python
class SmartEmbeddingFactory:
    def __init__(self):
        self._bge_model = None  # Not loaded until needed
        self._hybrid_model = None
    
    def get_bge_model(self):
        if self._bge_model is None:
            self._bge_model = self._load_bge_model()
        return self._bge_model
```

**Benefits**:
- Faster startup time
- Memory efficiency
- Load only what's needed
- Better error isolation

## Error Handling Patterns

### 1. Circuit Breaker Pattern

**Problem**: External service failures (MariaDB, Qdrant) can cascade.

**Solution**: Fail fast with graceful degradation.

```python
# Implementation in service clients
class MariaDBClient:
    def __init__(self):
        self.connection_failures = 0
        self.circuit_open = False
    
    def connect(self):
        if self.circuit_open:
            return False  # Fail fast
        # ... connection logic with failure tracking
```

### 2. Retry Pattern with Exponential Backoff

**Application**: Database connections, sync operations, search requests.

```python
# Configurable retry logic
SYNC_CONFIG = {
    "max_retry_attempts": 3,
    "retry_delay_seconds": 5,  # Base delay
    "exponential_backoff": True
}
```

### 3. Graceful Degradation Pattern

**Search Fallbacks**:
1. Hybrid search (preferred)
2. Dense-only search (if sparse fails)
3. Cached results (if all search fails)
4. Error message with guidance

## Performance Optimization Patterns

### 1. CPU-Specific Optimization Selection

**Hardware Detection → Optimization Mapping**:
```python
CPU_OPTIMIZATION_MAP = {
    'Intel': 'intel_optimized',  # Uses Intel MKL optimizations
    'AMD': 'optimized',          # General optimizations
    'Unknown': 'standard'        # Safe fallback
}
```

### 2. Batch Processing Pattern

**Large Dataset Handling**:
- Configurable batch sizes based on available memory
- Progress tracking for long-running operations
- Memory cleanup between batches
- Parallel processing where safe

### 3. Caching Pattern

**Multi-Level Caching**:
```python
# Model Level: Cached embeddings for repeated queries
# Service Level: Cached search results for identical queries
# Application Level: Connection pooling and model reuse
```

## Configuration Patterns

### 1. Environment-Based Configuration

**Hierarchical Configuration**:
```
Default Values → Environment Variables → Runtime Overrides
```

**Example**:
```python
BGE_M3_MODEL = os.getenv("BGE_M3_MODEL", "BAAI/bge-small-zh-v1.5")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
```

### 2. Validation Pattern

**Configuration Validation**:
```python
def validate_config() -> bool:
    # Check required variables
    # Validate ranges and formats
    # Test connections with provided credentials
    # Return boolean + detailed error messages
```

## Testing Patterns

### 1. Factory Testing Pattern

**Smart Factory Validation**:
```python
# Test automatic detection
# Test forced optimization selection
# Test fallback behavior
# Test performance characteristics
```

### 2. Mock Service Pattern

**External Dependency Mocking**:
```python
# MockMariaDBClient for database testing
# MockQdrantClient for vector operations
# Deterministic test data for consistent results
```

### 3. Performance Benchmarking Pattern

**Automated Performance Testing**:
```python
def benchmark_all_optimizations():
    # Test each optimization with same data
    # Compare performance metrics
    # Generate comparison reports
    # Validate optimization selection logic
```

## Security Patterns

### 1. Credential Management Pattern

**Secure Configuration**:
- Environment variables for sensitive data
- No hardcoded credentials
- Connection string validation
- Credential rotation support

### 2. Input Validation Pattern

**Search Query Sanitization**:
- Text length limits
- Character encoding validation
- SQL injection prevention (for MariaDB queries)
- Safe embedding text processing

## Monitoring and Observability Patterns

### 1. Structured Logging Pattern

**Consistent Log Format**:
```python
logger.info("Search completed", extra={
    'query_length': len(query),
    'results_count': len(results),
    'search_time_ms': search_time * 1000,
    'search_mode': 'hybrid'
})
```

### 2. Performance Metrics Pattern

**Key Performance Indicators**:
- Search latency (p50, p95, p99)
- Sync throughput (resumes/minute)
- Error rates by component
- Resource utilization (CPU, memory)

These patterns ensure the system is maintainable, performant, and robust while providing clear extension points for future enhancements.
