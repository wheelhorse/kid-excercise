# Resume Retrieval System with Hybrid Search

A Python-based resume retrieval system using **Qdrant + BAAI/bge-m3 + jieba/BM25** technologies to provide intelligent candidate search capabilities.

## 📋 Project Overview

This system builds a sophisticated resume search engine that:
- Syncs resume data from a remote MariaDB database
- Implements hybrid search using dense and sparse embeddings
- Supports both Chinese and English text processing
- Provides terminal-based interface for testing and operations
- Returns top 100 candidates based on job descriptions

## 🎯 Requirements Summary

### Core Functionality
1. **Database Integration**: Connect to remote MariaDB database containing resume information
2. **Hybrid Search Engine**: Initialize Qdrant with BAAI/bge-m3 + jieba/BM25 hybrid search
3. **Data Synchronization**: Insert and update Qdrant database when MariaDB changes
4. **Intelligent Search**: Allow hybrid search based on job descriptions and additional criteria
5. **ID Mapping**: Maintain index mapping to redirect back to MariaDB records
6. **Terminal Interface**: Command-line interface for testing and operations
7. **Language Support**: Full support for both Chinese and English text processing

### Search Modes
- **Default**: BGE-M3 dense embeddings only
- **Hybrid**: BGE-M3 dense + BM25 sparse combination
- **Configurable**: Easy switching between modes for performance comparison

## 🏗️ Technical Architecture

### Database Schema Integration

**MariaDB Tables:**
- `candidate`: Main candidate information
  - Fields: `candidate_id`, `first_name`, `last_name`, `email1`, `key_skills`, `notes`, `date_modified`
- `attachment`: Resume files and extracted text
  - Fields: `attachment_id`, `data_item_id`, `text`, `resume`, `date_modified`

**Qdrant Payload Schema:**
```python
payload = {
    "candidate_id": int,           # Primary key from MariaDB
    "name": str,                   # first_name + last_name
    "email": str,                  # email1 from candidate table
    "key_skills": str,             # key_skills field
    "notes": str,                  # notes field from candidate
    "resume_text": str,            # extracted text from attachment
    "last_modified": datetime      # latest modification timestamp
}
```

**Search Text Composition:**
```
searchable_text = first_name + last_name + key_skills + notes + attachment.text
```

### Vector Configuration
```python
vectors_config = {
    "bge-m3-dense": VectorParams(
        size=1024, 
        distance=Distance.COSINE
    ),
    # Optional for hybrid mode
    "bm25": sparse_vectors_config
}
```

## 📁 Project Structure

```
hybrid_search/
├── README.md                 # This documentation
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── main.py                   # Entry point and CLI interface
├── models/
│   ├── __init__.py
│   ├── embeddings.py        # BGE-M3 embedding models
│   └── tokenizers.py        # Jieba + BM25 tokenization
├── database/
│   ├── __init__.py
│   ├── mariadb_client.py    # MariaDB connection & queries (dummy wrapper)
│   └── qdrant_client.py     # Qdrant operations
├── core/
│   ├── __init__.py
│   ├── indexer.py           # Data synchronization & indexing
│   ├── searcher.py          # Hybrid search implementation
│   └── sync_manager.py      # Database sync interface
├── utils/
│   ├── __init__.py
│   ├── text_processor.py    # Chinese/English text processing
│   └── logger.py            # Logging utilities
└── tests/
    ├── __init__.py
    └── test_search.py       # Terminal testing interface
```

## ⚙️ Configuration

### Database Connections
```python
# config.py
MARIADB_CONFIG = {
    "host": "your-mariadb-host",
    "user": "username", 
    "password": "password",
    "database": "cats_dev"
}

QDRANT_CONFIG = {
    "url": "your-qdrant-cloud-url",
    "api_key": "your-api-key"
}
```

### Model Settings
```python
BGE_M3_MODEL = "BAAI/bge-m3"
COLLECTION_NAME = "resume_hybrid_search"
SEARCH_MODES = ["bge-m3", "hybrid"]
DEFAULT_TOP_K = 100
BATCH_SIZE = 32
```

## 🔧 Core Components

### 1. MariaDB Integration (`database/mariadb_client.py`)
- Dummy wrapper class for database operations
- Query candidates with resume attachments
- Track modification timestamps for sync
- Handle Chinese/English text encoding

### 2. Embedding Models (`models/embeddings.py`)
- BGE-M3 dense embedding (1024 dimensions)
- Optional BM25 sparse embedding
- Jieba tokenization for Chinese text
- Bilingual text preprocessing

### 3. Qdrant Operations (`database/qdrant_client.py`)
- Collection initialization with hybrid vectors
- Batch insertion and updates
- Hybrid search queries
- Result ranking and scoring

### 4. Search Interface (`core/searcher.py`)
```python
def hybrid_search(
    job_description: str,
    additional_info: str = "",
    search_mode: str = "bge-m3",  # or "hybrid"
    top_k: int = 100
) -> List[SearchResult]
```

### 5. Sync Management (`core/sync_manager.py`)
- Manual sync interface
- Incremental updates based on timestamps
- Batch processing for large datasets
- Error handling and logging

## 🚀 Usage Examples

### Terminal Interface
```bash
# Initialize the system
python main.py init

# Sync data from MariaDB
python main.py sync

# Search for candidates
python main.py search "Python developer with 5+ years experience"

# Search with additional criteria
python main.py search "Data scientist" --additional "Machine learning, pandas"

# Check system status
python main.py status

# Switch search mode
python main.py search "Java developer" --mode hybrid
```

### Python API
```python
from core.searcher import ResumeSearcher

searcher = ResumeSearcher()
results = searcher.hybrid_search(
    job_description="Senior Python Developer",
    additional_info="Django, PostgreSQL, AWS",
    search_mode="hybrid",
    top_k=50
)

for result in results:
    print(f"Candidate: {result.name}")
    print(f"Score: {result.score}")
    print(f"MariaDB ID: {result.candidate_id}")
```

## 🔄 Data Synchronization

### Sync Process
1. Query MariaDB for candidates with resumes
2. Compare modification timestamps
3. Extract and process text content
4. Generate embeddings (dense + sparse)
5. Upsert to Qdrant collection
6. Update sync metadata

### Incremental Updates
```sql
SELECT c.candidate_id, c.first_name, c.last_name, c.email1, 
       c.key_skills, c.notes, a.text as resume_text,
       GREATEST(c.date_modified, a.date_modified) as last_modified
FROM candidate c 
LEFT JOIN attachment a ON c.candidate_id = a.data_item_id 
WHERE a.resume = 1 
  AND a.data_item_type = 100 
  AND GREATEST(c.date_modified, a.date_modified) > ?
```

## 📊 Search Results Format

```python
SearchResult = {
    "candidate_id": int,        # MariaDB primary key
    "name": str,               # Full name
    "email": str,              # Contact email
    "score": float,            # Relevance score
    "key_skills": str,         # Skills summary
    "notes": str,              # Additional notes
    "resume_snippet": str,     # Relevant text excerpt
    "search_mode": str,        # Used search mode
    "rank": int               # Result ranking
}
```

## 🧪 Testing Strategy

### Unit Tests
- Embedding generation accuracy
- Text processing (Chinese/English)
- Database connection handling
- Search result ranking

### Integration Tests
- End-to-end search workflow
- Sync mechanism validation
- Performance benchmarking
- Multi-language query testing

### Performance Metrics
- Search latency (target: <500ms for 100 results)
- Sync throughput (target: 1000 resumes/minute)
- Memory usage optimization
- Relevance scoring accuracy

## 🔮 Implementation Roadmap

### Phase 1: Foundation
- [x] Project structure setup
- [ ] Configuration management
- [ ] MariaDB dummy wrapper
- [ ] Basic logging system

### Phase 2: Core Components
- [ ] BGE-M3 embedding integration
- [ ] Jieba tokenization setup
- [ ] Qdrant collection initialization
- [ ] Text processing utilities

### Phase 3: Search Engine
- [ ] Hybrid search implementation
- [ ] Result ranking and scoring
- [ ] Chinese/English text handling
- [ ] Performance optimization

### Phase 4: Data Management
- [ ] MariaDB sync mechanism
- [ ] Incremental update logic
- [ ] Error handling and recovery
- [ ] Batch processing optimization

### Phase 5: Interface & Testing
- [ ] Terminal CLI interface
- [ ] Search result formatting
- [ ] Performance testing
- [ ] Documentation completion

## 🚨 Key Technical Considerations

### Language Processing
- **Chinese**: Jieba segmentation, traditional/simplified handling
- **English**: Standard tokenization, stemming considerations
- **Mixed Content**: Bilingual text processing strategies

### Performance Optimization
- **Batch Processing**: Efficient embedding generation
- **Memory Management**: Large dataset handling
- **Caching**: Frequent query optimization
- **Indexing**: Fast retrieval strategies

### Error Handling
- **Database Connectivity**: Retry mechanisms
- **Embedding Failures**: Fallback strategies  
- **Sync Interruptions**: Resume capability
- **Invalid Data**: Graceful degradation

## 📝 Development Guidelines

### Code Style
- Follow PEP 8 standards
- Comprehensive docstrings
- Type hints for all functions
- Modular, testable design

### Logging Standards
- Structured logging with timestamps
- Different levels: DEBUG, INFO, WARN, ERROR
- Performance metrics logging
- Error stack traces

### Configuration Management
- Environment-based configs
- Secure credential handling
- Easy deployment switching
- Validation of all settings

---

## 🎯 Success Criteria

1. **Functional**: Successfully search and return top 100 relevant candidates
2. **Performance**: Sub-second search response times
3. **Accuracy**: High relevance matching for both languages
4. **Reliability**: Robust sync mechanism with error recovery
5. **Usability**: Intuitive terminal interface for testing
6. **Maintainability**: Clean, modular, well-documented codebase

This README serves as the definitive guide for implementing and maintaining the resume retrieval system. All implementation should follow these specifications and architectural decisions.
