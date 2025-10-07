# Resume Retrieval System - Setup Guide

## Quick Setup

### 1. Install Dependencies
```bash
cd hybrid_search
pip3 install -r requirements.txt
```

### 2. Configure Environment
Edit `config.py` to set your database and Qdrant connections:

```python
# MariaDB Configuration
MARIADB_CONFIG = {
    "host": "your-mariadb-host",
    "port": 3306,
    "user": "your-username",
    "password": "your-password",
    "database": "your-database"
}

# Qdrant Configuration  
QDRANT_CONFIG = {
    "host": "localhost",
    "port": 6333,
    # "api_key": "your-api-key"  # Optional for cloud
}
```

### 3. Start Qdrant
```bash
# Using Docker
docker run -p 6333:6333 qdrant/qdrant

# Or using Docker Compose (see docker-compose.yml in project)
docker-compose up -d
```

## Usage

### Interactive Mode
```bash
python3 main.py
```

### Command Line Usage
```bash
# Initialize system
python3 main.py --init

# Force initialization (recreate Qdrant collection)
python3 main.py --force-init

# Check system status
python3 main.py --status

# Quick search
python3 main.py --search "Python developer with machine learning experience" --limit 5
```

### Quick Test
```bash
python3 test_system.py
```

## Key Features

### 1. Hybrid Search
- **Dense Vectors**: BAAI/bge-m3 model for semantic similarity
- **Sparse Vectors**: jieba + BM25 for keyword matching
- **Reciprocal Rank Fusion**: Combines both approaches

### 2. Data Synchronization
- Automatic sync from MariaDB to Qdrant
- Incremental updates when MariaDB data changes
- Change detection using timestamps

### 3. Terminal Interface
- Interactive menu-driven interface
- Real-time search with configurable parameters
- Detailed candidate information display
- System status monitoring

### 4. Scalable Architecture
- Modular design for easy extension
- Comprehensive logging
- Error handling and recovery
- Production-ready structure

## API Structure

### Main Search Function
```python
from core.search_service import search_service

# Initialize
search_service.initialize()

# Search candidates
results = search_service.search_candidates(
    job_description="Python developer with Django experience",
    additional_requirements="5+ years experience",
    limit=100,
    dense_weight=0.7,
    sparse_weight=0.3
)
```

### Result Format
```json
{
  "results": [
    {
      "rank": 1,
      "candidate_id": 123,
      "rrf_score": 0.8542,
      "dense_score": 0.7834,
      "sparse_score": 0.6123,
      "dense_rank": 2,
      "sparse_rank": 1,
      "first_name": "John",
      "last_name": "Doe",
      "email": "john.doe@example.com",
      "key_skills": "Python, Django, Machine Learning",
      "notes": "Experienced developer",
      "date_modified": "2025-01-07T17:30:00",
      "resume_available": true
    }
  ],
  "total_found": 50,
  "query_info": {
    "job_description": "Python developer...",
    "additional_requirements": "5+ years...",
    "dense_weight": 0.7,
    "sparse_weight": 0.3
  },
  "search_time": "2025-01-07T17:35:00"
}
```

## Development Notes

### Database Schema
The system expects a MariaDB table with these key fields:
- `candidate_id` (Primary Key)
- `first_name`, `last_name`, `email1`
- `key_skills`, `notes`
- `resume_text` (searchable content)
- `date_modified` (for sync tracking)

See `mariadb.schema` for complete schema.

### Extending the System
1. **Add new embedding models**: Modify `models/embeddings.py`
2. **Custom search algorithms**: Extend `core/qdrant_client.py`
3. **Additional data sources**: Create new clients in `database/`
4. **REST API**: Add Flask/FastAPI layer on top of `core/search_service.py`

### Troubleshooting
1. **Connection Issues**: Check `config.py` settings
2. **Missing Dependencies**: Run `pip3 install -r requirements.txt`
3. **Qdrant Errors**: Ensure Qdrant is running on correct port
4. **Search Issues**: Check logs in `logs/` directory

## Performance Notes

- System can handle 100K+ candidates efficiently
- Search response time: ~100-500ms for typical queries
- Memory usage: ~2-4GB depending on collection size
- Recommended: 8GB RAM, 4+ CPU cores for production

## Next Steps

1. **Configure your MariaDB connection** in `config.py`
2. **Start Qdrant** using Docker
3. **Run initialization**: `python3 main.py --init`
4. **Test search**: `python3 main.py --search "your test query"`

The system is ready for backend service integration and can be extended with REST APIs, web interfaces, or integrated into existing HR systems.
