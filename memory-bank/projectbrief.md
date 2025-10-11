# Project Brief: Resume Retrieval System with Hybrid Search

## Project Overview

A sophisticated Python-based resume retrieval system that combines **Qdrant vector database + BAAI/bge-m3 embeddings + jieba/BM25** to provide intelligent candidate search capabilities for recruitment processes.

## Core Mission

Build a high-performance, bilingual (Chinese/English) resume search engine that can:
- Process job descriptions and return the top 100 most relevant candidates
- Combine semantic understanding (dense embeddings) with keyword matching (sparse embeddings)
- Automatically optimize for different CPU architectures
- Provide sub-second search response times
- Handle large-scale resume databases with efficient synchronization

## Primary Requirements

### Functional Requirements
1. **Hybrid Search Engine**: BGE-M3 dense + BM25 sparse embeddings with RRF fusion
2. **Database Integration**: Sync resume data from MariaDB to Qdrant vector store
3. **Bilingual Processing**: Full support for Chinese (jieba tokenization) and English text
4. **Terminal Interface**: Command-line testing and operations interface
5. **ID Mapping**: Maintain references back to original MariaDB records
6. **Real-time Updates**: Incremental synchronization based on modification timestamps

### Performance Requirements
- **Search Speed**: Sub-second response times for 100 candidates
- **Throughput**: 1000+ resumes processed per minute during sync
- **Accuracy**: High relevance matching across both languages
- **Scalability**: Handle large resume databases efficiently

### Technical Requirements
- **CPU Optimization**: Automatic detection and optimization for Intel/AMD architectures
- **Error Recovery**: Robust sync mechanism with retry capabilities
- **Configuration**: Environment-based settings for easy deployment
- **Logging**: Comprehensive performance and error tracking

## Success Criteria

### Must Have
- ✅ Successfully search and return top 100 relevant candidates
- ✅ Sub-second search response times
- ✅ High relevance matching for both Chinese and English
- ✅ Robust sync mechanism with error recovery
- ✅ Intuitive terminal interface for testing

### Should Have
- CPU-optimized performance across different architectures
- Comprehensive logging and monitoring
- Clean, modular, maintainable codebase
- Comprehensive test coverage

### Could Have
- Web interface for broader usage
- Advanced analytics and reporting
- Multi-tenant support
- Cloud deployment automation

## Technical Architecture Overview

```
MariaDB (Source) → Sync Manager → Qdrant (Vector Store)
                                      ↓
Job Description → Smart Embeddings → Hybrid Search → Top 100 Results
```

### Key Components
1. **Smart Embedding Factory**: Auto-detects CPU and selects optimal embedding implementation
2. **Hybrid Search Service**: Combines dense and sparse search with RRF ranking
3. **Sync Manager**: Handles incremental data updates from MariaDB
4. **Terminal Interface**: Interactive testing and operations

### Database Schema
- **MariaDB**: `candidate` + `attachment` tables with resume text
- **Qdrant**: Vector collection with candidate payloads and hybrid embeddings

## Project Scope

### In Scope
- Resume search functionality
- Hybrid embedding implementation
- CPU optimization
- Terminal interface
- Data synchronization
- Performance optimization
- Chinese/English support

### Out of Scope
- Web UI development (initially)
- Real-time notifications
- User authentication
- Advanced analytics dashboard
- Multi-database support

## Key Constraints

### Technical Constraints
- CPU-only inference (no GPU dependency)
- Memory efficient processing for large datasets
- Backward compatibility with existing MariaDB schema
- Environment-based configuration only

### Business Constraints
- Must work with existing CATS recruitment database
- No modifications to source MariaDB schema
- Maintain data consistency and integrity
- Support existing workflow patterns

## Risk Assessment

### High Risk
- **Data Sync Reliability**: Critical for maintaining accuracy
- **Search Performance**: Must meet sub-second requirements
- **Memory Usage**: Large embedding models on CPU systems

### Medium Risk
- **CPU Optimization**: Different performance across architectures
- **Text Encoding**: Chinese/English mixed content handling
- **Error Recovery**: Robust handling of various failure modes

### Low Risk
- **Configuration Management**: Well-established patterns
- **Logging Implementation**: Standard Python logging
- **Terminal Interface**: Straightforward CLI implementation

## Definition of Done

A candidate search system is complete when:
1. All functional requirements are implemented and tested
2. Performance benchmarks are met (sub-second search, 1000+ resumes/minute sync)
3. CPU optimization works across Intel and AMD architectures
4. Comprehensive error handling and recovery mechanisms are in place
5. Terminal interface provides all necessary testing capabilities
6. Documentation is complete and maintenance procedures are established
7. Code follows established patterns and is fully tested

## Project Timeline Phases

### Phase 1: Foundation ✅
- Project structure and configuration
- Basic embedding implementations
- Database connections and basic operations

### Phase 2: Core Features ✅
- Smart embedding factory with CPU optimization
- Hybrid search implementation
- Sync manager with incremental updates

### Phase 3: Integration & Testing 🔄
- Terminal interface completion
- End-to-end testing
- Performance benchmarking and optimization

### Phase 4: Production Ready
- Comprehensive error handling
- Documentation completion
- Deployment procedures
- Performance validation

This project brief serves as the foundational reference for all implementation decisions and architectural choices throughout the development lifecycle.
