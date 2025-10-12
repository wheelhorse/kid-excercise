"""
Common text preprocessing utilities for all embedding models
Centralizes text preprocessing logic to reduce duplication and improve maintainability
"""
import re
from typing import List
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from utils.text_processor import text_processor
from utils.logger import Logger
from config import MAX_TEXT_LENGTH
import os

logger = Logger.get_logger("hybrid_search.models.text_preprocessing")


def _custom_filter_text(text: str) -> str:
    """
    Apply custom filtering rules before standard text cleaning
    
    Args:
        text: Text to filter
        
    Returns:
        Filtered text
    """
    # Remove hash-like strings (e.g., '903032fca51a55dc1XV90t21F1NUw426UfOdWOKimfDQNhlm1w~~')
    # Pattern: alphanumeric strings of 40+ characters ending with '~~'
    text = re.sub(r'\b[a-zA-Z0-9_]{20,}~~\b', '', text)
    
    # Also remove similar patterns without '~~' but with similar characteristics
    # Pattern: long alphanumeric strings that look like hashes/tokens
    text = re.sub(r'\b[a-zA-Z0-9_]{22,}\b', '', text)
    
    # Replace '_rDOTr' with '.'
    text = text.replace('_rDOTr', '.')
    
    # Clean up multiple spaces that might result from removals
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def _process_single_text(text: str) -> str:
    """
    Process a single text with consistent preprocessing logic
    
    Args:
        text: Text to preprocess
        
    Returns:
        Preprocessed text
    """
    # Apply custom filtering first
    text = _custom_filter_text(text)
    
    # Apply length truncation
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]
    
    # Apply standard text cleaning
    return text_processor.clean_text(text)


def preprocess_texts_consistent(texts: List[str], use_multiprocessing: bool = True) -> List[str]:
    """
    Consistent text preprocessing for all embedding models
    Uses multiprocessing for larger batches, threading for smaller ones
    
    Args:
        texts: List of texts to preprocess
        use_multiprocessing: Whether to use multiprocessing for large batches
        
    Returns:
        List of preprocessed texts
    """
    # For very small batches, just process sequentially
    if len(texts) <= 10:
        return [_process_single_text(text) for text in texts]
    
    # For medium batches, use threading (I/O bound)
    if len(texts) <= 100 or not use_multiprocessing:
        max_workers = min(os.cpu_count() or 4, len(texts), 32)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            processed_texts = list(executor.map(_process_single_text, texts))
        return processed_texts
    
    # For large batches, use multiprocessing (CPU bound)
    max_workers = min(os.cpu_count() or 4, len(texts) // 20 + 1, 8)
    chunk_size = max(len(texts) // (max_workers * 2), 10)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        processed_texts = list(executor.map(
            _process_single_text, 
            texts, 
            chunksize=chunk_size
        ))
    
    return processed_texts


def preprocess_single_text(text: str) -> str:
    """
    Preprocess a single text
    
    Args:
        text: Single text to preprocess
        
    Returns:
        Preprocessed text
    """
    return _process_single_text(text)


# Convenience function for backward compatibility
def get_common_preprocessor():
    """Get the common text preprocessor function"""
    return preprocess_texts_consistent
