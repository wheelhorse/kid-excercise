"""
Logging utilities for the Resume Retrieval System
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from config import LOG_LEVEL, LOG_FORMAT, LOG_FILE


class Logger:
    """Centralized logging utility"""
    
    _loggers = {}
    
    @classmethod
    def get_logger(cls, name: str, log_file: Optional[str] = None) -> logging.Logger:
        """Get or create a logger instance"""
        if name in cls._loggers:
            return cls._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, LOG_LEVEL.upper()))
        
        # Remove existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create formatter
        formatter = logging.Formatter(LOG_FORMAT)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        if log_file or LOG_FILE:
            file_path = Path(log_file or LOG_FILE)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(file_path, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        logger.propagate = False
        cls._loggers[name] = logger
        
        return logger


# Global logger instances
main_logger = Logger.get_logger("hybrid_search.main")
db_logger = Logger.get_logger("hybrid_search.database")
search_logger = Logger.get_logger("hybrid_search.search")
sync_logger = Logger.get_logger("hybrid_search.sync")
embedding_logger = Logger.get_logger("hybrid_search.embedding")


def log_performance(func):
    """Decorator to log function performance"""
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger = Logger.get_logger(f"performance.{func.__module__}.{func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"Function {func.__name__} completed in {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Function {func.__name__} failed after {elapsed:.3f}s: {str(e)}")
            raise
    
    return wrapper


def log_method(logger_name: Optional[str] = None):
    """Decorator to log method calls with custom logger"""
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if logger_name:
                logger = Logger.get_logger(logger_name)
            else:
                logger = Logger.get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
            
            logger.debug(f"Calling {func.__name__} with args={args[:2]}{'...' if len(args) > 2 else ''}")
            
            try:
                result = func(self, *args, **kwargs)
                logger.debug(f"Method {func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"Method {func.__name__} failed: {str(e)}")
                raise
        
        return wrapper
    return decorator
