"""
MariaDB client for the Resume Retrieval System
"""
import pymysql
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from contextlib import contextmanager
import chardet

from utils.logger import Logger, log_method
from config import MARIADB_CONFIG

logger = Logger.get_logger("hybrid_search.mariadb")


@dataclass
class CandidateRecord:
    """Data class for candidate records"""
    candidate_id: int
    first_name: str
    last_name: str
    email1: str
    key_skills: str
    notes: str
    date_modified: datetime
    resume_text: Optional[str] = None


class MariaDBClient:
    """MariaDB client for candidate data operations"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the MariaDB client"""
        self.config = config or MARIADB_CONFIG
        self.connection = None
        self.last_sync_time = None
        logger.info("MariaDB client initialized")
    
    def _safe_decode_text(self, text: Any) -> str:
        """Safely decode text data with encoding error handling"""
        if text is None:
            return ""
        
        # If it's already a string, return it
        if isinstance(text, str):
            return text
        
        # If it's bytes, try to decode it safely
        if isinstance(text, bytes):
            try:
                # First try UTF-8
                return text.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    # Try to detect encoding
                    detected = chardet.detect(text)
                    if detected and detected['encoding']:
                        encoding = detected['encoding']
                        logger.warning(f"Detected non-UTF-8 encoding: {encoding}")
                        return text.decode(encoding, errors='replace')
                except Exception as e:
                    logger.warning(f"Encoding detection failed: {str(e)}")
                
                # Fallback: decode with latin-1 and replace errors
                try:
                    return text.decode('latin-1', errors='replace')
                except Exception as e:
                    logger.warning(f"Latin-1 decode failed: {str(e)}")
                    # Ultimate fallback: decode with UTF-8 and replace errors
                    return text.decode('utf-8', errors='replace')
        
        # For any other type, convert to string
        return str(text)
    
    def _sanitize_resume_text(self, text: str) -> str:
        """Sanitize resume text by removing problematic characters"""
        if not text:
            return ""
        
        # Remove null bytes and other control characters except newlines and tabs
        sanitized = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t\r')
        
        # Replace multiple whitespace with single space
        import re
        sanitized = re.sub(r'\s+', ' ', sanitized)
        
        return sanitized.strip()
    
    @log_method("hybrid_search.mariadb")
    def connect(self) -> bool:
        """Connect to MariaDB database with raw data handling"""
        try:
            # Connect without charset to get raw bytes
            self.connection = pymysql.connect(
                host=self.config["host"],
                port=self.config["port"],
                user=self.config["user"],
                password=self.config["password"],
                database=self.config["database"],
                charset=None,  # Use binary to get raw bytes
                autocommit=True,
                connect_timeout=10,
                read_timeout=30,
                use_unicode=False  # Disable automatic unicode conversion
            )
            logger.info(f"Connected to MariaDB at {self.config['host']}:{self.config['port']} (raw mode)")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MariaDB: {str(e)}")
            self.connection = None
            return False
    
    @log_method("hybrid_search.mariadb")
    def disconnect(self):
        """Disconnect from MariaDB database"""
        if self.connection:
            try:
                self.connection.close()
                self.connection = None
                logger.info("Disconnected from MariaDB")
            except Exception as e:
                logger.error(f"Error disconnecting from MariaDB: {str(e)}")
    
    def is_connected(self) -> bool:
        """Check if connected to database"""
        if not self.connection:
            return False
        try:
            self.connection.ping(reconnect=True)
            return True
        except Exception:
            return False
    
    @contextmanager
    def get_cursor(self):
        """Get database cursor with automatic cleanup"""
        if not self.is_connected():
            if not self.connect():
                raise Exception("Unable to connect to database")
        
        cursor = self.connection.cursor(pymysql.cursors.DictCursor)
        try:
            yield cursor
        finally:
            cursor.close()
    
    def _safe_fetchall(self, cursor) -> List[Dict[str, Any]]:
        """Safely fetch all rows, decoding raw bytes to strings"""
        rows = []
        
        # With binary connection, we always get raw bytes, so process each row
        while True:
            try:
                row = cursor.fetchone()
                if row is None:
                    break
                    
                # Process each field in the row to handle raw bytes
                safe_row = {}
                for key, value in row.items():
                    # Decode the key if it's bytes
                    safe_key = self._safe_decode_text(key) if isinstance(key, bytes) else key
                    # Decode the value
                    safe_row[safe_key] = self._safe_decode_text(value)
                    
                rows.append(safe_row)
                
            except Exception as row_error:
                logger.warning(f"Error processing row: {str(row_error)}")
                continue
        
        return rows
    
    @log_method("hybrid_search.mariadb")
    def get_candidates_with_resumes(self, limit: Optional[int] = None) -> List[CandidateRecord]:
        """Get all candidates with resume attachments"""
        try:
            with self.get_cursor() as cursor:
                # Query to get candidates with their resume attachments
                query = """
                SELECT DISTINCT
                    c.candidate_id,
                    c.first_name,
                    c.last_name,
                    c.email1,
                    c.key_skills,
                    c.notes,
                    c.date_modified,
                    GROUP_CONCAT(
                        CASE
                            WHEN a.text IS NOT NULL AND a.text != '' THEN a.text
                            ELSE NULL
                        END
                        SEPARATOR '\n\n'  -- or any separator you like
                    ) AS resume_text
                FROM candidate c
                LEFT JOIN attachment a 
                    ON a.data_item_id = c.candidate_id AND a.data_item_type = 100
                WHERE c.is_active = 1
                GROUP BY 
                    c.candidate_id,
                    c.first_name,
                    c.last_name,
                    c.email1,
                    c.key_skills,
                    c.notes,
                    c.date_modified
                ORDER BY c.date_modified DESC
                """
                
                if limit:
                    query += f" LIMIT {limit};"
                else:
                    query += f";"
                
                try:
                    cursor.execute(query)
                except Exception as e:
                    logger.error('execute sql command failed')
                    logger.error(f"===Error retrieving candidates: {str(e)}")
                    return []

                rows = self._safe_fetchall(cursor)
                
                candidates = []
                for row in rows:
                    # Safely decode and sanitize text fields
                    resume_text = self._safe_decode_text(row['resume_text'])
                    resume_text = self._sanitize_resume_text(resume_text)
                    
                    candidate = CandidateRecord(
                        candidate_id=row['candidate_id'],
                        first_name=self._safe_decode_text(row['first_name']),
                        last_name=self._safe_decode_text(row['last_name']),
                        email1=self._safe_decode_text(row['email1']),
                        key_skills=self._safe_decode_text(row['key_skills']),
                        notes=self._safe_decode_text(row['notes']),
                        date_modified=row['date_modified'],
                        resume_text=resume_text,
                    )
                    candidates.append(candidate)
                
                logger.info(f"Retrieved {len(candidates)} candidate records")
                return candidates
                
        except Exception as e:
            logger.error(f"===Error retrieving candidates: {str(e)}")
            return []
    
    @log_method("hybrid_search.mariadb")
    def get_candidates_modified_after(self, timestamp: datetime, limit: Optional[int] = None) -> List[CandidateRecord]:
        """Get candidates modified after a specific timestamp"""
        try:
            with self.get_cursor() as cursor:
                query = """
                SELECT DISTINCT
                    c.candidate_id,
                    c.first_name,
                    c.last_name,
                    c.email1,
                    c.key_skills,
                    c.notes,
                    c.date_modified,
                    GROUP_CONCAT(
                        CASE
                            WHEN a.text IS NOT NULL AND a.text != '' THEN a.text
                            ELSE NULL
                        END
                        SEPARATOR '\n\n'  -- or any separator you like
                    ) AS resume_text
                FROM candidate c
                LEFT JOIN attachment a 
                    ON a.data_item_id = c.candidate_id AND a.data_item_type = 100
                WHERE c.date_modified > %s
                    AND c.is_active = 1
                GROUP BY 
                    c.candidate_id,
                    c.first_name,
                    c.last_name,
                    c.email1,
                    c.key_skills,
                    c.notes,
                    c.date_modified
                ORDER BY c.date_modified DESC;
                """
                
                if limit:
                    query += f" LIMIT {limit}"
                
                cursor.execute(query, (timestamp,))
                rows = self._safe_fetchall(cursor)
                
                candidates = []
                for row in rows:
                    # Safely decode and sanitize text fields
                    resume_text = self._safe_decode_text(row['resume_text'])
                    resume_text = self._sanitize_resume_text(resume_text)
                    
                    candidate = CandidateRecord(
                        candidate_id=row['candidate_id'],
                        first_name=self._safe_decode_text(row['first_name']),
                        last_name=self._safe_decode_text(row['last_name']),
                        email1=self._safe_decode_text(row['email1']),
                        key_skills=self._safe_decode_text(row['key_skills']),
                        notes=self._safe_decode_text(row['notes']),
                        date_modified=row['date_modified'],
                        resume_text=resume_text,
                    )
                    candidates.append(candidate)
                
                logger.info(f"Retrieved {len(candidates)} candidates modified after {timestamp}")
                return candidates
                
        except Exception as e:
            logger.error(f"Error retrieving modified candidates: {str(e)}")
            return []
    
    @log_method("hybrid_search.mariadb")
    def get_candidate_by_id(self, candidate_id: int) -> Optional[CandidateRecord]:
        """Get a specific candidate by ID"""
        try:
            with self.get_cursor() as cursor:
                query = """
                SELECT DISTINCT
                    c.candidate_id,
                    c.first_name,
                    c.last_name,
                    c.email1,
                    c.key_skills,
                    c.notes,
                    c.date_modified,
                    GROUP_CONCAT(
                        CASE
                            WHEN a.text IS NOT NULL AND a.text != '' THEN a.text
                            ELSE NULL
                        END
                        SEPARATOR '\n\n'  -- or any separator you like
                    ) AS resume_text
                FROM candidate c
                LEFT JOIN attachment a 
                    ON a.data_item_id = c.candidate_id AND a.data_item_type = 100
                WHERE c.candidate_id = %s
                    AND c.is_active = 1
                GROUP BY 
                    c.candidate_id,
                    c.first_name,
                    c.last_name,
                    c.email1,
                    c.key_skills,
                    c.notes,
                    c.date_modified
                ORDER BY c.date_modified DESC;
                """
                
                cursor.execute(query, (candidate_id,))
                row = cursor.fetchone()
                
                if row:
                    # Process the single row to handle raw bytes
                    safe_row = {}
                    for key, value in row.items():
                        safe_key = self._safe_decode_text(key) if isinstance(key, bytes) else key
                        safe_row[safe_key] = self._safe_decode_text(value)
                    row = safe_row
                    # Safely decode and sanitize text fields
                    resume_text = self._safe_decode_text(row['resume_text'])
                    resume_text = self._sanitize_resume_text(resume_text)
                    
                    candidate = CandidateRecord(
                        candidate_id=row['candidate_id'],
                        first_name=self._safe_decode_text(row['first_name']),
                        last_name=self._safe_decode_text(row['last_name']),
                        email1=self._safe_decode_text(row['email1']),
                        key_skills=self._safe_decode_text(row['key_skills']),
                        notes=self._safe_decode_text(row['notes']),
                        date_modified=row['date_modified'],
                        resume_text=resume_text,
                    )
                    logger.info(f"Retrieved candidate {candidate_id}")
                    return candidate
                else:
                    logger.warning(f"Candidate {candidate_id} not found")
                    return None
                    
        except Exception as e:
            logger.error(f"Error retrieving candidate {candidate_id}: {str(e)}")
            return None
    
    @log_method("hybrid_search.mariadb")
    def get_total_candidates_count(self) -> int:
        """Get total number of active candidates"""
        try:
            with self.get_cursor() as cursor:
                query = """
                SELECT COUNT(c.candidate_id) as count
                FROM candidate c
                WHERE c.is_active = 1
                """
                
                cursor.execute(query)
                result = cursor.fetchone()
                count = result['count'] if result else 0
                
                logger.info(f"Total candidates count: {count}")
                return count
                
        except Exception as e:
            logger.error(f"Error getting candidates count: {str(e)}")
            return 0
    
    @log_method("hybrid_search.mariadb")
    def get_last_modification_time(self) -> Optional[datetime]:
        """Get the latest modification timestamp"""
        try:
            with self.get_cursor() as cursor:
                query = """
                SELECT MAX(c.date_modified) as latest_modified
                FROM candidate c
                LEFT JOIN attachment a ON (a.data_item_id = c.candidate_id AND a.data_item_type = 100)
                WHERE c.is_active = 1
                """
                
                cursor.execute(query)
                result = cursor.fetchone()
                
                if result and result['latest_modified']:
                    latest = result['latest_modified']
                    logger.info(f"Latest modification time: {latest}")
                    return latest
                else:
                    logger.info("No modification time found")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting last modification time: {str(e)}")
            return None
    
    def test_connection(self) -> bool:
        """Test database connectivity"""
        try:
            if not self.is_connected():
                if not self.connect():
                    return False
            
            with self.get_cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                return result is not None
                
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    @log_method("hybrid_search.mariadb")
    def test_text_encoding(self, limit: int = 10) -> Dict[str, Any]:
        """Test text encoding issues in attachment table"""
        try:
            with self.get_cursor() as cursor:
                # Get a small sample to test encoding
                query = """
                SELECT a.attachment_id, a.data_item_id, 
                       LENGTH(a.text) as text_length,
                       LEFT(a.text, 100) as text_sample
                FROM attachment a 
                WHERE a.text IS NOT NULL AND a.text != ''
                ORDER BY a.attachment_id 
                LIMIT %s
                """
                
                cursor.execute(query, (limit,))
                rows = self._safe_fetchall(cursor)
                
                results = {
                    'total_tested': len(rows),
                    'encoding_issues': 0,
                    'successfully_decoded': 0,
                    'problematic_records': []
                }
                
                for row in rows:
                    try:
                        # Try to decode the text
                        decoded_text = self._safe_decode_text(row['text_sample'])
                        results['successfully_decoded'] += 1
                    except Exception as e:
                        results['encoding_issues'] += 1
                        results['problematic_records'].append({
                            'data_item_id': row['data_item_id'],
                            'text_length': row['text_length'],
                            'error': str(e)
                        })
                
                logger.info(f"Encoding test results: {results}")
                return results
                
        except Exception as e:
            logger.error(f"Error testing text encoding: {str(e)}")
            return {'error': str(e)}
    
    def set_last_sync_time(self, timestamp: datetime):
        """Set the last successful sync timestamp"""
        self.last_sync_time = timestamp
        logger.info(f"Set last sync time to: {timestamp}")
    
    def get_last_sync_time(self) -> Optional[datetime]:
        """Get the last successful sync timestamp"""
        return self.last_sync_time
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()


# Global instance for easy access
mariadb_client = MariaDBClient()
