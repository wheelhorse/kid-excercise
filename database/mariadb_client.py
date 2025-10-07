"""
MariaDB client for the Resume Retrieval System
"""
import pymysql
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from contextlib import contextmanager

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
    attachment_id: Optional[int] = None


class MariaDBClient:
    """MariaDB client for candidate data operations"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the MariaDB client"""
        self.config = config or MARIADB_CONFIG
        self.connection = None
        self.last_sync_time = None
        logger.info("MariaDB client initialized")
    
    @log_method("hybrid_search.mariadb")
    def connect(self) -> bool:
        """Connect to MariaDB database"""
        try:
            self.connection = pymysql.connect(
                host=self.config["host"],
                port=self.config["port"],
                user=self.config["user"],
                password=self.config["password"],
                database=self.config["database"],
                charset=self.config["charset"],
                autocommit=True,
                connect_timeout=10,
                read_timeout=30
            )
            logger.info(f"Connected to MariaDB at {self.config['host']}:{self.config['port']}")
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
                    a.attachment_id,
                    a.text as resume_text
                FROM candidate c
                LEFT JOIN attachment a ON (a.data_item_id = c.candidate_id AND a.data_item_type = 100)
                WHERE a.text IS NOT NULL 
                    AND a.text != ''
                    AND c.is_active = 1
                ORDER BY c.date_modified DESC
                """
                
                if limit:
                    query += f" LIMIT {limit}"
                
                cursor.execute(query)
                rows = cursor.fetchall()
                
                candidates = []
                for row in rows:
                    candidate = CandidateRecord(
                        candidate_id=row['candidate_id'],
                        first_name=row['first_name'] or '',
                        last_name=row['last_name'] or '',
                        email1=row['email1'] or '',
                        key_skills=row['key_skills'] or '',
                        notes=row['notes'] or '',
                        date_modified=row['date_modified'],
                        resume_text=row['resume_text'],
                        attachment_id=row['attachment_id']
                    )
                    candidates.append(candidate)
                
                logger.info(f"Retrieved {len(candidates)} candidate records")
                return candidates
                
        except Exception as e:
            logger.error(f"Error retrieving candidates: {str(e)}")
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
                    a.attachment_id,
                    a.text as resume_text
                FROM candidate c
                LEFT JOIN attachment a ON (a.data_item_id = c.candidate_id AND a.data_item_type = 100)
                WHERE c.date_modified > %s
                    AND a.text IS NOT NULL 
                    AND a.text != ''
                    AND c.is_active = 1
                ORDER BY c.date_modified DESC
                """
                
                if limit:
                    query += f" LIMIT {limit}"
                
                cursor.execute(query, (timestamp,))
                rows = cursor.fetchall()
                
                candidates = []
                for row in rows:
                    candidate = CandidateRecord(
                        candidate_id=row['candidate_id'],
                        first_name=row['first_name'] or '',
                        last_name=row['last_name'] or '',
                        email1=row['email1'] or '',
                        key_skills=row['key_skills'] or '',
                        notes=row['notes'] or '',
                        date_modified=row['date_modified'],
                        resume_text=row['resume_text'],
                        attachment_id=row['attachment_id']
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
                    a.attachment_id,
                    a.text as resume_text
                FROM candidate c
                LEFT JOIN attachment a ON (a.data_item_id = c.candidate_id AND a.data_item_type = 100)
                WHERE c.candidate_id = %s
                    AND a.text IS NOT NULL 
                    AND a.text != ''
                    AND c.is_active = 1
                LIMIT 1
                """
                
                cursor.execute(query, (candidate_id,))
                row = cursor.fetchone()
                
                if row:
                    candidate = CandidateRecord(
                        candidate_id=row['candidate_id'],
                        first_name=row['first_name'] or '',
                        last_name=row['last_name'] or '',
                        email1=row['email1'] or '',
                        key_skills=row['key_skills'] or '',
                        notes=row['notes'] or '',
                        date_modified=row['date_modified'],
                        resume_text=row['resume_text'],
                        attachment_id=row['attachment_id']
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
        """Get total number of candidates with resumes"""
        try:
            with self.get_cursor() as cursor:
                query = """
                SELECT COUNT(DISTINCT c.candidate_id) as count
                FROM candidate c
                LEFT JOIN attachment a ON (a.data_item_id = c.candidate_id AND a.data_item_type = 100)
                WHERE a.text IS NOT NULL 
                    AND a.text != ''
                    AND c.is_active = 1
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
                WHERE a.text IS NOT NULL 
                    AND a.text != ''
                    AND c.is_active = 1
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
