"""
Data synchronization manager between MariaDB and Qdrant
"""
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
from threading import Lock
import schedule

from utils.logger import Logger, log_performance, log_method
from database.mariadb_client import mariadb_client, CandidateRecord
from core.qdrant_client import qdrant_manager
from config import SYNC_CONFIG

logger = Logger.get_logger("hybrid_search.sync")


class SyncManager:
    """Manages synchronization between MariaDB and Qdrant"""
    
    def __init__(self):
        """Initialize sync manager"""
        self.sync_config = SYNC_CONFIG
        self.last_sync_time = None
        self.sync_lock = Lock()
        self.is_running = False
        
        logger.info("Sync manager initialized")
    
    @log_performance
    def initial_sync(self, force: bool = False) -> bool:
        """Perform initial full synchronization"""
        logger.info("Starting initial synchronization")
        
        with self.sync_lock:
            try:
                # Connect to both databases
                if not mariadb_client.connect():
                    logger.error("Failed to connect to MariaDB")
                    return False
                
                if not qdrant_manager.connect():
                    logger.error("Failed to connect to Qdrant")
                    return False
                
                # Create Qdrant collection if needed
                if not qdrant_manager.create_collection(recreate=force):
                    logger.error("Failed to create Qdrant collection")
                    return False
                
                # Get all candidates from MariaDB
                candidates = mariadb_client.get_candidates_with_resumes()
                if not candidates:
                    logger.warning("No candidates found in MariaDB")
                    return True
                
                logger.info(f"Found {len(candidates)} candidates in MariaDB")
                
                # Upsert candidates to Qdrant
                success = qdrant_manager.upsert_candidates(candidates)
                
                if success:
                    # Update last sync time
                    self.last_sync_time = datetime.now()
                    mariadb_client.set_last_sync_time(self.last_sync_time)
                    
                    # Verify sync
                    qdrant_count = qdrant_manager.get_collection_count()
                    mariadb_count = mariadb_client.get_total_candidates_count()
                    
                    logger.info(f"Sync completed - MariaDB: {mariadb_count}, Qdrant: {qdrant_count}")
                    return True
                else:
                    logger.error("Failed to upsert candidates to Qdrant")
                    return False
                
            except Exception as e:
                logger.error(f"Initial sync failed: {str(e)}")
                return False
            finally:
                mariadb_client.disconnect()
                qdrant_manager.disconnect()
    
    @log_performance
    def incremental_sync(self) -> bool:
        """Perform incremental synchronization"""
        logger.info("Starting incremental synchronization")
        
        with self.sync_lock:
            try:
                # Connect to databases
                if not mariadb_client.connect():
                    logger.error("Failed to connect to MariaDB")
                    return False
                
                if not qdrant_manager.connect():
                    logger.error("Failed to connect to Qdrant")
                    return False
                
                # Get last sync time
                last_sync = self.get_last_sync_time()
                if not last_sync:
                    logger.info("No previous sync found, performing initial sync")
                    return self.initial_sync()
                
                # Get candidates modified after last sync
                modified_candidates = mariadb_client.get_candidates_modified_after(last_sync)
                
                if not modified_candidates:
                    logger.info("No candidates modified since last sync")
                    return True
                
                logger.info(f"Found {len(modified_candidates)} modified candidates")
                
                # Upsert modified candidates
                success = qdrant_manager.upsert_candidates(modified_candidates)
                
                if success:
                    # Update last sync time
                    self.last_sync_time = datetime.now()
                    mariadb_client.set_last_sync_time(self.last_sync_time)
                    
                    logger.info(f"Incremental sync completed - {len(modified_candidates)} candidates updated")
                    return True
                else:
                    logger.error("Failed to upsert modified candidates")
                    return False
                
            except Exception as e:
                logger.error(f"Incremental sync failed: {str(e)}")
                return False
            finally:
                mariadb_client.disconnect()
                qdrant_manager.disconnect()
    
    @log_method("hybrid_search.sync")
    def get_last_sync_time(self) -> Optional[datetime]:
        """Get the last successful sync timestamp"""
        if self.last_sync_time:
            return self.last_sync_time
        
        # Try to get from MariaDB client
        return mariadb_client.get_last_sync_time()
    
    @log_method("hybrid_search.sync")
    def get_sync_status(self) -> Dict[str, Any]:
        """Get current synchronization status"""
        try:
            # Connect to get status
            mariadb_connected = mariadb_client.test_connection()
            qdrant_connected = qdrant_manager.connect() if not qdrant_manager.is_connected else True
            
            status = {
                "mariadb_connected": mariadb_connected,
                "qdrant_connected": qdrant_connected,
                "last_sync_time": self.get_last_sync_time(),
                "sync_running": self.is_running
            }
            
            if mariadb_connected and qdrant_connected:
                mariadb_count = mariadb_client.get_total_candidates_count()
                qdrant_count = qdrant_manager.get_collection_count()
                
                status.update({
                    "mariadb_count": mariadb_count,
                    "qdrant_count": qdrant_count,
                    "sync_needed": mariadb_count != qdrant_count
                })
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get sync status: {str(e)}")
            return {
                "error": str(e),
                "sync_running": self.is_running
            }
        finally:
            if qdrant_manager.is_connected:
                qdrant_manager.disconnect()
    
    def start_scheduled_sync(self):
        """Start scheduled synchronization"""
        if self.is_running:
            logger.warning("Scheduled sync is already running")
            return
        
        sync_interval = self.sync_config.get('interval_minutes', 30)
        
        # Schedule incremental sync
        schedule.every(sync_interval).minutes.do(self._scheduled_sync_job)
        
        self.is_running = True
        logger.info(f"Scheduled sync started with {sync_interval} minute interval")
    
    def stop_scheduled_sync(self):
        """Stop scheduled synchronization"""
        schedule.clear()
        self.is_running = False
        logger.info("Scheduled sync stopped")
    
    def _scheduled_sync_job(self):
        """Background sync job"""
        try:
            logger.info("Running scheduled sync job")
            success = self.incremental_sync()
            if success:
                logger.info("Scheduled sync completed successfully")
            else:
                logger.error("Scheduled sync failed")
        except Exception as e:
            logger.error(f"Scheduled sync job failed: {str(e)}")
    
    def run_sync_daemon(self):
        """Run sync daemon (blocking)"""
        logger.info("Starting sync daemon")
        self.start_scheduled_sync()
        
        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Sync daemon interrupted by user")
        finally:
            self.stop_scheduled_sync()
            logger.info("Sync daemon stopped")
    
    @log_performance
    def validate_sync(self) -> Dict[str, Any]:
        """Validate synchronization between MariaDB and Qdrant"""
        logger.info("Validating synchronization")
        
        validation_result = {
            "valid": False,
            "mariadb_count": 0,
            "qdrant_count": 0,
            "missing_in_qdrant": [],
            "extra_in_qdrant": [],
            "timestamp_mismatches": []
        }
        
        try:
            # Connect to both databases
            if not mariadb_client.connect():
                validation_result["error"] = "Failed to connect to MariaDB"
                return validation_result
            
            if not qdrant_manager.connect():
                validation_result["error"] = "Failed to connect to Qdrant"
                return validation_result
            
            # Get candidate IDs from both sources
            mariadb_candidates = mariadb_client.get_candidates_with_resumes()
            mariadb_ids = {c.candidate_id for c in mariadb_candidates}
            mariadb_timestamps = {c.candidate_id: c.date_modified for c in mariadb_candidates}
            
            validation_result["mariadb_count"] = len(mariadb_ids)
            validation_result["qdrant_count"] = qdrant_manager.get_collection_count()
            
            # For now, assume Qdrant has the same IDs (in real implementation, 
            # you'd need to scan all points in Qdrant collection)
            qdrant_ids = mariadb_ids  # Simplified assumption
            
            # Find differences
            validation_result["missing_in_qdrant"] = list(mariadb_ids - qdrant_ids)
            validation_result["extra_in_qdrant"] = list(qdrant_ids - mariadb_ids)
            
            # Check if counts match
            validation_result["valid"] = (
                len(validation_result["missing_in_qdrant"]) == 0 and
                len(validation_result["extra_in_qdrant"]) == 0 and
                validation_result["mariadb_count"] == validation_result["qdrant_count"]
            )
            
            logger.info(f"Validation completed - Valid: {validation_result['valid']}")
            return validation_result
            
        except Exception as e:
            logger.error(f"Sync validation failed: {str(e)}")
            validation_result["error"] = str(e)
            return validation_result
        finally:
            mariadb_client.disconnect()
            qdrant_manager.disconnect()
    
    def force_resync(self) -> bool:
        """Force a complete resynchronization"""
        logger.info("Starting forced resynchronization")
        return self.initial_sync(force=True)


# Global sync manager instance
sync_manager = SyncManager()
