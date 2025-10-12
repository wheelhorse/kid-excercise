"""
Main search service for resume retrieval system
"""
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from utils.logger import Logger, log_performance, log_method
from core.qdrant_client import qdrant_manager
from core.sync_manager import sync_manager
from database.mariadb_client import mariadb_client, CandidateRecord
from config import SEARCH_CONFIG

logger = Logger.get_logger("hybrid_search.service")


class SearchService:
    """Main search service for resume retrieval"""
    
    def __init__(self):
        """Initialize search service"""
        self.search_config = SEARCH_CONFIG
        self.is_initialized = False
        
        logger.info("Search service initialized")
    
    @log_method("hybrid_search.service")
    def initialize(self, force_sync: bool = False) -> bool:
        """Initialize the search service"""
        logger.info("Initializing search service")
        
        try:
            # Initialize sync manager and perform initial sync if needed
            if force_sync or not self._is_data_synced():
                logger.info("Performing initial data synchronization")
                if not sync_manager.initial_sync(force=force_sync):
                    logger.error("Failed to sync data")
                    return False
            
            self.is_initialized = True
            logger.info("Search service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize search service: {str(e)}")
            return False
    
    def _is_data_synced(self) -> bool:
        """Check if data is already synchronized"""
        try:
            status = sync_manager.get_sync_status()
            return (
                status.get("mariadb_connected", False) and
                status.get("qdrant_connected", False) and
                status.get("qdrant_count", 0) > 0 and
                not status.get("sync_needed", True)
            )
        except Exception:
            return False
    
    @log_performance
    def search_candidates(
        self,
        job_description: str,
        additional_requirements: str = "",
        limit: int = 100,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        include_mariadb_details: bool = True
    ) -> Dict[str, Any]:
        """
        Search for top candidates based on job description and requirements
        
        Args:
            job_description: Main job description text
            additional_requirements: Additional search criteria
            limit: Number of results to return (max 100)
            dense_weight: Weight for dense (semantic) search
            sparse_weight: Weight for sparse (keyword) search
            include_mariadb_details: Whether to fetch full details from MariaDB
            
        Returns:
            Dict containing search results and metadata
        """
        if not self.is_initialized:
            logger.error("Search service not initialized")
            return {"error": "Search service not initialized", "results": []}
        
        # Limit to max 100 results
        limit = min(limit, 100)
        
        # Combine job description and additional requirements
        full_query = f"{job_description}"
        if additional_requirements.strip():
            full_query += f" {additional_requirements}"
        
        logger.info(f"Searching for candidates with query length: {len(full_query)} chars, limit: {limit}")
        
        try:
            # Connect to Qdrant
            if not qdrant_manager.connect():
                logger.error("Failed to connect to Qdrant")
                return {"error": "Failed to connect to search database", "results": []}
            
            # Perform hybrid search
            logger.info(f"Performing hybrid search with query: '{full_query[:100]}...' (truncated)")
            logger.info(f"Search parameters: limit={limit}, dense_weight={dense_weight}, sparse_weight={sparse_weight}")
            
            search_results = qdrant_manager.hybrid_search(
                query=full_query,
                limit=limit,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight
            )
            
            # Debug: Log raw search results with candidate IDs and names
            if search_results:
                logger.info(f"Raw hybrid search returned {len(search_results)} results:")
                for i, result in enumerate(search_results[:10]):  # Log first 10 results
                    candidate_id = result.get("candidate_id", "unknown")
                    payload = result.get("payload", {})
                    first_name = payload.get("first_name", "")
                    last_name = payload.get("last_name", "")
                    name = f"{first_name} {last_name}".strip() or "No name"
                    rrf_score = result.get("rrf_score", 0.0)
                    dense_score = result.get("dense_score", 0.0)
                    sparse_score = result.get("sparse_score", 0.0)
                    
                    logger.info(f"  [{i+1}] ID:{candidate_id} Name:'{name}' RRF:{rrf_score:.4f} Dense:{dense_score:.4f} Sparse:{sparse_score:.4f}")
                
                if len(search_results) > 10:
                    logger.info(f"  ... and {len(search_results) - 10} more results")
                    
                # Special check for target candidates
                target_names = ["徐佳芸", "赵浩海"]
                found_targets = []
                for result in search_results:
                    payload = result.get("payload", {})
                    first_name = payload.get("first_name", "")
                    last_name = payload.get("last_name", "")
                    full_name = f"{first_name}{last_name}".strip()
                    
                    if any(target in full_name for target in target_names):
                        found_targets.append({
                            "name": full_name,
                            "candidate_id": result.get("candidate_id"),
                            "rank": search_results.index(result) + 1,
                            "rrf_score": result.get("rrf_score", 0.0)
                        })
                
                if found_targets:
                    logger.info(f"Target candidates found in results: {found_targets}")
                else:
                    logger.warning(f"Target candidates {target_names} NOT found in search results!")
            else:
                logger.warning("Hybrid search returned no results")
            
            if not search_results:
                logger.info("No candidates found matching the criteria")
                return {
                    "results": [],
                    "total_found": 0,
                    "query_info": {
                        "job_description": job_description,
                        "additional_requirements": additional_requirements,
                        "dense_weight": dense_weight,
                        "sparse_weight": sparse_weight
                    },
                    "search_time": datetime.now().isoformat()
                }
            
            # Enrich results with MariaDB details if requested
            if include_mariadb_details:
                enriched_results = self._enrich_with_mariadb_details(search_results)
            else:
                enriched_results = search_results
            
            # Prepare final results
            final_results = []
            for i, result in enumerate(enriched_results):
                candidate_info = {
                    "rank": result.get("final_rank", i + 1),
                    "candidate_id": result["candidate_id"],
                    "rrf_score": result.get("rrf_score", 0.0),
                    "dense_score": result.get("dense_score"),
                    "sparse_score": result.get("sparse_score"),
                    "dense_rank": result.get("dense_rank"),
                    "sparse_rank": result.get("sparse_rank")
                }
                
                # Add candidate details from payload or MariaDB
                payload = result.get("payload", {})
                if "mariadb_details" in result:
                    mariadb_details = result["mariadb_details"]
                    candidate_info.update({
                        "first_name": mariadb_details.first_name,
                        "last_name": mariadb_details.last_name,
                        "email": mariadb_details.email1,
                        "key_skills": mariadb_details.key_skills,
                        "notes": mariadb_details.notes,
                        "date_modified": mariadb_details.date_modified.isoformat() if mariadb_details.date_modified and isinstance(mariadb_details.date_modified, datetime) else str(mariadb_details.date_modified) if mariadb_details.date_modified else None,
                        "resume_available": bool(mariadb_details.resume_text)
                    })
                else:
                    candidate_info.update({
                        "first_name": payload.get("first_name"),
                        "last_name": payload.get("last_name"),
                        "email": payload.get("email1"),
                        "key_skills": payload.get("key_skills"),
                        "notes": payload.get("notes"),
                        "date_modified": payload.get("date_modified"),
                        "resume_available": bool(payload.get("search_text"))
                    })
                
                final_results.append(candidate_info)
            
            # Extract candidate IDs for summary
            candidate_ids = [result["candidate_id"] for result in final_results]
            
            search_response = {
                "results": final_results,
                "total_found": len(final_results),
                "candidate_ids": candidate_ids,  # Added candidate ID summary
                "query_info": {
                    "job_description": job_description,
                    "additional_requirements": additional_requirements,
                    "dense_weight": dense_weight,
                    "sparse_weight": sparse_weight,
                    "limit_requested": limit
                },
                "search_time": datetime.now().isoformat()
            }
            
            logger.info(f"Search completed - found {len(final_results)} candidates with IDs: {candidate_ids}")
            return search_response
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return {"error": str(e), "results": []}
        finally:
            qdrant_manager.disconnect()
    
    def _enrich_with_mariadb_details(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich search results with full MariaDB details"""
        try:
            if not mariadb_client.connect():
                logger.warning("Failed to connect to MariaDB for result enrichment")
                return search_results
            
            enriched_results = []
            for result in search_results:
                candidate_id = result["candidate_id"]
                
                # Get full candidate details from MariaDB
                candidate_details = mariadb_client.get_candidate_by_id(candidate_id)
                
                if candidate_details:
                    result["mariadb_details"] = candidate_details
                
                enriched_results.append(result)
            
            logger.debug(f"Enriched {len(enriched_results)} results with MariaDB details")
            return enriched_results
            
        except Exception as e:
            logger.error(f"Failed to enrich results with MariaDB details: {str(e)}")
            return search_results
        finally:
            mariadb_client.disconnect()
    
    @log_method("hybrid_search.service")
    def get_candidate_details(self, candidate_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific candidate"""
        try:
            if not mariadb_client.connect():
                logger.error("Failed to connect to MariaDB")
                return None
            
            candidate = mariadb_client.get_candidate_by_id(candidate_id)
            if not candidate:
                logger.warning(f"Candidate {candidate_id} not found")
                return None
            
            return {
                "candidate_id": candidate.candidate_id,
                "first_name": candidate.first_name,
                "last_name": candidate.last_name,
                "email": candidate.email1,
                "key_skills": candidate.key_skills,
                "notes": candidate.notes,
                "date_modified": candidate.date_modified.isoformat() if candidate.date_modified and isinstance(candidate.date_modified, datetime) else str(candidate.date_modified) if candidate.date_modified else None,
                "resume_text": candidate.resume_text,
                "attachment_id": candidate.attachment_id
            }
            
        except Exception as e:
            logger.error(f"Failed to get candidate details: {str(e)}")
            return None
        finally:
            mariadb_client.disconnect()
    
    @log_method("hybrid_search.service")
    def get_service_status(self) -> Dict[str, Any]:
        """Get current service status"""
        try:
            sync_status = sync_manager.get_sync_status()
            
            return {
                "service_initialized": self.is_initialized,
                "sync_status": sync_status,
                "search_config": self.search_config
            }
            
        except Exception as e:
            logger.error(f"Failed to get service status: {str(e)}")
            return {"error": str(e)}
    
    @log_method("hybrid_search.service")
    def trigger_sync(self, force: bool = False) -> bool:
        """Manually trigger data synchronization"""
        try:
            if force:
                return sync_manager.force_resync()
            else:
                return sync_manager.incremental_sync()
        except Exception as e:
            logger.error(f"Failed to trigger sync: {str(e)}")
            return False


# Global search service instance
search_service = SearchService()
