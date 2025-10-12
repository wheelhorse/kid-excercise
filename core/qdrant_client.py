"""
Qdrant client for hybrid search with BGE-M3 and BM25
"""
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, SparseVector, SparseVectorParams,
    CollectionInfo, SearchRequest, QueryRequest, Filter,
    FieldCondition, MatchValue, SearchParams, NamedSparseVector
)
from qdrant_client.http.exceptions import UnexpectedResponse

from utils.logger import Logger, log_performance, log_method
from config import QDRANT_CONFIG, COLLECTION_NAME, BATCH_SIZE
from models.embeddings_smart import get_smart_hybrid_model
from database.mariadb_client import CandidateRecord

logger = Logger.get_logger("hybrid_search.qdrant")


class QdrantManager:
    """Qdrant collection manager for hybrid search"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Qdrant manager"""
        self.config = config or QDRANT_CONFIG
        self.collection_name = COLLECTION_NAME
        self.client = None
        self.is_connected = False
        
        logger.info(f"Qdrant manager initialized for collection: {self.collection_name}")
    
    @log_method("hybrid_search.qdrant")
    def connect(self) -> bool:
        """Connect to Qdrant"""
        try:
            self.client = QdrantClient(
                url=self.config['url'],
                api_key=self.config.get('api_key'),
                timeout=self.config.get('timeout', 60.0),
                prefer_grpc=self.config.get('prefer_grpc', True)
            )
            
            # Test connection
            collections = self.client.get_collections()
            self.is_connected = True
            
            logger.info(f"Connected to Qdrant at {self.config['url']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            return False
    
    @log_method("hybrid_search.qdrant")
    def create_collection(self, recreate: bool = False) -> bool:
        """Create collection with hybrid vector configuration"""
        if not self.is_connected:
            logger.error("Not connected to Qdrant")
            return False
        
        try:
            # Check if collection exists
            if recreate:
                try:
                    self.client.delete_collection(self.collection_name)
                    logger.info(f"Deleted existing collection: {self.collection_name}")
                except UnexpectedResponse:
                    pass
                except Exception:
                    pass
            
            # Check existence without exceptions
            try:
                # Check existence (no exceptions thrown)
                if self.client.collection_exists(self.collection_name):
                    logger.info(f"Collection {self.collection_name} already exists")
                    return True
                else:
                    logger.info(f"Collection {self.collection_name} does not exist, creating...")
                    pass

            except Exception as e:
                logger.error(f"Failed to check collection {self.collection_name}: {str(e)}")
                return False
            
            # Get embedding dimensions - use default values if model not fitted yet
            try:
                hybrid_model = get_smart_hybrid_model()
                dense_dim = hybrid_model.get_dense_dim()
            except Exception:
                dense_dim = 512  # BGE-M3 default dimension
            
            logger.info(f"Creating collection with dense_dim={dense_dim}")
            
            # Create collection with hybrid vectors
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=dense_dim,
                        distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams()
                }
            )
            
            logger.info(f"Created collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection: {str(e)}")
            return False
    
    @log_method("hybrid_search.qdrant")
    def get_collection_info(self) -> Optional[CollectionInfo]:
        """Get collection information"""
        if not self.is_connected:
            return None
        
        try:
            return self.client.get_collection(self.collection_name)
        except Exception as e:
            logger.error(f"Failed to get collection info: {str(e)}")
            return None
    
    @log_method("hybrid_search.qdrant")
    def collection_exists(self) -> bool:
        """Check if collection exists"""
        if not self.is_connected:
            return False
        
        try:
            self.client.get_collection(self.collection_name)
            return True
        except UnexpectedResponse:
            return False
        except Exception as e:
            logger.error(f"Error checking collection existence: {str(e)}")
            return False
    
    @log_performance
    def upsert_candidates(self, candidates: List[CandidateRecord]) -> bool:
        """Insert candidate records into Qdrant"""
        if not self.is_connected:
            logger.error("Not connected to Qdrant")
            return False
        
        if not candidates:
            logger.warning("No candidates to upsert")
            return True
        
        # Check if collection exists before upserting
        if not self.collection_exists():
            logger.error(f"Collection {self.collection_name} doesn't exist. Cannot upsert candidates.")
            return False
        
        logger.info(f"Upserting {len(candidates)} candidates to {self.collection_name}")

        try:
            # Prepare texts for embedding
            texts = []
            for candidate in candidates:
                search_text = self._create_search_text(candidate)
                texts.append(search_text)
            
            # Get embeddings - use lazy factory function
            hybrid_model = get_smart_hybrid_model()
            if not hybrid_model.is_fitted:
                logger.info("Fitting smart hybrid model on candidate texts")
                hybrid_model.fit(texts)
            
            embeddings = hybrid_model.encode(texts)
            dense_embeddings = embeddings["dense"]
            sparse_embeddings = embeddings["sparse"]
            
            # Create points for batch upsert
            points = []
            for i, candidate in enumerate(candidates):
                # Prepare sparse vector
                sparse_data = sparse_embeddings[i]
                sparse_vector = SparseVector(
                    indices=sparse_data["indices"],
                    values=sparse_data["values"]
                )
                
                # Safely format date_modified
                date_modified = None
                if candidate.date_modified:
                    if isinstance(candidate.date_modified, datetime):
                        date_modified = candidate.date_modified.isoformat()
                    else:
                        date_modified = str(candidate.date_modified)
                
                # Create point - convert integer ID to string to avoid UUID parsing error
                point = PointStruct(
                    id=id(str(candidate.candidate_id)),
                    vector={
                        "dense": dense_embeddings[i].tolist(),
                        "sparse": sparse_vector
                    },
                    payload={
                        "candidate_id": candidate.candidate_id,
                        "first_name": candidate.first_name,
                        "last_name": candidate.last_name,
                        "email1": candidate.email1,
                        "key_skills": candidate.key_skills,
                        "notes": candidate.notes,
                        "date_modified": date_modified,
                        "search_text": texts[i][:500],  # Truncated for payload
                    }
                )
                points.append(point)
            
            # Batch upsert
            batch_size = BATCH_SIZE
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=True
                )
                logger.debug(f"Upserted batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")
            
            logger.info(f"Successfully upserted {len(candidates)} candidates with {len(points)} Points")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert candidates: {str(e)}")
            return False
    
    def _create_search_text(self, candidate: CandidateRecord) -> str:
        """Create searchable text from candidate record"""
        parts = []
        
        if candidate.first_name:
            parts.append(candidate.first_name)
        if candidate.last_name:
            parts.append(candidate.last_name)
            parts.append(candidate.last_name+candidate.first_name)
        if candidate.key_skills:
            parts.append(candidate.key_skills)
        if candidate.notes:
            parts.append(candidate.notes)
        if candidate.resume_text:
            parts.append(candidate.resume_text)
        
        return " ".join(parts)
    
    @log_performance
    def hybrid_search(
        self, 
        query: str, 
        limit: int = 100, 
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining dense and sparse vectors"""
        if not self.is_connected:
            logger.error("Not connected to Qdrant")
            return []
        
        logger.info(f"Performing hybrid search for query: '{query[:50]}...'")
        
        try:
            # Encode query - use lazy factory function
            hybrid_model = get_smart_hybrid_model()
            query_embeddings = hybrid_model.encode_query(query)
            dense_query = query_embeddings["dense"]
            sparse_query_data = query_embeddings["sparse"]
            
            sparse_query = SparseVector(
                indices=sparse_query_data["indices"],
                values=sparse_query_data["values"]
            )
            
            print(sparse_query)
            # Perform searches
            dense_results = self._dense_search(dense_query, limit)
            sparse_results = self._sparse_search(sparse_query, limit)
            
            # Combine results using RRF (Reciprocal Rank Fusion)
            combined_results = self._rrf_fusion(
                dense_results, sparse_results, 
                dense_weight, sparse_weight, limit
            )
            
            logger.info(f"Hybrid search returned {len(combined_results)} results")
            return combined_results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {str(e)}")
            return []
    
    def _dense_search(self, query_vector: np.ndarray, limit: int) -> List[Dict[str, Any]]:
        """Perform dense vector search"""
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=("dense", query_vector.tolist()),
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            search_results = []
            for i, result in enumerate(results):
                search_results.append({
                    "candidate_id": result.payload["candidate_id"],
                    "dense_score": float(result.score),
                    "dense_rank": i + 1,
                    "payload": result.payload
                })
            
            return search_results
            
        except Exception as e:
            logger.error(f"Dense search failed: {str(e)}")
            return []
    
    def _sparse_search(self, query_vector: SparseVector, limit: int) -> List[Dict[str, Any]]:
        """Perform sparse vector search"""
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=NamedSparseVector(  # Fixed: Use keyword args
                    name="sparse",
                    vector=query_vector
                ),
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            search_results = []
            for i, result in enumerate(results):
                search_results.append({
                    "candidate_id": result.payload["candidate_id"],
                    "sparse_score": float(result.score),
                    "sparse_rank": i + 1,
                    "payload": result.payload
                })
            
            return search_results
            
        except Exception as e:
            logger.error(f"Sparse search failed: {str(e)}")
            return []
    
    def _rrf_fusion(
        self, 
        dense_results: List[Dict[str, Any]], 
        sparse_results: List[Dict[str, Any]],
        dense_weight: float,
        sparse_weight: float,
        limit: int,
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion of dense and sparse results"""
        
        # Create candidate score map
        candidate_scores = {}
        
        # Add dense scores
        for result in dense_results:
            candidate_id = result["candidate_id"]
            rank = result["dense_rank"]
            rrf_score = dense_weight / (k + rank)
            
            candidate_scores[candidate_id] = {
                "candidate_id": candidate_id,
                "rrf_score": rrf_score,
                "dense_score": result.get("dense_score"),
                "dense_rank": rank,
                "payload": result["payload"]
            }
        
        # Add sparse scores
        for result in sparse_results:
            candidate_id = result["candidate_id"]
            rank = result["sparse_rank"]
            rrf_score = sparse_weight / (k + rank)
            
            if candidate_id in candidate_scores:
                candidate_scores[candidate_id]["rrf_score"] += rrf_score
                candidate_scores[candidate_id]["sparse_score"] = result.get("sparse_score")
                candidate_scores[candidate_id]["sparse_rank"] = rank
            else:
                candidate_scores[candidate_id] = {
                    "candidate_id": candidate_id,
                    "rrf_score": rrf_score,
                    "sparse_score": result.get("sparse_score"),
                    "sparse_rank": rank,
                    "payload": result["payload"]
                }
        
        # Sort by RRF score and return top results
        sorted_results = sorted(
            candidate_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )
        
        # Add final ranking
        for i, result in enumerate(sorted_results):
            result["final_rank"] = i + 1
        
        return sorted_results[:limit]
    
    @log_method("hybrid_search.qdrant")
    def delete_candidates(self, candidate_ids: List[int]) -> bool:
        """Delete candidates by IDs"""
        if not self.is_connected or not candidate_ids:
            return False
        
        try:
            # Convert integer IDs to string IDs to match upsert format
            string_ids = [str(cid) for cid in candidate_ids]
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=string_ids
            )
            logger.info(f"Deleted {len(candidate_ids)} candidates from {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete candidates: {str(e)}")
            return False
    
    @log_method("hybrid_search.qdrant")
    def get_collection_count(self) -> int:
        """Get total number of points in collection"""
        if not self.is_connected:
            return 0
        
        # Check existence without exceptions
        try:
            # Check existence (no exceptions thrown)
            if self.client.collection_exists(self.collection_name):
                logger.info(f"Collection {self.collection_name} exists, continue ...")
                pass
            else:
                logger.info(f"Collection {self.collection_name} does not exist, need to create ...")
                return 0
        except Exception as e:
            logger.error(f"Failed to delete candidates: {str(e)}")
            return 0 

        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count
        except UnexpectedResponse as e:
            if "doesn't exist" in str(e):
                logger.warning(f"Collection {self.collection_name} doesn't exist, returning count 0")
                return 0
            else:
                logger.error(f"Failed to get collection count: {str(e)}")
                return 0
        except Exception as e:
            logger.error(f"Failed to get collection count: {str(e)}")
            return 0
    
    @log_method("hybrid_search.qdrant")
    def find_candidates_by_name(self, target_names: List[str]) -> List[Dict[str, Any]]:
        """Find specific candidates by name in Qdrant database"""
        if not self.is_connected:
            logger.error("Not connected to Qdrant")
            return []
        
        if not self.collection_exists():
            logger.error(f"Collection {self.collection_name} doesn't exist")
            return []
        
        try:
            logger.info(f"Searching Qdrant for candidates with names: {target_names}")
            
            # Get all candidates from Qdrant (scroll through all points)
            found_candidates = []
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Large limit to get all candidates
                with_payload=True,
                with_vectors=False
            )
            
            points = scroll_result[0]  # Get points from scroll result
            total_points = len(points)
            logger.info(f"Retrieved {total_points} candidates from Qdrant")
            
            # Search for target names in the points
            for point in points:
                payload = point.payload
                first_name = payload.get("first_name", "") or ""
                last_name = payload.get("last_name", "") or ""
                full_name = f"{first_name}{last_name}".strip()
                
                # Check if any target name matches this candidate
                for target_name in target_names:
                    if target_name in full_name or full_name in target_name:
                        found_candidates.append({
                            "point_id": str(point.id),
                            "candidate_id": payload.get("candidate_id"),
                            "full_name": full_name,
                            "first_name": first_name,
                            "last_name": last_name,
                            "email": payload.get("email1"),
                            "key_skills": payload.get("key_skills"),
                            "search_text_snippet": payload.get("search_text", "")[:200],
                            "target_matched": target_name
                        })
                        logger.info(f"Found candidate in Qdrant: {full_name} (ID: {payload.get('candidate_id')}) matches '{target_name}'")
                        break
            
            if not found_candidates:
                logger.warning(f"No candidates found in Qdrant matching names: {target_names}")
                # Show sample of candidates for reference
                logger.info("Sample candidates in Qdrant:")
                for i, point in enumerate(points[:5]):
                    payload = point.payload
                    first_name = payload.get("first_name", "") or ""
                    last_name = payload.get("last_name", "") or ""
                    full_name = f"{first_name}{last_name}".strip()
                    logger.info(f"  [{i+1}] ID:{payload.get('candidate_id')} Name:'{full_name}'")
            
            return found_candidates
            
        except Exception as e:
            logger.error(f"Failed to find candidates by name: {str(e)}")
            return []
    
    @log_method("hybrid_search.qdrant")
    def get_candidate_by_id(self, candidate_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific candidate from Qdrant by candidate_id"""
        if not self.is_connected:
            logger.error("Not connected to Qdrant")
            return None
        
        if not self.collection_exists():
            logger.error(f"Collection {self.collection_name} doesn't exist")
            return None
        
        try:
            # Search for candidate with specific candidate_id
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="candidate_id",
                            match=MatchValue(value=candidate_id)
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False
            )
            
            points = scroll_result[0]
            if points:
                point = points[0]
                payload = point.payload
                logger.info(f"Found candidate {candidate_id} in Qdrant: {payload.get('first_name')}{payload.get('last_name')}")
                return {
                    "point_id": str(point.id),
                    "candidate_id": payload.get("candidate_id"),
                    "full_name": f"{payload.get('first_name', '')}{payload.get('last_name', '')}".strip(),
                    "first_name": payload.get("first_name"),
                    "last_name": payload.get("last_name"),
                    "email": payload.get("email1"),
                    "key_skills": payload.get("key_skills"),
                    "search_text": payload.get("search_text")
                }
            else:
                logger.warning(f"Candidate {candidate_id} not found in Qdrant")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get candidate by ID: {str(e)}")
            return None
    
    def disconnect(self):
        """Disconnect from Qdrant"""
        if self.client:
            self.client.close()
        self.is_connected = False
        logger.info("Disconnected from Qdrant")
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()


# Global instance
qdrant_manager = QdrantManager()
