#!/usr/bin/env python3
"""
Debug script for testing search functionality and finding specific candidates
"""
import sys
sys.path.append('.')

from core.search_service import search_service
from database.mariadb_client import mariadb_client
from core.qdrant_client import qdrant_manager
from utils.logger import Logger

logger = Logger.get_logger("hybrid_search.debug")

def test_database_candidates():
    """Check if target candidates exist in MariaDB"""
    print("=" * 60)
    print("Testing Database for Target Candidates")
    print("=" * 60)
    
    target_names = ["徐佳芸", "赵浩海"]
    
    try:
        if not mariadb_client.connect():
            print("❌ Failed to connect to MariaDB")
            return
        
        # Get all candidates to check for our targets
        candidates = mariadb_client.get_all_candidates_basic()
        print(f"Total candidates in database: {len(candidates)}")
        
        found_targets = []
        for candidate in candidates:
            full_name = f"{candidate.first_name or ''}{candidate.last_name or ''}".strip()
            if any(target in full_name for target in target_names):
                found_targets.append({
                    "id": candidate.candidate_id,
                    "name": full_name,
                    "first_name": candidate.first_name,
                    "last_name": candidate.last_name,
                    "has_resume": bool(candidate.resume_text),
                    "key_skills": candidate.key_skills
                })
        
        if found_targets:
            print(f"\n✅ Found {len(found_targets)} target candidates in MariaDB:")
            for target in found_targets:
                print(f"  ID: {target['id']}")
                print(f"  Name: '{target['name']}'")
                print(f"  First: '{target['first_name']}', Last: '{target['last_name']}'")
                print(f"  Has Resume: {target['has_resume']}")
                print(f"  Key Skills: {target['key_skills']}")
                print()
        else:
            print(f"❌ Target candidates {target_names} NOT found in MariaDB")
            
            # Show a sample of candidates for reference
            print("\nSample candidates in database:")
            for i, candidate in enumerate(candidates[:5]):
                full_name = f"{candidate.first_name or ''}{candidate.last_name or ''}".strip()
                print(f"  [{i+1}] ID:{candidate.candidate_id} Name:'{full_name}'")
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")
    finally:
        mariadb_client.disconnect()

def test_search_with_target_names():
    """Test search functionality with target candidate names"""
    print("=" * 60)
    print("Testing Search with Target Names")
    print("=" * 60)
    
    # Initialize search service
    if not search_service.initialize():
        print("❌ Failed to initialize search service")
        return
    
    test_queries = [
        "徐佳芸",
        "赵浩海", 
        "徐佳",
        "佳芸",
        "Python 开发",
        "工程师"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        print("-" * 40)
        
        try:
            results = search_service.search_candidates(
                job_description=query,
                limit=10,
                dense_weight=0.7,
                sparse_weight=0.3
            )
            
            if results.get("error"):
                print(f"❌ Search error: {results['error']}")
                continue
            
            candidates = results.get("results", [])
            print(f"Found {len(candidates)} candidates")
            
            # Show top 3 results
            for i, candidate in enumerate(candidates[:3]):
                name = f"{candidate.get('first_name', '')} {candidate.get('last_name', '')}".strip()
                print(f"  [{i+1}] ID:{candidate.get('candidate_id')} Name:'{name}' Score:{candidate.get('rrf_score', 0):.4f}")
            
            # Check if our targets are in the results
            target_names = ["徐佳芸", "赵浩海"]
            found_targets = []
            for candidate in candidates:
                full_name = f"{candidate.get('first_name', '')}{candidate.get('last_name', '')}".strip()
                if any(target in full_name for target in target_names):
                    found_targets.append({
                        "name": full_name,
                        "rank": candidates.index(candidate) + 1,
                        "score": candidate.get('rrf_score', 0)
                    })
            
            if found_targets:
                print(f"  ✅ Target candidates found: {found_targets}")
            else:
                print(f"  ❌ Target candidates not in top results")
                
        except Exception as e:
            print(f"❌ Search failed: {e}")

def test_qdrant_candidates():
    """Check if target candidates exist in Qdrant database"""
    print("=" * 60)
    print("Testing Qdrant Database for Target Candidates")
    print("=" * 60)
    
    target_names = ["徐佳芸", "赵浩海"]
    
    try:
        if not qdrant_manager.connect():
            print("❌ Failed to connect to Qdrant")
            return
        
        # Check if collection exists
        if not qdrant_manager.collection_exists():
            print("❌ Qdrant collection doesn't exist")
            return
        
        # Get collection count
        total_count = qdrant_manager.get_collection_count()
        print(f"Total candidates in Qdrant: {total_count}")
        
        # Search for target candidates in Qdrant
        found_targets = qdrant_manager.find_candidates_by_name(target_names)
        
        if found_targets:
            print(f"\n✅ Found {len(found_targets)} target candidates in Qdrant:")
            for target in found_targets:
                print(f"  Point ID: {target['point_id']}")
                print(f"  Candidate ID: {target['candidate_id']}")
                print(f"  Name: '{target['full_name']}'")
                print(f"  First: '{target['first_name']}', Last: '{target['last_name']}'")
                print(f"  Email: {target['email']}")
                print(f"  Key Skills: {target['key_skills']}")
                print(f"  Target Matched: '{target['target_matched']}'")
                print(f"  Search Text: {target['search_text_snippet']}...")
                print()
        else:
            print(f"❌ Target candidates {target_names} NOT found in Qdrant")
            print("This means they either:")
            print("  1. Don't exist in MariaDB")
            print("  2. Weren't synced to Qdrant")
            print("  3. Are in Qdrant but with different name format")
        
        # Check specific candidate IDs if we found them in MariaDB
        print(f"\n🔍 Cross-checking with MariaDB results...")
        
    except Exception as e:
        print(f"❌ Error checking Qdrant: {e}")
    finally:
        qdrant_manager.disconnect()

def test_service_status():
    """Test service status and configuration"""
    print("=" * 60)
    print("Testing Service Status")
    print("=" * 60)
    
    try:
        status = search_service.get_service_status()
        
        print(f"Service Initialized: {status.get('service_initialized', False)}")
        print(f"Sync Status: {status.get('sync_status', {})}")
        
        sync_status = status.get('sync_status', {})
        print(f"  MariaDB Connected: {sync_status.get('mariadb_connected', False)}")
        print(f"  Qdrant Connected: {sync_status.get('qdrant_connected', False)}")
        print(f"  Qdrant Count: {sync_status.get('qdrant_count', 0)}")
        print(f"  Sync Needed: {sync_status.get('sync_needed', True)}")
        
    except Exception as e:
        print(f"❌ Failed to get service status: {e}")

if __name__ == "__main__":
    test_service_status()
    test_database_candidates()
    test_qdrant_candidates()
    test_search_with_target_names()
    
    print("\n" + "=" * 60)
    print("🔍 Debug testing completed!")
    print("Check the logs above for detailed search debugging information.")
    print("=" * 60)
