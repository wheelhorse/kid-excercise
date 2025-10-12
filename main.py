#!/usr/bin/env python3
"""
Terminal-based testing interface for Resume Retrieval System
"""
import sys
import json
import argparse
from typing import Dict, Any, List
from datetime import datetime

from core.search_service import search_service
from core.sync_manager import sync_manager
from database.mariadb_client import mariadb_client
from utils.logger import Logger

logger = Logger.get_logger("hybrid_search.main")


class TerminalInterface:
    """Terminal interface for testing the resume retrieval system"""
    
    def __init__(self):
        """Initialize terminal interface"""
        self.search_service = search_service
        self.sync_manager = sync_manager
        
    def print_banner(self):
        """Print application banner"""
        print("=" * 70)
        print("    Resume Retrieval System - Terminal Interface")
        print("    Technology: Qdrant + BAAI/bge-m3 + jieba/BM25")
        print("=" * 70)
        print()
    
    def print_menu(self):
        """Print main menu"""
        print("\n" + "=" * 50)
        print("Main Menu:")
        print("1. Initialize System")
        print("2. Search Candidates")
        print("3. Get Candidate Details")
        print("4. Sync Status")
        print("5. Force Sync")
        print("6. Test Search Examples")
        print("7. System Status")
        print("8. Exit")
        print("=" * 50)
    
    def initialize_system(self, force: bool = False):
        """Initialize the search system"""
        print(f"\n{'Initializing system (force=' + str(force) + ')...'}")
        print("-" * 50)
        
        success = self.search_service.initialize(force_sync=force)
        
        if success:
            print("✅ System initialized successfully!")
            
            # Show status
            status = self.search_service.get_service_status()
            sync_status = status.get("sync_status", {})
            
            print(f"📊 MariaDB candidates: {sync_status.get('mariadb_count', 'N/A')}")
            print(f"📊 Qdrant points: {sync_status.get('qdrant_count', 'N/A')}")
            print(f"🕒 Last sync: {sync_status.get('last_sync_time', 'Never')}")
        else:
            print("❌ System initialization failed!")
            print("Please check the logs for details.")
    
    def search_candidates(self):
        """Interactive candidate search"""
        print("\n🔍 Candidate Search")
        print("-" * 50)
        
        if not self.search_service.is_initialized:
            print("❌ System not initialized. Please initialize first.")
            return
        
        # Get job description
        print("Enter job description:")
        job_description = input("> ").strip()
        
        if not job_description:
            print("❌ Job description cannot be empty!")
            return
        
        # Get additional requirements (optional)
        print("\nEnter additional requirements (optional):")
        additional_requirements = input("> ").strip()
        
        # Get search parameters
        try:
            print(f"\nNumber of results (1-100, default: 10):")
            limit_input = input("> ").strip()
            limit = int(limit_input) if limit_input else 10
            limit = max(1, min(100, limit))
            
            print(f"\nDense weight (0.0-1.0, default: 0.7):")
            dense_weight_input = input("> ").strip()
            dense_weight = float(dense_weight_input) if dense_weight_input else 0.7
            dense_weight = max(0.0, min(1.0, dense_weight))
            
            sparse_weight = 1.0 - dense_weight
            
        except ValueError:
            print("❌ Invalid input! Using default values.")
            limit = 10
            dense_weight = 0.7
            sparse_weight = 0.3
        
        # Perform search
        print(f"\n🔄 Searching for top {limit} candidates...")
        print(f"   Dense weight: {dense_weight:.2f}, Sparse weight: {sparse_weight:.2f}")
        
        start_time = datetime.now()
        results = self.search_service.search_candidates(
            job_description=job_description,
            additional_requirements=additional_requirements,
            limit=limit,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight
        )
        end_time = datetime.now()
        
        search_time = (end_time - start_time).total_seconds()
        
        if "error" in results:
            print(f"❌ Search failed: {results['error']}")
            return
        
        # Display results
        candidates = results.get("results", [])
        total_found = results.get("total_found", 0)
        
        print(f"\n📋 Search Results ({total_found} found in {search_time:.2f}s)")
        print("=" * 100)
        
        if not candidates:
            print("No candidates found matching your criteria.")
            return
        
        # Results table header
        print(f"{'Rank':<4} {'ID':<4} {'Name':<20} {'RRF Score':<10} {'Dense':<8} {'Sparse':<8} {'Skills':<25}")
        print("-" * 100)
        
        # Display each result
        for candidate in candidates:
            rank = candidate.get("rank", "N/A")
            candidate_id = candidate.get("candidate_id", "N/A")
            first_name = candidate.get("first_name", "")
            last_name = candidate.get("last_name", "")
            name = f"{first_name} {last_name}".strip()[:19]
            rrf_score = candidate.get("rrf_score", 0.0)
            dense_rank = candidate.get("dense_rank", "N/A")
            sparse_rank = candidate.get("sparse_rank", "N/A")
            skills = candidate.get("key_skills", "")[:24]
            
            print(f"{rank:<4} {candidate_id:<4} {name:<20} {rrf_score:<10.4f} {dense_rank:<8} {sparse_rank:<8} {skills:<25}")
        
        # Ask if user wants details
        print(f"\nEnter candidate rank for details (1-{len(candidates)}) or press Enter to continue:")
        detail_input = input("> ").strip()
        
        if detail_input.isdigit():
            rank = int(detail_input)
            if 1 <= rank <= len(candidates):
                # Get the actual candidate_id from the selected candidate
                selected_candidate = candidates[rank - 1]
                candidate_id = selected_candidate.get('candidate_id')
                if candidate_id:
                    self.show_candidate_details(candidate_id)
                else:
                    print("❌ Candidate ID not found!")
            else:
                print(f"❌ Invalid rank! Please enter a number between 1 and {len(candidates)}.")
    
    def show_candidate_details(self, candidate_id: int):
        """Show detailed candidate information"""
        print(f"\n👤 Candidate {candidate_id} Details")
        print("-" * 50)
        
        details = self.search_service.get_candidate_details(candidate_id)
        
        if not details:
            print(f"❌ Candidate {candidate_id} not found!")
            return
        
        print(f"ID: {details.get('candidate_id')}")
        print(f"Name: {details.get('first_name', '')} {details.get('last_name', '')}")
        print(f"Email: {details.get('email', 'N/A')}")
        print(f"Key Skills: {details.get('key_skills', 'N/A')}")
        print(f"Notes: {details.get('notes', 'N/A')}")
        print(f"Last Modified: {details.get('date_modified', 'N/A')}")
        print(f"Resume Available: {'Yes' if details.get('resume_text') else 'No'}")
        
        if details.get('resume_text'):
            print(f"\nResume Preview (first 300 chars):")
            print("-" * 30)
            print(details['resume_text'][:300] + "..." if len(details['resume_text']) > 300 else details['resume_text'])
    
    def show_sync_status(self):
        """Show synchronization status"""
        print("\n🔄 Synchronization Status")
        print("-" * 50)
        
        status = self.sync_manager.get_sync_status()
        
        print(f"MariaDB Connected: {'✅' if status.get('mariadb_connected') else '❌'}")
        print(f"Qdrant Connected: {'✅' if status.get('qdrant_connected') else '❌'}")
        print(f"Last Sync Time: {status.get('last_sync_time', 'Never')}")
        print(f"Sync Running: {'✅' if status.get('sync_running') else '❌'}")
        
        if status.get('mariadb_count') is not None:
            print(f"MariaDB Count: {status.get('mariadb_count')}")
        if status.get('qdrant_count') is not None:
            print(f"Qdrant Count: {status.get('qdrant_count')}")
        if status.get('sync_needed') is not None:
            print(f"Sync Needed: {'Yes' if status.get('sync_needed') else 'No'}")
        
        if "error" in status:
            print(f"❌ Error: {status['error']}")
    
    def force_sync(self):
        """Force data synchronization"""
        print("\n🔄 Force Synchronization")
        print("-" * 50)
        
        print("This will recreate the Qdrant collection and sync all data.")
        confirm = input("Are you sure? (y/N): ").strip().lower()
        
        if confirm != 'y':
            print("Sync cancelled.")
            return
        
        print("🔄 Starting forced synchronization...")
        success = self.search_service.trigger_sync(force=True)
        
        if success:
            print("✅ Force sync completed successfully!")
        else:
            print("❌ Force sync failed!")
    
    def test_search_examples(self):
        """Test with predefined search examples"""
        print("\n🧪 Test Search Examples")
        print("-" * 50)
        
        if not self.search_service.is_initialized:
            print("❌ System not initialized. Please initialize first.")
            return
        
        examples = [
            {
                "name": "Python Developer",
                "job_description": "Python developer with Django experience and machine learning background",
                "additional_requirements": "5+ years experience, web development"
            },
            {
                "name": "Frontend Developer",
                "job_description": "Frontend developer with React and Vue.js experience",
                "additional_requirements": "TypeScript, responsive design"
            },
            {
                "name": "DevOps Engineer",
                "job_description": "DevOps engineer with Kubernetes and cloud experience",
                "additional_requirements": "AWS, Docker, CI/CD"
            },
            {
                "name": "Data Scientist",
                "job_description": "Data scientist with machine learning and statistical analysis experience",
                "additional_requirements": "Python, R, big data"
            }
        ]
        
        print("Available test examples:")
        for i, example in enumerate(examples, 1):
            print(f"{i}. {example['name']}")
        
        try:
            choice = int(input("\nSelect example (1-4): ").strip())
            if 1 <= choice <= len(examples):
                example = examples[choice - 1]
                
                print(f"\n🔍 Testing: {example['name']}")
                print(f"Job Description: {example['job_description']}")
                print(f"Additional Requirements: {example['additional_requirements']}")
                
                results = self.search_service.search_candidates(
                    job_description=example['job_description'],
                    additional_requirements=example['additional_requirements'],
                    limit=5
                )
                
                if "error" in results:
                    print(f"❌ Search failed: {results['error']}")
                else:
                    candidates = results.get("results", [])
                    print(f"\n📋 Found {len(candidates)} candidates:")
                    
                    for candidate in candidates[:5]:
                        candidate_id = candidate.get('candidate_id', 'N/A')
                        name = f"{candidate.get('first_name', '')} {candidate.get('last_name', '')}".strip()
                        score = candidate.get('rrf_score', 0.0)
                        skills = candidate.get('key_skills', '')[:40]
                        print(f"  {candidate.get('rank', 'N/A')}. ID:{candidate_id} {name} (Score: {score:.4f}) - {skills}")
            else:
                print("❌ Invalid choice!")
        except ValueError:
            print("❌ Invalid input!")
    
    def show_system_status(self):
        """Show overall system status"""
        print("\n📊 System Status")
        print("-" * 50)
        
        status = self.search_service.get_service_status()
        
        print(f"Service Initialized: {'✅' if status.get('service_initialized') else '❌'}")
        
        sync_status = status.get("sync_status", {})
        print(f"MariaDB Connected: {'✅' if sync_status.get('mariadb_connected') else '❌'}")
        print(f"Qdrant Connected: {'✅' if sync_status.get('qdrant_connected') else '❌'}")
        
        if sync_status.get('mariadb_count') is not None and sync_status.get('qdrant_count') is not None:
            mariadb_count = sync_status['mariadb_count']
            qdrant_count = sync_status['qdrant_count']
            print(f"Data Synchronized: {'✅' if mariadb_count == qdrant_count else '❌'}")
            print(f"  MariaDB: {mariadb_count} candidates")
            print(f"  Qdrant: {qdrant_count} points")
        
        search_config = status.get("search_config", {})
        if search_config:
            print(f"Search Config: {json.dumps(search_config, indent=2)}")
    
    def run_interactive(self):
        """Run interactive terminal interface"""
        self.print_banner()
        
        while True:
            self.print_menu()
            
            try:
                choice = input("\nEnter your choice (1-8): ").strip()
                
                if choice == "1":
                    force = input("Force initialization? (y/N): ").strip().lower() == 'y'
                    self.initialize_system(force=force)
                
                elif choice == "2":
                    self.search_candidates()
                
                elif choice == "3":
                    try:
                        candidate_id = int(input("Enter candidate ID: ").strip())
                        self.show_candidate_details(candidate_id)
                    except ValueError:
                        print("❌ Invalid candidate ID!")
                
                elif choice == "4":
                    self.show_sync_status()
                
                elif choice == "5":
                    self.force_sync()
                
                elif choice == "6":
                    self.test_search_examples()
                
                elif choice == "7":
                    self.show_system_status()
                
                elif choice == "8":
                    print("\n👋 Goodbye!")
                    break
                
                else:
                    print("❌ Invalid choice! Please enter 1-8.")
                
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                print(f"❌ Unexpected error: {str(e)}")
                input("\nPress Enter to continue...")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Resume Retrieval System Terminal Interface")
    parser.add_argument("--init", action="store_true", help="Initialize system and exit")
    parser.add_argument("--force-init", action="store_true", help="Force initialize system and exit")
    parser.add_argument("--status", action="store_true", help="Show system status and exit")
    parser.add_argument("--search", type=str, help="Search query and exit")
    parser.add_argument("--limit", type=int, default=10, help="Search result limit (default: 10)")
    
    args = parser.parse_args()
    
    interface = TerminalInterface()
    
    # Handle command line arguments
    if args.init or args.force_init:
        interface.print_banner()
        interface.initialize_system(force=args.force_init)
        return
    
    if args.status:
        interface.show_system_status()
        return
    
    if args.search:
        interface.print_banner()
        if not search_service.is_initialized:
            print("Initializing system...")
            if not search_service.initialize():
                print("❌ Failed to initialize system!")
                return
        
        results = search_service.search_candidates(
            job_description=args.search,
            limit=args.limit
        )
        
        if "error" in results:
            print(f"❌ Search failed: {results['error']}")
        else:
            candidates = results.get("results", [])
            candidate_ids = results.get("candidate_ids", [])
            print(f"\n📋 Found {len(candidates)} candidates:")
            print(f"📋 Candidate IDs: {candidate_ids}")
            for candidate in candidates:
                candidate_id = candidate.get('candidate_id', 'N/A')
                name = f"{candidate.get('last_name', '')} {candidate.get('first_name', '')}".strip()
                score = candidate.get('rrf_score', 0.0)
                skills = candidate.get('key_skills', '')
                sparse_score = candidate.get('sparse_score', '')
                dense_score = candidate.get('dense_score', '')
                print(f"  {candidate.get('rank', 'N/A')}. ID:{candidate_id} {name} (Score: {score:.4f}, sparse-{sparse_score}, dense-{dense_score}) - {skills}")
        return
    
    # Run interactive mode
    interface.run_interactive()


if __name__ == "__main__":
    main()
