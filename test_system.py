#!/usr/bin/env python3
"""
Quick test script for Resume Retrieval System
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from main import TerminalInterface

def quick_test():
    """Run a quick test of the system"""
    print("=" * 60)
    print("Quick Test - Resume Retrieval System")
    print("=" * 60)
    
    interface = TerminalInterface()
    
    # Test system status
    print("\n1. Testing System Status...")
    interface.show_system_status()
    
    # Test initialization (without force)
    print("\n2. Testing System Initialization...")
    interface.initialize_system(force=False)
    
    # Test sync status
    print("\n3. Testing Sync Status...")
    interface.show_sync_status()
    
    # Test search examples
    print("\n4. Testing Search Examples...")
    interface.test_search_examples()
    
    print("\n" + "=" * 60)
    print("Quick test completed!")
    print("Run 'python3 main.py' for interactive mode")
    print("=" * 60)

if __name__ == "__main__":
    quick_test()
