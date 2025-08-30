#!/usr/bin/env python3
"""Test script to verify authentication fixes."""

import backend
from pathlib import Path

def test_path_validation():
    """Test path traversal protection."""
    print("Testing path validation...")
    
    # Test valid path
    try:
        valid_path = backend._validate_path("data/Filtered_Data/s01_ex01_s01.csv")
        print(f"✅ Valid path accepted: {valid_path}")
    except ValueError as e:
        print(f"❌ Valid path rejected: {e}")
    
    # Test path traversal attack
    try:
        malicious_path = backend._validate_path("../../../etc/passwd")
        print(f"❌ Malicious path accepted: {malicious_path}")
    except ValueError as e:
        print(f"✅ Malicious path blocked: {e}")

def test_user_operations():
    """Test user registration and deregistration with error handling."""
    print("\nTesting user operations...")
    
    # Test getting users
    users = backend.get_registered_users()
    print(f"Current users: {users}")
    
    # Test user info
    if users:
        user_info = backend.get_user_info(users[0])
        print(f"User info for {users[0]}: {user_info}")

def test_authentication_error_handling():
    """Test authentication with various error conditions."""
    print("\nTesting authentication error handling...")
    
    # Test with non-existent user
    result = backend.authenticate("nonexistent_user", "fake_file.csv")
    print(f"Non-existent user result: {result}")
    
    # Test with path traversal
    result = backend.authenticate("alice", "../../../etc/passwd")
    print(f"Path traversal result: {result}")

if __name__ == "__main__":
    print("🧠 Testing EEG Authentication Security Fixes\n")
    
    test_path_validation()
    test_user_operations()
    test_authentication_error_handling()
    
    print("\n✅ All security tests completed!")