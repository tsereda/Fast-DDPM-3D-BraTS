#!/usr/bin/env python3
"""
Test script for distributed training setup
"""
import os
import sys
import torch

# Add the current directory to path
sys.path.append('.')

from distributed_utils import setup_distributed

class MockArgs:
    def __init__(self):
        self.distributed = True
        self.local_rank = -1
        self.rank = -1
        self.world_size = -1
        self.dist_backend = 'nccl'
        self.dist_url = 'env://'
        self.gpu = 0
        self.multi_gpu = False

def test_distributed_setup():
    """Test distributed setup"""
    print("=== Testing Distributed Setup ===")
    
    args = MockArgs()
    
    # Test 1: No environment variables (should fall back to single GPU)
    print("\n--- Test 1: No environment variables ---")
    rank, world_size, device = setup_distributed(args)
    print(f"Result: rank={rank}, world_size={world_size}, device={device}")
    
    # Test 2: Set environment variables
    print("\n--- Test 2: With environment variables ---")
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '2'
    os.environ['LOCAL_RANK'] = '0'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    args = MockArgs()  # Reset args
    
    try:
        rank, world_size, device = setup_distributed(args)
        print(f"Result: rank={rank}, world_size={world_size}, device={device}")
    except Exception as e:
        print(f"Expected error (no other processes): {e}")
    
    # Clean up
    for key in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT']:
        if key in os.environ:
            del os.environ[key]

if __name__ == '__main__':
    test_distributed_setup()
