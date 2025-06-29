"""
GPU memory detection and automatic volume size adjustment
"""
import torch


def get_gpu_memory_gb():
    """Get available GPU memory in GB"""
    if not torch.cuda.is_available():
        return 0
    
    try:
        # Get total GPU memory
        total_memory = torch.cuda.get_device_properties(0).total_memory
        return total_memory / (1024**3)  # Convert to GB
    except:
        return 0


def get_recommended_volume_size(gpu_memory_gb=None):
    """
    Get recommended volume size based on GPU memory
    
    Args:
        gpu_memory_gb: GPU memory in GB, auto-detected if None
        
    Returns:
        tuple: (height, width, depth) for volume size
    """
    if gpu_memory_gb is None:
        gpu_memory_gb = get_gpu_memory_gb()
    
    # Conservative estimates for 3D diffusion models with batch_size=1
    size_map = {
        8: (64, 64, 64),    # 8GB GPU
        11: (80, 80, 80),   # 11GB GPU (RTX 2080 Ti)
        16: (96, 96, 96),   # 16GB GPU (T4, V100 16GB)
        24: (112, 112, 112), # 24GB GPU (RTX 3090, RTX 4090)
        32: (128, 128, 128), # 32GB GPU (V100 32GB)
        40: (144, 144, 144), # 40GB GPU (A100)
        80: (160, 160, 160), # 80GB GPU (A100 80GB)
    }
    
    # Find the best fit
    for memory_threshold in sorted(size_map.keys()):
        if gpu_memory_gb <= memory_threshold:
            return size_map[memory_threshold]
    
    # If GPU has more than 80GB, use largest size
    return (160, 160, 160)


def check_memory_usage(model, volume_size, batch_size=1, verbose=True):
    """
    Check memory usage for a given model and volume size
    
    Args:
        model: PyTorch model
        volume_size: tuple of (H, W, D)
        batch_size: batch size to test
        verbose: whether to print memory info
        
    Returns:
        dict: memory usage statistics
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    device = torch.device('cuda')
    model = model.to(device)
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    try:
        # Create dummy input
        x = torch.randn(batch_size, 4, *volume_size).to(device)
        t = torch.randint(0, 1000, (batch_size,)).to(device)
        
        # Forward pass
        with torch.no_grad():
            output = model(x, t)
        
        # Get memory stats
        current_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        
        stats = {
            "current_memory_gb": current_memory,
            "peak_memory_gb": peak_memory,
            "volume_size": volume_size,
            "batch_size": batch_size,
            "success": True
        }
        
        if verbose:
            print(f"Memory usage for volume {volume_size}:")
            print(f"  Current: {current_memory:.2f} GB")
            print(f"  Peak: {peak_memory:.2f} GB")
            
            if peak_memory > 20:
                print("  ⚠️  High memory usage!")
            else:
                print("  ✓ Memory usage looks good")
        
        return stats
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            stats = {
                "error": f"Out of memory with volume size {volume_size}",
                "volume_size": volume_size,
                "batch_size": batch_size,
                "success": False
            }
            if verbose:
                print(f"❌ Out of memory with volume size {volume_size}")
            return stats
        else:
            raise e
    finally:
        # Cleanup
        torch.cuda.empty_cache()


def auto_adjust_volume_size(model, target_memory_gb=None):
    """
    Automatically find the largest volume size that fits in memory
    
    Args:
        model: PyTorch model
        target_memory_gb: target memory usage, defaults to 80% of available
        
    Returns:
        tuple: optimal volume size
    """
    if not torch.cuda.is_available():
        return (64, 64, 64)  # Default fallback
    
    gpu_memory = get_gpu_memory_gb()
    if target_memory_gb is None:
        target_memory_gb = gpu_memory * 0.8  # Use 80% of available memory
    
    # Test sizes from small to large
    test_sizes = [
        (64, 64, 64),
        (80, 80, 80), 
        (96, 96, 96),
        (112, 112, 112),
        (128, 128, 128),
        (144, 144, 144),
        (160, 160, 160)
    ]
    
    best_size = (64, 64, 64)  # Safe default
    
    for size in test_sizes:
        stats = check_memory_usage(model, size, verbose=False)
        
        if stats.get("success", False):
            if stats["peak_memory_gb"] <= target_memory_gb:
                best_size = size
            else:
                break  # This size is too big, stop testing
        else:
            break  # Out of memory, stop testing
    
    print(f"Recommended volume size: {best_size}")
    print(f"Target memory usage: {target_memory_gb:.1f} GB")
    
    return best_size
