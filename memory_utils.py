"""
Memory management utilities for 3D medical imaging training
"""
import torch
import torch.distributed as dist
import logging
import time
import gc


class MemoryManager:
    """Enhanced memory management for 3D diffusion training"""
    
    def __init__(self, device, aggressive_cleanup=True):
        self.device = device
        self.aggressive_cleanup = aggressive_cleanup
        self.peak_memory = 0
        self.cleanup_counter = 0
        self.memory_threshold_gb = 10.0  # Configurable threshold
        self.last_cleanup_time = time.time()
        
    def cleanup_gpu_memory(self, force=False):
        """Comprehensive GPU memory cleanup"""
        if not torch.cuda.is_available():
            return
            
        try:
            # Standard cleanup
            torch.cuda.empty_cache()
            
            if force or self.aggressive_cleanup:
                # More aggressive cleanup
                if hasattr(torch.cuda, 'ipc_collect'):
                    torch.cuda.ipc_collect()
                
                # Force garbage collection
                gc.collect()
                
                # Clear any lingering autograd history
                torch.cuda.synchronize()
                
            self.cleanup_counter += 1
            self.last_cleanup_time = time.time()
            
        except Exception as e:
            logging.warning(f"GPU memory cleanup failed: {e}")
    
    def get_memory_stats(self):
        """Get detailed memory statistics"""
        if not torch.cuda.is_available():
            return {}
            
        try:
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3   # GB
            max_allocated = torch.cuda.max_memory_allocated(self.device) / 1024**3  # GB
            
            self.peak_memory = max(self.peak_memory, allocated)
            
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'max_allocated_gb': max_allocated,
                'peak_session_gb': self.peak_memory,
                'cleanup_calls': self.cleanup_counter
            }
        except Exception as e:
            logging.warning(f"Memory stats failed: {e}")
            return {}
    
    def log_memory_usage(self, prefix=""):
        """Log current memory usage"""
        stats = self.get_memory_stats()
        if stats:
            logging.info(f"{prefix}GPU Memory - "
                        f"Allocated: {stats['allocated_gb']:.2f}GB, "
                        f"Reserved: {stats['reserved_gb']:.2f}GB, "
                        f"Peak: {stats['peak_session_gb']:.2f}GB")
    
    def check_memory_threshold(self, threshold_gb=None):
        """Check if memory usage exceeds threshold"""
        if threshold_gb is None:
            threshold_gb = self.memory_threshold_gb
            
        stats = self.get_memory_stats()
        if stats and stats['allocated_gb'] > threshold_gb:
            logging.warning(f"High memory usage: {stats['allocated_gb']:.2f}GB > {threshold_gb}GB")
            return True
        return False
    
    def adaptive_cleanup(self):
        """Perform adaptive memory cleanup based on usage and time"""
        current_time = time.time()
        time_since_cleanup = current_time - self.last_cleanup_time
        
        # Force cleanup if memory threshold exceeded or enough time passed
        if self.check_memory_threshold() or time_since_cleanup > 30:  # 30 seconds
            self.cleanup_gpu_memory(force=True)
            return True
        return False


class DistributedMemoryManager(MemoryManager):
    """Extended memory manager for distributed training"""
    
    def __init__(self, device, rank, world_size, aggressive_cleanup=True):
        super().__init__(device, aggressive_cleanup)
        self.rank = rank
        self.world_size = world_size
        
    def log_memory_usage(self, prefix=""):
        """Log memory usage with rank information"""
        from .distributed_utils import is_main_process
        if is_main_process(self.rank):
            stats = self.get_memory_stats()
            if stats:
                logging.info(f"{prefix}[Rank {self.rank}] Memory - "
                           f"Allocated: {stats['allocated_gb']:.2f}GB, "
                           f"Reserved: {stats['reserved_gb']:.2f}GB, "
                           f"Peak: {stats['peak_session_gb']:.2f}GB")
    
    def sync_cleanup_across_processes(self):
        """Synchronize memory cleanup across all processes"""
        if self.world_size > 1 and dist.is_initialized():
            # Synchronize before cleanup
            dist.barrier()
            self.cleanup_gpu_memory(force=True)
            dist.barrier()
        else:
            self.cleanup_gpu_memory(force=True)


def safe_batch_processing(batch, device, memory_manager, max_retries=3):
    """
    Safely process batch data with memory management and error recovery
    
    Args:
        batch: Input batch dictionary
        device: Target device
        memory_manager: MemoryManager instance
        max_retries: Maximum retry attempts for OOM recovery
        
    Returns:
        Processed batch dict or None if processing fails
    """
    for attempt in range(max_retries):
        try:
            # Check memory before processing
            if memory_manager.check_memory_threshold():
                memory_manager.cleanup_gpu_memory(force=True)
            
            # Process batch tensors
            processed_batch = {}
            
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    # Move to device with non_blocking for efficiency
                    processed_batch[key] = value.to(device, non_blocking=True)
                    
                    # Validate tensor after transfer
                    if torch.any(torch.isnan(processed_batch[key])) or torch.any(torch.isinf(processed_batch[key])):
                        logging.warning(f"NaN/Inf detected in batch tensor '{key}' - skipping batch")
                        return None
                        
                else:
                    processed_batch[key] = value
            
            # Final memory check after processing
            if memory_manager.check_memory_threshold(threshold_gb=12.0):  # Higher threshold after processing
                logging.warning("Memory usage high after batch processing")
                memory_manager.cleanup_gpu_memory()
            
            return processed_batch
            
        except torch.cuda.OutOfMemoryError as e:
            logging.error(f"OOM error in batch processing (attempt {attempt + 1}/{max_retries}): {e}")
            
            # Aggressive cleanup on OOM
            memory_manager.cleanup_gpu_memory(force=True)
            
            if attempt == max_retries - 1:
                logging.error("Failed to process batch after maximum retries")
                return None
                
            # Wait a bit before retry
            time.sleep(1)
            
        except Exception as e:
            logging.error(f"Error in batch processing: {e}")
            return None
    
    return None


def optimize_model_for_memory(model, enable_gradient_checkpointing=True):
    """
    Apply memory optimizations to the model
    
    Args:
        model: PyTorch model
        enable_gradient_checkpointing: Whether to enable gradient checkpointing
    """
    try:
        # Enable gradient checkpointing if supported
        if enable_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logging.info("Enabled gradient checkpointing")
        
        # Set model to use memory efficient attention if available
        if hasattr(model, 'set_attention_slice'):
            model.set_attention_slice("auto")
            logging.info("Enabled memory efficient attention")
            
        # Enable memory efficient cross attention
        if hasattr(model, 'enable_xformers_memory_efficient_attention'):
            try:
                model.enable_xformers_memory_efficient_attention()
                logging.info("Enabled xformers memory efficient attention")
            except Exception as e:
                logging.warning(f"Could not enable xformers attention: {e}")
                
    except Exception as e:
        logging.warning(f"Model memory optimization failed: {e}")


def dynamic_batch_size_adjustment(current_batch_size, memory_stats, target_memory_gb=8.0):
    """
    Dynamically adjust batch size based on memory usage
    
    Args:
        current_batch_size: Current batch size
        memory_stats: Memory statistics dict
        target_memory_gb: Target memory usage in GB
        
    Returns:
        Adjusted batch size
    """
    if not memory_stats:
        return current_batch_size
    
    current_memory = memory_stats.get('allocated_gb', 0)
    
    if current_memory > target_memory_gb * 1.2:  # 20% over target
        # Reduce batch size
        new_batch_size = max(1, current_batch_size // 2)
        logging.info(f"Reducing batch size from {current_batch_size} to {new_batch_size} due to high memory usage")
        return new_batch_size
    elif current_memory < target_memory_gb * 0.6:  # 40% under target
        # Increase batch size
        new_batch_size = min(current_batch_size * 2, 8)  # Cap at reasonable size
        logging.info(f"Increasing batch size from {current_batch_size} to {new_batch_size}")
        return new_batch_size
    
    return current_batch_size