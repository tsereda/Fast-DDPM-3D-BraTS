"""
Memory management utilities for 3D medical imaging training
"""
import torch
import logging
import time
import gc


class MemoryManager:
    """Memory management for 3D diffusion training"""
    
    def __init__(self, device, aggressive_cleanup=True):
        self.device = device
        self.aggressive_cleanup = aggressive_cleanup
        self.peak_memory = 0
        self.cleanup_counter = 0
        self.memory_threshold_gb = 10.0
        self.last_cleanup_time = time.time()
        
    def cleanup_gpu_memory(self, force=False):
        """Comprehensive GPU memory cleanup"""
        if not torch.cuda.is_available():
            return
            
        try:
            torch.cuda.empty_cache()
            
            if force or self.aggressive_cleanup:
                if hasattr(torch.cuda, 'ipc_collect'):
                    torch.cuda.ipc_collect()
                gc.collect()
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


def safe_batch_processing(batch, device, memory_manager, max_retries=3):
    """
    Safely process batch data with memory management and error recovery
    """
    for attempt in range(max_retries):
        try:
            if memory_manager.check_memory_threshold():
                memory_manager.cleanup_gpu_memory(force=True)
            
            processed_batch = {}
            
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    processed_batch[key] = value.to(device, non_blocking=True)
                    
                    if torch.any(torch.isnan(processed_batch[key])) or torch.any(torch.isinf(processed_batch[key])):
                        logging.warning(f"NaN/Inf detected in batch tensor '{key}' - skipping batch")
                        return None
                else:
                    processed_batch[key] = value
            
            if memory_manager.check_memory_threshold(threshold_gb=12.0):
                logging.warning("Memory usage high after batch processing")
                memory_manager.cleanup_gpu_memory()
            
            return processed_batch
            
        except torch.cuda.OutOfMemoryError as e:
            logging.error(f"OOM error in batch processing (attempt {attempt + 1}/{max_retries}): {e}")
            memory_manager.cleanup_gpu_memory(force=True)
            
            if attempt == max_retries - 1:
                logging.error("Failed to process batch after maximum retries")
                return None
            time.sleep(1)
            
        except Exception as e:
            logging.error(f"Error in batch processing: {e}")
            return None
    
    return None


def optimize_model_for_memory(model, enable_gradient_checkpointing=True):
    """Apply memory optimizations to the model"""
    try:
        if enable_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logging.info("Enabled gradient checkpointing")
        
        if hasattr(model, 'set_attention_slice'):
            model.set_attention_slice("auto")
            logging.info("Enabled memory efficient attention")
            
        if hasattr(model, 'enable_xformers_memory_efficient_attention'):
            try:
                model.enable_xformers_memory_efficient_attention()
                logging.info("Enabled xformers memory efficient attention")
            except Exception as e:
                logging.warning(f"Could not enable xformers attention: {e}")
                
    except Exception as e:
        logging.warning(f"Model memory optimization failed: {e}")


def dynamic_batch_size_adjustment(current_batch_size, memory_stats, target_memory_gb=8.0):
    """Dynamically adjust batch size based on memory usage"""
    if not memory_stats:
        return current_batch_size
    
    current_memory = memory_stats.get('allocated_gb', 0)
    
    if current_memory > target_memory_gb * 1.2:  # 20% over target
        new_batch_size = max(1, current_batch_size // 2)
        logging.info(f"Reducing batch size from {current_batch_size} to {new_batch_size} due to high memory usage")
        return new_batch_size
    elif current_memory < target_memory_gb * 0.6:  # 40% under target
        new_batch_size = min(current_batch_size * 2, 8)  # Cap at reasonable size
        logging.info(f"Increasing batch size from {current_batch_size} to {new_batch_size}")
        return new_batch_size
    
    return current_batch_size