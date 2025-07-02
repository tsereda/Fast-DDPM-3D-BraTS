# Multi-GPU Training Guide for Fast-DDPM 3D BraTS

This guide explains how to use multi-GPU training with the Fast-DDPM 3D BraTS project.

## Overview

The implementation supports two multi-GPU approaches:

1. **DataParallel (DP)**: Simple multi-GPU on single node, automatically distributes batch across GPUs
2. **DistributedDataParallel (DDP)**: More efficient, supports multi-node training, better scaling

## Quick Start

### Option 1: Using the Multi-GPU Launcher (Recommended)

```bash
# Automatic GPU detection with DataParallel
python scripts/train_multi_gpu.py \
    --data_root /path/to/brats/data \
    --use_wandb \
    --wandb_project "fast-ddpm-3d-brats-multigpu"

# Specific GPUs with DistributedDataParallel
python scripts/train_multi_gpu.py \
    --data_root /path/to/brats/data \
    --gpus "0,1,2,3" \
    --use_ddp \
    --use_wandb \
    --wandb_project "fast-ddpm-3d-brats-multigpu"
```

### Option 2: Direct Training Script

```bash
# DataParallel (single node, multiple GPUs)
python scripts/train_3d.py \
    --data_root /path/to/brats/data \
    --multi_gpu \
    --use_wandb

# DistributedDataParallel (manual setup)
torchrun --standalone --nproc_per_node=4 scripts/train_3d.py \
    --data_root /path/to/brats/data \
    --distributed \
    --use_wandb
```

## Configuration

### Batch Size Scaling

The effective batch size is calculated as:
```
effective_batch_size = batch_size * world_size * gradient_accumulation_steps
```

- `batch_size`: Per-GPU batch size (default: 1 for 3D volumes)
- `world_size`: Number of GPUs
- `gradient_accumulation_steps`: Gradient accumulation (default: 4)

**Example with 4 GPUs:**
- Per-GPU batch size: 1
- World size: 4
- Gradient accumulation: 4
- **Effective batch size: 1 × 4 × 4 = 16**

### Configuration File Updates

Update `configs/fast_ddpm_3d.yml`:

```yaml
training:
    batch_size: 1                  # Per-GPU batch size
    gradient_accumulation_steps: 4   # Adjust based on memory
    multi_gpu: true               # Enable multi-GPU
    distributed: false            # Use DP (false) or DDP (true)
    sync_batchnorm: true         # Sync batch norm across GPUs
```

## Memory Optimization

### Per-GPU Memory Usage
- **Single GPU**: ~8-12GB for 128³ volumes
- **Multi-GPU**: Same per-GPU, but total throughput increases

### Batch Size Guidelines
| Volume Size | GPUs | Per-GPU Batch | Grad Accum | Effective Batch |
|-------------|------|---------------|------------|-----------------|
| 64³         | 2    | 2             | 4          | 16              |
| 128³        | 2    | 1             | 4          | 8               |
| 128³        | 4    | 1             | 4          | 16              |
| 256³        | 4    | 1             | 8          | 32              |

## Advanced Usage

### SLURM Clusters

Use the provided SLURM script:

```bash
# Edit the script to set your data path
vim scripts/slurm_multi_gpu.sh

# Submit job
sbatch scripts/slurm_multi_gpu.sh
```

### Kubernetes/Nautilus

The pod configuration supports multi-GPU:

```yaml
resources:
  requests:
    nvidia.com/gpu: "2"  # Request 2 GPUs
  limits:
    nvidia.com/gpu: "2"
```

### Environment Variables

Key environment variables for distributed training:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Available GPUs
export MASTER_ADDR=localhost          # Master node address
export MASTER_PORT=12355              # Master port
export WORLD_SIZE=4                   # Total number of processes
export RANK=0                         # Process rank (0 for master)
export LOCAL_RANK=0                   # Local process rank
```

## Performance Expectations

### Scaling Efficiency
- **2 GPUs**: ~1.8x speedup
- **4 GPUs**: ~3.2x speedup  
- **8 GPUs**: ~6.0x speedup

### Memory Efficiency
- **DistributedDataParallel**: More memory efficient
- **DataParallel**: Simpler but less efficient

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce per-GPU batch size
   # Increase gradient accumulation
   # Use gradient checkpointing
   ```

2. **NCCL Initialization Failed**
   ```bash
   # Check network connectivity
   # Verify CUDA versions match
   # Use gloo backend for debugging
   ```

3. **Uneven GPU Utilization**
   ```bash
   # Use DistributedDataParallel instead of DataParallel
   # Check data loading bottlenecks
   ```

### Debug Mode

```bash
# Test with smaller dataset
python scripts/train_multi_gpu.py \
    --data_root /path/to/brats/data \
    --debug \
    --gpus "0,1"
```

### Monitoring

- **W&B Dashboard**: Monitor training metrics across GPUs
- **nvidia-smi**: Watch GPU utilization and memory
- **htop**: Monitor CPU and memory usage

## Examples

### Local Development (2 GPUs)
```bash
python scripts/train_multi_gpu.py \
    --data_root ./data/BraTS2023 \
    --gpus "0,1" \
    --debug \
    --use_wandb \
    --doc "dev_multigpu_test"
```

### Production Training (4 GPUs)
```bash
python scripts/train_multi_gpu.py \
    --data_root /datasets/BraTS2023 \
    --gpus auto \
    --use_ddp \
    --use_wandb \
    --wandb_project "fast-ddpm-3d-production" \
    --doc "production_4gpu_$(date +%Y%m%d)"
```

### High Memory Volumes (256³)
```bash
python scripts/train_multi_gpu.py \
    --data_root /datasets/BraTS2023 \
    --config configs/fast_ddpm_3d_large.yml \
    --gpus "0,1,2,3,4,5,6,7" \
    --use_ddp \
    --use_wandb
```

## Best Practices

1. **Use DistributedDataParallel** for better scaling
2. **Monitor GPU utilization** to ensure all GPUs are working
3. **Start with debug mode** to verify setup
4. **Use gradient checkpointing** for large volumes
5. **Sync batch normalization** across GPUs
6. **Save checkpoints only on main process** to avoid conflicts

## Performance Tips

1. **Pin memory** in data loaders
2. **Use multiple workers** for data loading
3. **Enable mixed precision** training
4. **Optimize data pipeline** to avoid GPU starvation
5. **Use appropriate backend** (NCCL for GPUs, Gloo for debugging)

For more details, see the implementation in `scripts/train_3d.py` and `scripts/train_multi_gpu.py`.
