#!/bin/bash
#SBATCH --job-name=fast_ddpm_3d_multigpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4           # Number of GPUs per node
#SBATCH --gres=gpu:4                  # Request 4 GPUs
#SBATCH --cpus-per-task=8            # CPU cores per GPU
#SBATCH --mem=128G                    # Total memory
#SBATCH --time=24:00:00              # 24 hours
#SBATCH --partition=gpu              # GPU partition
#SBATCH --output=logs/train_multigpu_%j.out
#SBATCH --error=logs/train_multigpu_%j.err

# Create logs directory
mkdir -p logs

# Load modules (adjust based on your cluster)
# module load python/3.8
# module load cuda/11.7

# Activate virtual environment if needed
# source venv/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=4

# Data path - adjust to your data location
DATA_ROOT="/path/to/your/brats/data"

echo "Starting multi-GPU training with $SLURM_NTASKS_PER_NODE GPUs"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"

# Option 1: Use the launcher script (recommended)
python scripts/train_multi_gpu.py \
    --data_root "$DATA_ROOT" \
    --config configs/fast_ddpm_3d.yml \
    --exp ./experiments \
    --doc "fast_ddpm_3d_multigpu_${SLURM_JOB_ID}" \
    --use_ddp \
    --use_wandb \
    --wandb_project "fast-ddpm-3d-brats-multigpu"

# Option 2: Direct distributed launch (alternative)
# torchrun --standalone --nproc_per_node=4 scripts/train_3d.py \
#     --data_root "$DATA_ROOT" \
#     --config configs/fast_ddpm_3d.yml \
#     --exp ./experiments \
#     --doc "fast_ddpm_3d_multigpu_${SLURM_JOB_ID}" \
#     --distributed \
#     --use_wandb \
#     --wandb_project "fast-ddpm-3d-brats-multigpu"

echo "Training completed!"
