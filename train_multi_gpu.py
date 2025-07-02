#!/usr/bin/env python3
"""
Multi-GPU training launcher for Fast-DDPM 3D BraTS
"""

import os
import sys
import argparse
import subprocess
import torch

def get_available_gpus():
    """Get list of available GPU IDs"""
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-GPU Fast-DDPM 3D Training Launcher')
    parser.add_argument('--config', type=str, default='configs/fast_ddpm_3d.yml', help='Path to config file')
    parser.add_argument('--data_root', type=str, required=True, help='Path to BraTS data')
    parser.add_argument('--exp', type=str, default='./experiments', help='Experiment directory')
    parser.add_argument('--doc', type=str, default='fast_ddpm_3d_brats_multigpu', help='Experiment name')
    parser.add_argument('--gpus', type=str, default='auto', help='GPU IDs to use (e.g., "0,1,2,3" or "auto")')
    parser.add_argument('--backend', type=str, default='nccl', choices=['nccl', 'gloo'], help='Distributed backend')
    parser.add_argument('--port', type=str, default='12355', help='Master port for distributed training')
    parser.add_argument('--use_ddp', action='store_true', help='Use DistributedDataParallel instead of DataParallel')
    parser.add_argument('--debug', action='store_true', help='Debug mode with smaller dataset')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='fast-ddpm-3d-brats', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Get available GPUs
    if args.gpus == 'auto':
        gpu_ids = get_available_gpus()
    else:
        gpu_ids = [int(x) for x in args.gpus.split(',')]
    
    if not gpu_ids:
        print("No GPUs available!")
        sys.exit(1)
    
    print(f"Using GPUs: {gpu_ids}")
    print(f"Number of GPUs: {len(gpu_ids)}")
    
    # Set environment variables
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
    
    if args.use_ddp and len(gpu_ids) > 1:
        # Use DistributedDataParallel
        print("Launching DistributedDataParallel training...")
        
        # Set up distributed training environment
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = args.port
        os.environ['WORLD_SIZE'] = str(len(gpu_ids))
        
        # Launch training processes
        processes = []
        for i, gpu_id in enumerate(gpu_ids):
            env = os.environ.copy()
            env['RANK'] = str(i)
            env['LOCAL_RANK'] = str(i)
            env['CUDA_VISIBLE_DEVICES'] = str(i)  # Each process sees only its GPU
            
            cmd = [
                sys.executable, 'main_train.py',
                '--config', args.config,
                '--data_root', args.data_root,
                '--exp', args.exp,
                '--doc', args.doc,
                '--distributed',
                '--world_size', str(len(gpu_ids)),
                '--rank', str(i),
                '--local_rank', str(i),
                '--dist_backend', args.backend,
                '--gpu', str(i)
            ]
            
            if args.debug:
                cmd.append('--debug')
            if args.use_wandb:
                cmd.extend(['--use_wandb', '--wandb_project', args.wandb_project])
                if args.wandb_entity:
                    cmd.extend(['--wandb_entity', args.wandb_entity])
            
            print(f"Starting process {i} on GPU {gpu_id}")
            print(f"Command: {' '.join(cmd)}")
            
            p = subprocess.Popen(cmd, env=env)
            processes.append(p)
        
        # Wait for all processes to complete
        try:
            for p in processes:
                p.wait()
        except KeyboardInterrupt:
            print("Terminating all processes...")
            for p in processes:
                p.terminate()
            for p in processes:
                p.wait()
    
    else:
        # Use DataParallel or single GPU
        print("Launching DataParallel/Single GPU training...")
        
        cmd = [
            sys.executable, 'main_train.py',
            '--config', args.config,
            '--data_root', args.data_root,
            '--exp', args.exp,
            '--doc', args.doc,
            '--gpu', str(gpu_ids[0])
        ]
        
        if len(gpu_ids) > 1:
            cmd.append('--multi_gpu')
            print(f"Using DataParallel with {len(gpu_ids)} GPUs")
        
        if args.debug:
            cmd.append('--debug')
        if args.use_wandb:
            cmd.extend(['--use_wandb', '--wandb_project', args.wandb_project])
            if args.wandb_entity:
                cmd.extend(['--wandb_entity', args.wandb_entity])
        
        print(f"Command: {' '.join(cmd)}")
        subprocess.run(cmd)

if __name__ == '__main__':
    main()
