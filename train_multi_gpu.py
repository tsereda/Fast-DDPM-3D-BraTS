#!/usr/bin/env python3
"""
Multi-GPU training launcher for Fast-DDPM 3D BraTS
"""

import os
import sys
import argparse
import subprocess
import torch
import socket

def find_free_port():
    """Find a free port for distributed training"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def get_available_gpus():
    """Get list of available GPU IDs"""
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return []
    
    device_count = torch.cuda.device_count()
    print(f"Detected {device_count} CUDA devices")
    
    # Print GPU information
    for i in range(device_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
        print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    
    return list(range(device_count))

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-GPU Fast-DDPM 3D Training Launcher')
    parser.add_argument('--config', type=str, default='configs/fast_ddpm_3d.yml', help='Path to config file')
    parser.add_argument('--data_root', type=str, required=True, help='Path to BraTS data')
    parser.add_argument('--exp', type=str, default='./experiments', help='Experiment directory')
    parser.add_argument('--doc', type=str, default='fast_ddpm_3d_brats_multigpu', help='Experiment name')
    parser.add_argument('--gpus', type=str, default='auto', help='GPU IDs to use (e.g., "0,1,2,3" or "auto")')
    parser.add_argument('--backend', type=str, default='nccl', choices=['nccl', 'gloo'], help='Distributed backend')
    parser.add_argument('--port', type=str, default='auto', help='Master port for distributed training (auto for random port)')
    parser.add_argument('--use_ddp', action='store_true', help='Use DistributedDataParallel instead of DataParallel')
    parser.add_argument('--debug', action='store_true', help='Debug mode with smaller dataset')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='fast-ddpm-3d-brats', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=== Multi-GPU Training Launcher ===")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    # Get available GPUs
    if args.gpus == 'auto':
        gpu_ids = get_available_gpus()
        if len(gpu_ids) > 1:
            print(f"Auto-detected {len(gpu_ids)} GPUs - enabling multi-GPU training")
        else:
            print(f"Auto-detected {len(gpu_ids)} GPU - single GPU training")
    else:
        gpu_ids = [int(x) for x in args.gpus.split(',')]
        print(f"User specified GPUs: {gpu_ids}")
    
    if not gpu_ids:
        print("❌ No GPUs available!")
        sys.exit(1)
    
    print(f"✅ Using GPUs: {gpu_ids}")
    print(f"✅ Number of GPUs: {len(gpu_ids)}")
    
    # Decide on training strategy
    if len(gpu_ids) > 1:
        if args.use_ddp:
            print("🚀 Strategy: DistributedDataParallel (DDP)")
        else:
            print("🚀 Strategy: DataParallel (DP) - enabling multi_gpu flag")
            # Enable multi_gpu flag for main training script
    else:
        print("🚀 Strategy: Single GPU training")
    
    # Set environment variables
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
    
    if args.use_ddp and len(gpu_ids) > 1:
        # Use DistributedDataParallel
        print("Launching DistributedDataParallel training...")
        
        # Get a free port
        if args.port == 'auto':
            master_port = str(find_free_port())
            print(f"Using auto-selected port: {master_port}")
        else:
            master_port = args.port
        
        # Set up distributed training environment
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = master_port
        os.environ['WORLD_SIZE'] = str(len(gpu_ids))
        
        # Launch training processes
        processes = []
        for i, gpu_id in enumerate(gpu_ids):
            env = os.environ.copy()
            env['RANK'] = str(i)
            env['LOCAL_RANK'] = str(i)
            # Keep original CUDA_VISIBLE_DEVICES so each process can see all assigned GPUs
            # but they will use their local rank to select the correct one
            
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
            print(f"Environment - RANK: {i}, LOCAL_RANK: {i}, CUDA_VISIBLE_DEVICES: {env.get('CUDA_VISIBLE_DEVICES')}")
            
            try:
                p = subprocess.Popen(cmd, env=env)
                processes.append(p)
            except Exception as e:
                print(f"Failed to start process {i}: {e}")
                # Clean up any started processes
                for started_p in processes:
                    started_p.terminate()
                raise
        
        # Wait for all processes to complete
        try:
            exit_codes = []
            for i, p in enumerate(processes):
                exit_code = p.wait()
                exit_codes.append(exit_code)
                if exit_code != 0:
                    print(f"⚠️  Process {i} exited with code {exit_code}")
                else:
                    print(f"✅ Process {i} completed successfully")
            
            if any(code != 0 for code in exit_codes):
                print(f"❌ Some processes failed. Exit codes: {exit_codes}")
                sys.exit(1)
            else:
                print("🎉 All processes completed successfully!")
                
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user. Terminating all processes...")
            for p in processes:
                p.terminate()
            for p in processes:
                p.wait()
            sys.exit(1)
    
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
