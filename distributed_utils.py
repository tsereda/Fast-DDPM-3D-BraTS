"""
Distributed training utilities
"""
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import logging


def setup_distributed(args):
    """
    Initialize distributed training environment
    
    Args:
        args: Command line arguments
        
    Returns:
        rank, world_size, device
    """
    if args.distributed:
        # Initialize distributed training
        if args.local_rank == -1:
            if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
                args.rank = int(os.environ["RANK"])
                args.world_size = int(os.environ['WORLD_SIZE'])
                args.local_rank = int(os.environ['LOCAL_RANK'])
            elif 'SLURM_PROCID' in os.environ:
                # SLURM environment
                args.rank = int(os.environ['SLURM_PROCID'])
                args.local_rank = args.rank % torch.cuda.device_count()
                args.world_size = int(os.environ['SLURM_NPROCS'])
            else:
                print('Not using distributed mode')
                args.distributed = False
                args.multi_gpu = torch.cuda.device_count() > 1
                return 0, 1, torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

        # Set device
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f'cuda:{args.local_rank}')
        
        # Initialize process group
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank
        )
        
        # Wait for all processes
        dist.barrier()
        
        print(f"Distributed training initialized: rank {args.rank}/{args.world_size}, local_rank {args.local_rank}")
        
        return args.rank, args.world_size, device
    
    elif args.multi_gpu and torch.cuda.device_count() > 1:
        # Simple DataParallel setup
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        return 0, 1, device
    
    else:
        # Single GPU setup
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        return 0, 1, device


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    """Check if current process is the main process"""
    return rank == 0


def save_on_master(state, filename, rank):
    """Save checkpoint only on master process"""
    if is_main_process(rank):
        torch.save(state, filename)


def reduce_loss_dict(loss_dict, world_size):
    """
    Reduce loss dictionary across all processes
    
    Args:
        loss_dict: Dictionary of losses
        world_size: Number of processes
        
    Returns:
        Reduced loss dictionary
    """
    if world_size == 1:
        return loss_dict
        
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # Only main process keeps the reduced losses
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def get_effective_batch_size(batch_size, world_size, gradient_accumulation_steps):
    """Calculate effective batch size for distributed training"""
    return batch_size * world_size * gradient_accumulation_steps


def main_worker(rank, world_size, args, main_func):
    """Worker function for distributed training"""
    args.rank = rank
    args.world_size = world_size
    args.local_rank = rank
    main_func(args)


def launch_distributed_training(args, main_func):
    """Launch distributed training using multiprocessing"""
    if args.distributed and args.world_size > 1:
        # Launch distributed training
        mp.spawn(main_worker, args=(args.world_size, args, main_func), nprocs=args.world_size, join=True)
    else:
        # Single process training
        try:
            main_func(args)
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            cleanup_distributed()
            raise