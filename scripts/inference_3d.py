#!/usr/bin/env python3
"""
3D Fast-DDPM Inference Script for BraTS Modality Synthesis
"""

import os
import sys
import argparse
import yaml
import torch
import nibabel as nib
import numpy as np
from tqdm import tqdm

# Add path
sys.path.append('.')
sys.path.append('..')

from models.fast_ddpm_3d import FastDDPM3D
from functions.denoising_3d import generalized_steps_3d, unified_4to4_generalized_steps
from data.brain_3d_unified import BraTS3DUnifiedDataset
from data.image_folder import get_available_3d_vol_names


def parse_args():
    parser = argparse.ArgumentParser(description='3D Fast-DDPM Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory with BraTS data')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--target_modality', type=str, choices=['t1n', 't1c', 't2w', 't2f'], 
                       required=True, help='Target modality to generate')
    parser.add_argument('--timesteps', type=int, default=10, help='Number of sampling timesteps')
    parser.add_argument('--eta', type=float, default=0.0, help='DDIM eta parameter')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--scheduler', type=str, default='uniform', choices=['uniform', 'non-uniform'])
    return parser.parse_args()


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """Beta schedule for diffusion"""
    if beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "cosine":
        steps = num_diffusion_timesteps + 1
        s = 0.008
        x = np.linspace(0, num_diffusion_timesteps, steps)
        alphas_cumprod = np.cos(((x / num_diffusion_timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = np.clip(betas, 0, 0.999)
    else:
        raise NotImplementedError(f"Beta schedule {beta_schedule} not implemented")
    
    return betas


def get_sampling_sequence(scheduler, timesteps, num_timesteps):
    """Get sampling sequence for Fast-DDPM"""
    if scheduler == 'uniform':
        skip = num_timesteps // timesteps
        seq = range(0, num_timesteps, skip)
    elif scheduler == 'non-uniform':
        if timesteps == 10:
            seq = [0, 199, 399, 599, 699, 799, 849, 899, 949, 999]
        else:
            # Adaptive non-uniform
            num_1 = int(timesteps * 0.4)
            num_2 = int(timesteps * 0.6)
            stage_1 = torch.linspace(0, 699, num_1 + 1)[:-1]
            stage_2 = torch.linspace(699, 999, num_2)
            stage_1 = torch.ceil(stage_1).long()
            stage_2 = torch.ceil(stage_2).long()
            seq = torch.cat((stage_1, stage_2)).tolist()
    else:
        raise ValueError(f"Unknown scheduler: {scheduler}")
    
    return list(seq)


def load_brats_case(case_path, modalities, volume_size, target_modality):
    """Load a BraTS case with available modalities"""
    import torchvision.transforms as transforms
    
    def reorient_volume(nifti_img):
        orig_ornt = nib.orientations.io_orientation(nifti_img.affine)
        targ_ornt = nib.orientations.axcodes2ornt("IPL")
        transform = nib.orientations.ornt_transform(orig_ornt, targ_ornt)
        return nifti_img.as_reoriented(transform)
    
    def crop_volume(nifti_img):
        data = np.array(nifti_img.get_fdata())
        # Crop to 144x192x192 as in BraSyn
        data = data[8:152, 24:216, 24:216]
        return data
    
    def normalize_volume(volume):
        v_min = np.amin(volume)
        v_max = np.amax(volume) 
        if v_max > v_min:
            # Normalize to [-1, 1] range to match training data
            volume = 2 * (volume - v_min) / (v_max - v_min) - 1
        return volume
    
    # Find available files
    vol_files, _ = get_available_3d_vol_names(case_path)
    
    volumes = {}
    available_modalities = []
    
    for modality in modalities:
        if vol_files[modality] is not None:
            file_path = os.path.join(case_path, vol_files[modality])
            
            # Load and process
            nifti_img = nib.load(file_path)
            nifti_img = reorient_volume(nifti_img)
            data = crop_volume(nifti_img)
            data = normalize_volume(data)
            
            # Resize if needed
            if data.shape != volume_size:
                data = torch.tensor(data).float()
                data = torch.nn.functional.interpolate(
                    data.unsqueeze(0).unsqueeze(0), 
                    size=volume_size, 
                    mode='trilinear', 
                    align_corners=False
                ).squeeze()
                data = data.numpy()
            
            volumes[modality] = torch.tensor(data, dtype=torch.float32)
            available_modalities.append(modality)
            
            # Store affine for saving
            if modality == available_modalities[0]:
                reference_affine = nifti_img.affine
    
    # Create input tensor [4, H, W, D]
    input_tensor = torch.zeros(4, *volume_size)
    modality_mask = torch.zeros(4)
    
    modality_order = ['t1n', 't1c', 't2w', 't2f']
    for i, modality in enumerate(modality_order):
        if modality in volumes and modality != target_modality:
            input_tensor[i] = volumes[modality]
            modality_mask[i] = 1.0
    
    # Get target if available (for comparison)
    target_tensor = None
    if target_modality in volumes:
        target_tensor = volumes[target_modality]
    
    return input_tensor, modality_mask, target_tensor, reference_affine, available_modalities


def save_generated_volume(volume, affine, output_path):
    """Save generated volume as NIfTI"""
    volume_np = volume.cpu().numpy()
    
    # Denormalize from [-1, 1] to [0, 1] range
    volume_np = (volume_np + 1) / 2
    volume_np = np.clip(volume_np, 0, 1)
    
    # Scale to appropriate intensity range (0-1000 for medical images)
    volume_np = (volume_np * 1000).astype(np.float32)
    
    # Create NIfTI image
    nifti_img = nib.Nifti1Image(volume_np, affine)
    nib.save(nifti_img, output_path)


def main():
    args = parse_args()
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']
    
    # Initialize model
    model = FastDDPM3D(config).to(device)
    
    # Load weights
    if 'module.' in list(checkpoint['model_state_dict'].keys())[0]:
        # Model was saved with DataParallel
        model = torch.nn.DataParallel(model)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from step {checkpoint['step']}")
    
    # Setup diffusion parameters
    betas = get_beta_schedule(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    )
    betas = torch.from_numpy(betas).float().to(device)
    
    # Get sampling sequence
    seq = get_sampling_sequence(args.scheduler, args.timesteps, len(betas))
    print(f"Sampling sequence: {seq}")
    
    # Process cases
    modalities = ['t1n', 't1c', 't2w', 't2f']
    volume_size = tuple(config.data.volume_size)
    
    # Find all case directories
    case_dirs = [d for d in os.listdir(args.input_dir) 
                if os.path.isdir(os.path.join(args.input_dir, d))]
    case_dirs.sort()
    
    print(f"Found {len(case_dirs)} cases")
    print(f"Target modality: {args.target_modality}")
    print(f"Volume size: {volume_size}")
    
    results = []
    
    for case_dir in tqdm(case_dirs, desc="Processing cases"):
        case_path = os.path.join(args.input_dir, case_dir)
        
        try:
            # Load case
            input_tensor, modality_mask, target_tensor, reference_affine, available_modalities = load_brats_case(
                case_path, modalities, volume_size, args.target_modality
            )
            
            print(f"\nCase: {case_dir}")
            print(f"Available modalities: {available_modalities}")
            print(f"Missing: {args.target_modality}")
            
            # Skip if target modality is not missing
            if args.target_modality in available_modalities:
                print(f"Target modality {args.target_modality} already exists, skipping...")
                continue
            
            # Prepare input
            input_batch = input_tensor.unsqueeze(0).to(device)  # [1, 4, H, W, D]
            modality_mask_batch = modality_mask.unsqueeze(0).to(device)  # [1, 4]
            
            # Determine target modality index
            modality_order = ['t1n', 't1c', 't2w', 't2f']
            target_idx = modality_order.index(args.target_modality)
            
            # Initial noise
            noise_shape = (1, 1, *volume_size)
            x = torch.randn(noise_shape).to(device)
            
            print(f"Input shape: {input_batch.shape}")
            print(f"Noise shape: {x.shape}")
            print(f"Target index: {target_idx}")
            
            with torch.no_grad():
                # Sampling - corrected function call
                print("Generating...")
                xs, x0_preds = unified_4to4_generalized_steps(
                    x, input_batch, target_idx, seq, model, betas, eta=args.eta
                )
                
                # Get final result
                generated = xs[-1].squeeze(0).squeeze(0)  # [H, W, D]
                
                # Clamp to valid range
                generated = torch.clamp(generated, -1, 1)
                
                print(f"Generated shape: {generated.shape}")
                print(f"Generated range: [{generated.min():.3f}, {generated.max():.3f}]")
                
                # Save result
                output_filename = f"{case_dir}_{args.target_modality}_generated.nii.gz"
                output_path = os.path.join(args.output_dir, output_filename)
                
                save_generated_volume(generated, reference_affine, output_path)
                print(f"Saved: {output_path}")
                
                # If target exists, save for comparison
                if target_tensor is not None:
                    target_filename = f"{case_dir}_{args.target_modality}_ground_truth.nii.gz"
                    target_path = os.path.join(args.output_dir, target_filename)
                    save_generated_volume(target_tensor, reference_affine, target_path)
                    
                    # Calculate simple metrics
                    mse = torch.mean((generated - target_tensor) ** 2).item()
                    mae = torch.mean(torch.abs(generated - target_tensor)).item()
                    print(f"MSE: {mse:.6f}, MAE: {mae:.6f}")
                    
                    results.append({
                        'case': case_dir,
                        'mse': mse,
                        'mae': mae,
                        'available': available_modalities
                    })
                
        except FileNotFoundError as e:
            print(f"File not found for {case_dir}: {e}")
            continue
        except torch.cuda.OutOfMemoryError as e:
            print(f"GPU out of memory for {case_dir}: {e}")
            torch.cuda.empty_cache()
            continue
        except ValueError as e:
            print(f"Value error processing {case_dir}: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error processing {case_dir}: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            continue
    
    # Summary
    if results:
        avg_mse = np.mean([r['mse'] for r in results])
        avg_mae = np.mean([r['mae'] for r in results])
        print(f"\nSummary:")
        print(f"Processed {len(results)} cases with ground truth")
        print(f"Average MSE: {avg_mse:.6f}")
        print(f"Average MAE: {avg_mae:.6f}")
        
        # Save results
        import json
        results_path = os.path.join(args.output_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    print(f"\nInference completed! Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()