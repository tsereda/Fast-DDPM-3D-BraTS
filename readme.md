# Fast-DDPM-3D-BraTS

3D Brain MRI synthesis using Fast-DDPM with unified 4→4 architecture for BraTS 2025.

## Overview

Converting Fast-DDPM to 3D volumetric processing with unified multi-modal training (any modality to any modality) instead of separate 3→1 models.

**Goals:**
- 3D Fast-DDPM (maintain 10-timestep efficiency)
- Unified 4→4 training (T1n, T1c, T2w, T2f ↔ any)
- Bayesian optimization for timestep selection
- BraTS 2025 competition submission

## Setup

```bash
git clone [this-repo]
cd Fast-DDPM-3D-BraTS

conda create -n fast-ddpm-brats python=3.10
conda activate fast-ddpm-brats
pip install -r requirements.txt
```

## Implementation Plan

### Phase 1: Get Fast-DDPM working in 3D
1. Convert `models/diffusion.py` → `models/diffusion_3d.py` (Conv2d → Conv3d)
2. Convert `functions/denoising.py` → `functions/denoising_3d.py` (handle 5D tensors)
3. Convert `runners/diffusion.py` → `runners/diffusion_3d.py` (3D data loading)
4. Test on small volumes (64x64x64) first

### Phase 2: BraTS data integration  
1. Extract BraSyn tutorial's data handling
2. Implement 3D BraTS dataset loading
3. Use proven cropping strategy (144x192x192)
4. Test training pipeline end-to-end

### Phase 3: Unified 4→4 architecture
1. Modify model to handle variable input channels
2. Implement random modality masking during training
3. Single model trains on all modality combinations
4. Compare vs separate 3→1 models

### Phase 4: Optimization and evaluation
1. Bayesian optimization for timestep selection
2. BraTS evaluation pipeline integration
3. Performance benchmarking

## Current Status

- [ ] 3D model conversion
- [ ] 3D data pipeline  
- [ ] Basic 3D training working
- [ ] BraTS data integration
- [ ] Unified 4→4 architecture
- [ ] Bayesian optimization
- [ ] Competition submission

## Literature Review Summary

### Fast-DDPM Foundation
**Jiang et al. (2025)** demonstrated 100x faster training and 0.01x sampling time vs standard DDPM using only 10 timesteps. Key innovations:
- Dual noise schedulers (uniform/non-uniform)
- Aligned training-sampling strategies
- Medical imaging applications: SSIM 0.89-0.91 on T1w→T2w tasks

### 3D Medical Diffusion Models
**Medical Diffusion (Nature 2023)** showed 3D volumetric processing superior to 2D slice-by-slice:
- Dice improvements: 0.91→0.95 with 3D synthetic pre-training
- Eliminated inter-slice artifacts
- Required >24GB VRAM for 128³ volumes

**3D Wavelet Diffusion (BraTS 2024 #2)** achieved PSNR 22.8, SSIM 0.91:
- 8x spatial reduction via DWT 
- Full resolution training on single 48GB GPU
- Operates in wavelet domain, reconstructs via IDWT

### Unified Multi-Modal Approaches
**Evidence strongly favors unified 4→4 over separate 3→1 models:**
- 4x parameter reduction (13.3M vs 52M+)
- 5.7x faster training
- Single model handles any missing modality scenario
- StarGAN-inspired frameworks with modality labels

### BraTS Competition Context
**BraTS Syn 2025** expanded to 12 tasks with lighthouse status:
- 5-metric ranking: 3 Dice + 2 SSIM (tumor/non-tumor regions)
- 4,000+ training cases, multi-institutional
- FDA/NIH partnerships require clinical validation
- Performance targets: SSIM >0.91, Dice >0.85

## Quick Commands

## Quick Commands

```bash
# Basic 2D→3D conversion test (start here)
python test_3d_conversion.py

# Train basic 3D model on small volumes
python train_3d.py --config configs/basic_3d.yml --volume_size 64 64 64

# Test unified 4→4 training
python train_unified.py --config configs/unified_4to4.yml
```

## Repository Structure

```
Fast-DDPM-3D-BraTS/
├── core/                     # Modified Fast-DDPM components
│   ├── models/
│   │   ├── diffusion_3d.py  # 3D conversion of Fast-DDPM
│   │   └── unified_4to4.py  # Unified architecture
│   ├── functions/
│   └── runners/
├── data/                     # BraTS data handling
├── configs/                  # Training configurations  
├── scripts/                  # Training/inference scripts
└── tests/                    # Unit tests
```

## Key Files to Create

1. **core/models/diffusion_3d.py** - Conv2d → Conv3d conversion
2. **data/brats_3d_unified.py** - 4→4 dataset class  
3. **configs/basic_3d.yml** - 3D training config
4. **scripts/train_3d.py** - Main training script

## Memory Considerations

- Start with 64³ volumes (fits in 16GB GPU)
- Use batch_size=1 for 3D training
- Enable gradient checkpointing
- Mixed precision training (autocast)

## References

**Core Papers:**
- Fast-DDPM (JBHI 2025): https://ieeexplore.ieee.org/abstract/document/10979336
- BraTS 2023 Challenge: https://arxiv.org/abs/2305.09011  
- 3D Medical Diffusion (Nature 2023): https://www.nature.com/articles/s41598-023-34341-2

**Code Repositories:**
- Fast-DDPM: https://github.com/mirthAI/Fast-DDPM
- BraSyn Tutorial: https://github.com/WinstonHuTiger/BraSyn_tutorial