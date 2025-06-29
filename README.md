# Fast-DDPM 3D for BraTS

3D implementation of Fast-DDPM for brain MRI synthesis. Generates missing MRI modalities from available ones using only 10 denoising steps.

## Features
- **Fast-DDPM with variance learning** - 10-step generation instead of 1000
- **Unified 4→4 training** - Single model handles any input/output combination
- **Memory efficient** - Works on 16GB GPUs

## Quick Start

### Install
```bash
conda create -n fastddpm3d python=3.10
conda activate fastddpm3d
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install nibabel numpy scipy tqdm pyyaml
```

### Train
```bash
python scripts/train_3d.py \
    --data_root ./BraTS2023 \
    --config configs/fast_ddpm_3d.yml \
    --exp ./experiments
```

### Generate Missing Modalities
```bash
python scripts/inference_3d.py \
    --checkpoint ./experiments/best_model.pth \
    --input_dir ./test_cases \
    --output_dir ./outputs \
    --target_modality t1c \
    --timesteps 10
```

## Model Details

The model uses Fast-DDPM's learned variance to reduce sampling from 1000 steps to just 10 steps while maintaining quality.

**Input**: Any 1-3 of the 4 BraTS modalities (T1n, T1c, T2w, T2f)  
**Output**: Any missing modality

## Memory Requirements
- 64³ volumes: 8GB VRAM
- 96³ volumes: 16GB VRAM  
- 128³ volumes: 24GB VRAM

## Citation
If you use this code, please cite the Fast-DDPM paper and BraTS challenge.