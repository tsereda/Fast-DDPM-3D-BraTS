#!/bin/bash
# Fast-DDPM-3D-BraTS Environment Setup Script
# Compatible with the pod.yml configuration

set -e  # Exit on any error

echo "🚀 Starting Fast-DDPM-3D-BraTS Environment Setup..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update && sudo apt-get install -y p7zip-full wget git

# Initialize conda/mamba
echo "🐍 Initializing conda environment..."
source /opt/conda/etc/profile.d/mamba.sh

# Create and activate environment
echo "🔧 Creating brasyn environment..."
mamba env create -f environment.yml -y || {
    echo "⚠️ Environment creation failed, trying with conda..."
    conda env create -f environment.yml -y
}

echo "✅ Activating brasyn environment..."
mamba activate brasyn || conda activate brasyn

# Verify PyTorch installation
echo "🔍 Verifying PyTorch CUDA installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Install additional pip packages if needed
echo "📋 Installing additional requirements..."
pip install -r requirements.txt --upgrade

# Verify key imports
echo "🧪 Testing key imports..."
python -c "
try:
    import torch
    import nibabel as nib
    import numpy as np
    import tqdm
    import yaml
    print('✅ All core imports successful')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

# Test model imports
echo "🧪 Testing model imports..."
python -c "
import sys
sys.path.append('.')
try:
    from models.diffusion_3d import Model3D
    print('✅ Model3D import successful')
except ImportError as e:
    print(f'⚠️ Model3D import failed (expected if not yet created): {e}')
"

echo "🎉 Environment setup complete!"
echo ""
echo "To use the environment:"
echo "  source /opt/conda/etc/profile.d/mamba.sh"
echo "  mamba activate brasyn"
echo ""
echo "To verify installation:"
echo "  python test_3d_model.py"
echo ""
echo "Memory recommendations by GPU:"
echo "  8GB GPU:  volume_size: [64, 64, 64]"
echo "  16GB GPU: volume_size: [96, 96, 96]" 
echo "  24GB GPU: volume_size: [112, 112, 112]"
echo "  32GB+ GPU: volume_size: [128, 128, 128]"
