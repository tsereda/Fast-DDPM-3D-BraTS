"""
3D Diffusion Model Interface
Imports the FastDDPM3D model for backward compatibility
"""

from .fast_ddpm_3d import FastDDPM3D

# Alias for training scripts
Model3D = FastDDPM3D
Model = FastDDPM3D
