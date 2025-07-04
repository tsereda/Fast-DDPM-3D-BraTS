import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from pathlib import Path
import random
import logging

logger = logging.getLogger(__name__)


class BraTS3DUnifiedDataset(Dataset):
    """
    BraTS 3D dataset with unified [-1, 1] normalization, simplified interface, reproducible crops,
    and robust modality handling. Returns only available_modalities (non-target, loaded).
    """
    
    def __init__(self, data_root, phase='train', volume_size=(64, 64, 64), min_input_modalities=3, samples_per_volume=4):
        self.data_root = Path(data_root)
        self.phase = phase
        self.volume_size = tuple(volume_size)
        self.min_input_modalities = min_input_modalities
        self.samples_per_volume = samples_per_volume
        self.modalities = ['t1n', 't1c', 't2w', 't2f']
        self.cases = self._find_cases()
        if len(self.cases) == 0:
            raise ValueError(f"No valid cases found in {data_root}")
        self.samples = [(case, crop_idx) for case in self.cases for crop_idx in range(self.samples_per_volume)]
        logger.info(f"Found {len(self.cases)} cases, {len(self.samples)} total samples for {phase} phase (patches)")

    def _find_cases(self):
        cases = []
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root {self.data_root} does not exist")
        for item in self.data_root.iterdir():
            if item.is_dir():
                available_count = sum(1 for mod in self.modalities if self._find_modality_file(item, mod) is not None)
                if available_count >= self.min_input_modalities:
                    cases.append(item)
        return sorted(cases, key=lambda x: x.name)

    def _find_modality_file(self, case_dir, modality):
        patterns = [
            f'*-{modality}.nii*',
            f'*_{modality}.nii*',
            f'*{modality.upper()}.nii*',
            f'*{modality}.nii*'
        ]
        for pattern in patterns:
            files = list(case_dir.glob(pattern))
            if files:
                exact_files = []
                for f in files:
                    fname = f.stem.lower()
                    if (f'-{modality}.' in f.name.lower() or 
                        f'_{modality}.' in f.name.lower() or
                        f'{modality}.' in f.name.lower() or
                        fname.endswith(f'-{modality}') or
                        fname.endswith(f'_{modality}')):
                        exact_files.append(f)
                if exact_files:
                    exact_files.sort(key=lambda x: (x.suffix != '.gz', x.name))
                    return exact_files[0]
        return None

    def _load_full_volume(self, filepath):
        try:
            nii = nib.load(str(filepath))
            volume = nii.get_fdata().astype(np.float32)
            return volume, nii.affine
        except Exception as e:
            logger.warning(f"Error loading {filepath}: {e}")
            return None, None

    def _get_random_crop_coords(self, volume_shape, idx):
        # Use reproducible seed based on sample idx
        random.seed(idx)
        coords = []
        for i, (vol_dim, crop_dim) in enumerate(zip(volume_shape, self.volume_size)):
            if vol_dim <= crop_dim:
                start = 0
            else:
                max_start = vol_dim - crop_dim
                start = random.randint(0, max_start)
            coords.append((start, start + crop_dim))
        return coords

    def _extract_crop(self, volume, crop_coords):
        x_start, x_end = crop_coords[0]
        y_start, y_end = crop_coords[1]
        z_start, z_end = crop_coords[2]
        crop = np.zeros(self.volume_size, dtype=volume.dtype)
        actual_x = min(x_end, volume.shape[0]) - x_start
        actual_y = min(y_end, volume.shape[1]) - y_start
        actual_z = min(z_end, volume.shape[2]) - z_start
        if actual_x > 0 and actual_y > 0 and actual_z > 0:
            crop[:actual_x, :actual_y, :actual_z] = volume[
                x_start:x_start+actual_x,
                y_start:y_start+actual_y,
                z_start:z_start+actual_z
            ]
        return crop

    def _normalize_volume(self, volume):
        """Normalize to [-1, 1] range"""
        if not np.any(volume > 0):
            return np.zeros_like(volume)
        v_min = np.amin(volume)
        v_max = np.amax(volume)
        if v_max > v_min:
            volume = 2 * (volume - v_min) / (v_max - v_min) - 1
        return np.clip(volume, -1.0, 1.0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        case_dir, crop_idx = self.samples[idx]
        modality_volumes = {}
        loaded_modalities = []
        volume_shape = None
        for modality in self.modalities:
            modality_file = self._find_modality_file(case_dir, modality)
            if modality_file:
                volume, affine = self._load_full_volume(modality_file)
                if volume is not None:
                    modality_volumes[modality] = volume
                    loaded_modalities.append(modality)
                    if volume_shape is None:
                        volume_shape = volume.shape
        if len(loaded_modalities) < self.min_input_modalities:
            logger.warning(f"Case {case_dir.name}: only {len(loaded_modalities)} modalities found, filling with zeros.")
            volume_shape = (240, 240, 155)
        for modality in self.modalities:
            if modality not in modality_volumes:
                if volume_shape is None:
                    volume_shape = (240, 240, 155)
                modality_volumes[modality] = np.zeros(volume_shape, dtype=np.float32)
        crop_coords = self._get_random_crop_coords(volume_shape, idx)
        processed_volumes = {}
        for modality in self.modalities:
            volume = modality_volumes.get(modality, np.zeros(volume_shape, dtype=np.float32))
            cropped = self._extract_crop(volume, crop_coords)
            normalized = self._normalize_volume(cropped)
            processed_volumes[modality] = torch.FloatTensor(normalized)
        # Select target modality from loaded modalities
        if self.phase == 'train' and len(loaded_modalities) > 0:
            target_modality = random.choice(loaded_modalities)
        elif len(loaded_modalities) > 0:
            target_modality = loaded_modalities[0]
        else:
            target_modality = self.modalities[0]
        target_idx = self.modalities.index(target_modality)
        target_volume = processed_volumes[target_modality]
        input_modalities = torch.stack([processed_volumes[mod] for mod in self.modalities])
        input_modalities[target_idx] = torch.zeros_like(input_modalities[target_idx])
        available_modalities = [mod for mod in loaded_modalities if mod != target_modality]
        return {
            'input': input_modalities,      # [4, H, W, D] with target as ZEROS, range [-1,1]
            'target': target_volume,        # [H, W, D] target modality, range [-1,1]
            'target_idx': target_idx,
            'case_name': case_dir.name,
            'target_modality': target_modality,
            'available_modalities': available_modalities,
            'crop_coords': crop_coords,
            'processing_mode': 'patch'
        }