# Adapted from BraSyn tutorial for unified 4→4 Fast-DDPM training
import os
import random
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class BraTS3DUnifiedDataset(Dataset):
    """
    Unified 4→4 BraTS dataset for Fast-DDPM
    - Input: Any combination of 4 modalities (T1n, T1c, T2w, T2f)
    - Output: Any of the 4 modalities (unified training)
    """
    
    def __init__(self, data_root, phase='train', volume_size=(144, 192, 192)):
        self.data_root = data_root
        self.phase = phase
        self.volume_size = volume_size
        self.modalities = ['t1n', 't1c', 't2w', 't2f']
        
        # Get all patient folders
        if phase == 'train':
            dataset_name = "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
        elif phase == 'val':
            dataset_name = "ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData"
        else:
            dataset_name = ""
            
        self.data_path = os.path.join(data_root, dataset_name)
        self.patient_folders = [f for f in os.listdir(self.data_path) 
                               if os.path.isdir(os.path.join(self.data_path, f))]
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: self._reorient_volume(x)),
            transforms.Lambda(lambda x: self._crop_volume(x)),
            transforms.Lambda(lambda x: self._normalize_volume(x)),
            transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32))
        ])
    
    def __len__(self):
        return len(self.patient_folders)
    
    def __getitem__(self, idx):
        patient_folder = self.patient_folders[idx]
        patient_path = os.path.join(self.data_path, patient_folder)
        
        # Load all 4 modalities
        volumes = {}
        for modality in self.modalities:
            file_pattern = f"*-{modality}.nii.gz"
            files = [f for f in os.listdir(patient_path) if f.endswith(f"-{modality}.nii.gz")]
            if files:
                volume_path = os.path.join(patient_path, files[0])
                volume = nib.load(volume_path)
                volumes[modality] = self.transform(volume)
        
        # For unified training: randomly select input/output modalities
        available_modalities = list(volumes.keys())
        
        if self.phase == 'train':
            # Random training: any → any
            num_input_modalities = random.randint(1, 3)  # 1-3 input modalities
            input_modalities = random.sample(available_modalities, num_input_modalities)
            remaining = [m for m in available_modalities if m not in input_modalities]
            target_modality = random.choice(remaining) if remaining else random.choice(available_modalities)
        else:
            # For validation: use all available → first missing
            input_modalities = available_modalities[:-1]
            target_modality = available_modalities[-1]
        
        # Stack input modalities
        input_volumes = []
        for modality in self.modalities:
            if modality in input_modalities:
                input_volumes.append(volumes[modality])
            else:
                # Zero padding for missing modalities
                input_volumes.append(torch.zeros_like(volumes[available_modalities[0]]))
        
        input_tensor = torch.stack(input_volumes, dim=0)  # [4, H, W, D]
        target_tensor = volumes[target_modality]  # [H, W, D]
        
        # Create modality mask (which modalities are available)
        modality_mask = torch.zeros(4)
        for i, modality in enumerate(self.modalities):
            if modality in input_modalities:
                modality_mask[i] = 1.0
        
        return {
            'input': input_tensor,
            'target': target_tensor,
            'modality_mask': modality_mask,
            'target_modality': target_modality,
            'patient_id': patient_folder
        }
    
    def _reorient_volume(self, nifti_img):
        """Reorient to IPL orientation"""
        # Implementation from BraSyn
        orig_ornt = nib.orientations.io_orientation(nifti_img.affine)
        targ_ornt = nib.orientations.axcodes2ornt("IPL")
        transform = nib.orientations.ornt_transform(orig_ornt, targ_ornt)
        return nifti_img.as_reoriented(transform)
    
    def _crop_volume(self, nifti_img):
        """Crop to target size"""
        data = np.array(nifti_img.get_fdata())
        # Crop to 144x192x192 as in BraSyn
        data = data[8:152, 24:216, 24:216]
        return data
    
    def _normalize_volume(self, volume):
        """Normalize to [0, 1]"""
        v_min = np.amin(volume)
        v_max = np.amax(volume) - v_min
        if v_max > 0:
            volume = (volume - v_min) / v_max
        return volume