# data/dataset.py

import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from .transforms import SegmentationTransform
from rich import print

class AerialDeadTreeSegDataset(Dataset):
    def __init__(self, rgb_paths, nrg_paths, mask_paths, transform=None, modality="merged", mode="train"):
        self.rgb_paths  = rgb_paths
        self.nrg_paths  = nrg_paths
        self.mask_paths = mask_paths
        self.transform  = transform
        self.modality    = modality
        self.mode        = mode
        
        if self.transform is None:
            print("[yellow]Warning: No transform provided. Using default SegmentationTransform.[/yellow]")
            self.transform = SegmentationTransform(modality=modality)
        
        if self.modality not in ["rgb", "nrg", "merged"]:
            print(f"[yellow]Warning: Unknown modality '{self.modality}'. Defaulting to 'merged'.[/yellow]")
            self.modality = "merged" # Default to merged if not specified

        self._load_strategies = {
            "rgb": self._load_rgb,
            "nrg": self._load_nrg,
            "merged": self._load_merged
        }

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        # Load mask
        mask_path = self.mask_paths[idx]
        mask = Image.open(mask_path).convert("L")

        # Load image based on the specified modality
        img = self._load_strategies[self.modality](idx)

        img, mask = self.transform(img, mask)
        
        if self.mode == "test":
            suffix = os.path.basename(mask_path).split("_", 1)[1]
            name = str(suffix).split(".", 1)[0]
            return {"image": img, "mask": mask, "name": name}
        else:
            return {"image": img, "mask": mask}


    def _load_rgb(self, idx):
        """Load RGB image"""
        return Image.open(self.rgb_paths[idx]).convert("RGB")

    def _load_nrg(self, idx):
        return Image.open(self.nrg_paths[idx]).convert("RGB")
    
    def _load_merged(self, idx):
        img_rgb = Image.open(self.rgb_paths[idx]).convert("RGB")
        img_nrg = Image.open(self.nrg_paths[idx]).convert("RGB")
        
        # Merging RGB and NIR channels
        rgb_arr = np.array(img_rgb)
        nir_arr = np.array(img_nrg)[:, :, 0]  # Just take the NIR channel
        merged = np.dstack([rgb_arr, nir_arr])  # H×W×4
        
        return Image.fromarray(merged.astype(np.uint8))
    
    def get_mean_std(self):
        """Get mean and std for normalization based on the modality."""
        
        # Decide num of channels based on the modality
        sample = self._load_strategies
        channels_num = np.array(sample).shape[2]

        sum_     = np.zeros(channels_num, dtype=np.float64)
        sum_sq   = np.zeros(channels_num, dtype=np.float64)
        pixel_ct = 0
        
        for i in range(len(self)):
            img = self._load_strategies[self.modality](i)
            arr = np.array(img, dtype=np.float32) / 255.0    # Normalize to [0, 1]
            h, w, _ = arr.shape
            arr_flat = arr.reshape(-1, channels_num)                    # (H*W)×C
            sum_     += arr_flat.sum(axis=0)
            sum_sq   += (arr_flat ** 2).sum(axis=0)
            pixel_ct += h * w
        
        mean = sum_ / pixel_ct
        var  = sum_sq / pixel_ct - mean**2
        std  = np.sqrt(var)
        
        return mean.tolist(), std.tolist()
        