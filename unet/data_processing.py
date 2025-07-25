# data_processing.py
import os
import glob
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import config

class SegmentationTransform:
    def __init__(self):
        self.resize    = T.Resize(config.IMG_SIZE)
        self.to_tensor = T.ToTensor()
    def __call__(self, img, mask):
        img  = self.resize(img)
        mask = self.resize(mask)
        img  = self.to_tensor(img)
        mask = self.to_tensor(mask)
        mask = (mask > 0.5).float()
        return img, mask

class ForestDataset(Dataset):
    def __init__(self, rgb_paths, nrg_paths, mask_paths, transform=None):
        self.rgb_paths  = rgb_paths
        self.nrg_paths  = nrg_paths
        self.mask_paths = mask_paths
        self.transform  = transform

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        # 读取 RGB
        img_rgb = Image.open(self.rgb_paths[idx]).convert("RGB")
        # 读取 NRG, 三通道里第一个是 NIR
        img_nrg = Image.open(self.nrg_paths[idx]).convert("RGB")
        rgb_arr = np.array(img_rgb)
        nir_arr = np.array(img_nrg)[:, :, 0]
        merged  = np.dstack([rgb_arr, nir_arr])                # H×W×4
        merged  = Image.fromarray(merged.astype(np.uint8))
        mask    = Image.open(self.mask_paths[idx]).convert("L")
        if self.transform:
            img_tensor, mask_tensor = self.transform(merged, mask)
        return img_tensor, mask_tensor

def make_dataloaders(rgb_dir, nrg_dir):
    rgb_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    nrg_paths = sorted(glob.glob(os.path.join(nrg_dir, "*.png")))

    # 构造 mask 路径
    mask_paths = []
    for p in rgb_paths:
        base     = os.path.basename(p)            # e.g. "RGB_ar037_2019_n_06_04_0.png"
        parts    = base.split("_")
        mask_name= "mask_" + "_".join(parts[1:])
        mask_paths.append(os.path.join(config.MASK_DIR, mask_name))

    # 80/20 划分
    train_rgb, val_rgb, train_nrg, val_nrg, train_m, val_m = train_test_split(
        rgb_paths, nrg_paths, mask_paths,
        test_size=config.VAL_SPLIT, random_state=42
    )

    transform     = SegmentationTransform()
    train_ds      = ForestDataset(train_rgb, train_nrg, train_m, transform)
    val_ds        = ForestDataset(val_rgb,   val_nrg,   val_m, transform)

    train_loader  = DataLoader(train_ds,
                               batch_size=config.BATCH_SIZE,
                               shuffle=True,
                               num_workers=0)
    val_loader    = DataLoader(val_ds,
                               batch_size=config.BATCH_SIZE,
                               shuffle=False,
                               num_workers=0)
    return train_loader, val_loader
