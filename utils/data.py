import os
import kagglehub
import torch
import glob
import numpy as np
from PIL import Image

import albumentations as A
from rich import print
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule

def identity_image(x, **kwargs):
        return x

def binarize_mask(x, **kwargs):
    return (x > 0.5).astype('float32')

def download_dataset(handle: str = "meteahishali/aerial-imagery-for-standing-dead-tree-segmentation") -> str:
    """Download the dataset from Kaggle.
    Args:
        handle (str): The Kaggle dataset handle, e.g., "meteahishali/aerial-imagery-for-standing-dead-tree-segmentation".
    Returns:
        str: Path to the downloaded dataset files.
    """
    
    home_path = os.path.expanduser("~")
    kaggle_datasets_dir = os.path.join(home_path, ".cache", "kagglehub", "datasets")
    dataset_folder_dir = os.path.join(kaggle_datasets_dir, handle.split("/")[0], handle.split("/")[1], "versions", "1", )

    if os.path.exists(dataset_folder_dir):
        path = dataset_folder_dir
    else:
        print(f"[white]Downloading dataset from Kaggle: {handle}...[/white]")
        path = kagglehub.dataset_download(handle)
        print(f"[green]Dataset successfully downloaded to {path}[/green]")
    
    path = os.path.join(path, "USA_segmentation")
    
    # List all folder names in the dataset directory
    dataset_folders_expected = ["RGB_images", "NRG_images", "masks"]
    folder_names = os.listdir(path)
    
    if not all(folder in folder_names for folder in dataset_folders_expected):
        raise RuntimeError(f"Expected folders {dataset_folders_expected} not found in {path}. Check your dataset download.")

    return path

class ForestDataset(Dataset):
    def __init__(self, rgb_paths, nrg_paths, mask_paths, transform=None, pattern="merged"):
        self.rgb_paths  = rgb_paths
        self.nrg_paths  = nrg_paths
        self.mask_paths = mask_paths
        self.transform  = transform
        self.pattern    = pattern
        
        if self.pattern not in ["rgb", "nrg", "merged"]:
            print(f"[yellow]Warning: Unknown pattern '{self.pattern}'. Defaulting to 'merged'.[/yellow]")
            self.pattern = "merged" # Default to merged if not specified

        self._load_strategies = {
            "rgb": self._load_rgb,
            "nrg": self._load_nrg,
            "merged": self._load_merged
        }

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        # Load mask
        mask = Image.open(self.mask_paths[idx]).convert("L")
        
        # Load image based on the specified pattern
        img = self._load_strategies[self.pattern](idx)

        # Apply transformations
        if self.transform:
            img, mask = self.transform(img, mask)
        
        return {"image": img, "mask": mask}
        
    def _load_rgb(self, idx):
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

class SegmentationTransform:
    def __init__(self, target_size=256, in_channels=3, mode="train"):
        self.mode = mode
        if mode not in ["train", "val"]:
            raise ValueError(f"Invalid mode: {mode}. Expected 'train' or 'val'.")

        TARGET_SIZE = (target_size, target_size) if isinstance(target_size, int) else target_size
        
        # Get normalization parameters based on input channels
        mean, std = self._get_normalization_params(in_channels)
        
        transforms = [] 
        
        if mode == "train":
            transforms.extend([
                A.SmallestMaxSize(max_size=TARGET_SIZE[0] * 2, p=1.0),
                A.RandomCrop(height=TARGET_SIZE[0], width=TARGET_SIZE[1], p=1.0),
                A.SquareSymmetry(p=1.0),  # Replaces Horizontal/Vertical Flips
                # A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(std_range=(0.1, 0.2), p=0.2),
            ])
        else:
            transforms.extend([
                A.SmallestMaxSize(max_size=TARGET_SIZE[0] * 2, p=1.0),
                A.CenterCrop(height=TARGET_SIZE[0] * 2, width=TARGET_SIZE[1] * 2, p=1.0),
            ])
        
        transforms.extend([
            A.Normalize(mean=mean, std=std),
            A.Lambda(
                image=identity_image,
                mask=binarize_mask,
            ),
            A.ToTensorV2(),
        ])
        
        self.transform = A.Compose(transforms)
    
    def _get_normalization_params(self, in_channels):
        if in_channels == 3:
            return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        elif in_channels == 4:
            return (0.485, 0.456, 0.406, 0.5), (0.229, 0.224, 0.225, 0.2)
        else:
            raise ValueError(f"Unsupported number of input channels: {in_channels}")
        
    def __call__(self, img, mask):
        transformed = self.transform(image=np.array(img), mask=np.array(mask))
        img, mask = transformed['image'], transformed['mask']
        
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)  # Ensure mask has a channel dimension
            
        return img, mask

class USASegmentationDataModule(LightningDataModule):
    def __init__(self, val_split=0.1, test_split=0.2, seed=42, pattern="rgb", 
                 batch_size=32, num_workers=0, target_size=256):
        super().__init__()
        self.save_hyperparameters()
        
        # Calculate splits
        self.train_split = 1 - val_split - test_split
        self.in_channels = 3 if pattern in ["rgb", "nrg"] else 4

    def prepare_data(self):
        """Download dataset if not already present"""
        self.dataset_path = download_dataset()

    def setup(self, stage=None):
        """Setup dataset"""
        # Get all file paths
        rgb_dir = os.path.join(self.dataset_path, "RGB_images")
        nrg_dir = os.path.join(self.dataset_path, "NRG_images")
        mask_dir = os.path.join(self.dataset_path, "masks")
        
        rgb_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
        nrg_paths = sorted(glob.glob(os.path.join(nrg_dir, "*.png")))
        mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        
        # Split data
        splits = self._split_data(rgb_paths, nrg_paths, mask_paths)
        
        # Based on the stage, we create the corresponding datasets
        if stage in ("fit", None):
            self._train_dataset = self._create_dataset(splits["train"], mode="train")
            self._val_dataset = self._create_dataset(splits["val"], mode="val")
        
        if stage in ("validate", None) and not hasattr(self, '_val_dataset'):
            self._val_dataset = self._create_dataset(splits["val"], mode="val")
        
        if stage in ("test", None):
            self._test_dataset = self._create_dataset(splits["test"], mode="val")

    def _split_data(self, rgb_paths, nrg_paths, mask_paths):
        """Split dataset"""
        total_len = len(rgb_paths)
        train_len = int(self.train_split * total_len)
        val_len = int(self.hparams.val_split * total_len)

        # Generate random indices
        generator = torch.Generator().manual_seed(self.hparams.seed)
        indices = torch.randperm(total_len, generator=generator).tolist()
        
        # Split indices
        train_indices = indices[:train_len]
        val_indices = indices[train_len:train_len + val_len]
        test_indices = indices[train_len + val_len:]
        
        return {
            "train": ([rgb_paths[i] for i in train_indices],
                     [nrg_paths[i] for i in train_indices],
                     [mask_paths[i] for i in train_indices]),
            "val": ([rgb_paths[i] for i in val_indices],
                   [nrg_paths[i] for i in val_indices],
                   [mask_paths[i] for i in val_indices]),
            "test": ([rgb_paths[i] for i in test_indices],
                    [nrg_paths[i] for i in test_indices],
                    [mask_paths[i] for i in test_indices])
        }

    def _create_dataset(self, paths_tuple, mode):
        """Create dataset"""
        rgb_paths, nrg_paths, mask_paths = paths_tuple
        transform = SegmentationTransform(
            target_size=self.hparams.target_size,
            in_channels=self.in_channels,
            mode=mode
        )
        return ForestDataset(rgb_paths, nrg_paths, mask_paths, transform, self.hparams.pattern)

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset, 
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self._test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers
        )
    
    def on_exception(self, exception):
        # clean up state after the trainer faced an exception
        print(f"[red]Exception during data loading: {exception}[/red]")
        self._train_dataset = None

    def teardown(self, *args, **kwargs):
        # clean up state after the trainer stops, delete files...
        # called on every process in DDP
        print("[blue]Teardown called, cleaning up datasets...[/blue]")
        self._train_dataset = None

if __name__ == "__main__":
    path = download_dataset()
    print(f"[green]Dataset downloaded to {path}[/green]")