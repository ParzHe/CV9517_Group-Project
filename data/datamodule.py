from lightning import LightningDataModule
from .utils import download_dataset
import os
import glob
import torch
from .dataset import AerialDeadTreeSegDataset
from .transforms import SegmentationTransform
from torch.utils.data import DataLoader

class AerialDeadTreeSegDataModule(LightningDataModule):
    def __init__(self, val_split=0.1, test_split=0.2, seed=42, pattern="rgb", 
                 batch_size=32, num_workers=0, target_size=224, **kwargs):
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
        return AerialDeadTreeSegDataset(rgb_paths, nrg_paths, mask_paths, transform, self.hparams.pattern)

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