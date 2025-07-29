# data/utils.py

import kagglehub
import os
from rich import print
from utils import paths

def download_dataset(handle: str = "meteahishali/aerial-imagery-for-standing-dead-tree-segmentation") -> str:
    """Download the dataset from Kaggle.
    Args:
        handle (str): The Kaggle dataset handle, e.g., "meteahishali/aerial-imagery-for-standing-dead-tree-segmentation".
    Returns:
        str: Path to the downloaded dataset files.
    """
    
    kaggle_datasets_dir = paths.kaggle_datasets_dir
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

if __name__ == "__main__":
    # Example usage
    dataset_path = download_dataset()
    print(f"Dataset downloaded to: {dataset_path}")