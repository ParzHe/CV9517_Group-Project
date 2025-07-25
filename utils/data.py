import os
import kagglehub

def download_dataset(handle: str = "meteahishali/aerial-imagery-for-standing-dead-tree-segmentation") -> str:
    """Download the dataset from Kaggle.
    Args:
        handle (str): The Kaggle dataset handle, e.g., "meteahishali/aerial-imagery-for-standing-dead-tree-segmentation".
    Returns:
        str: Path to the downloaded dataset files.
    """
    
    path = kagglehub.dataset_download(handle)
    path = os.path.join(path, "USA_segmentation")
    
    # List all folder names in the dataset directory
    dataset_folders_expected = ["RGB_images", "NRG_images", "masks"]
    folder_names = os.listdir(path)
    
    if not all(folder in folder_names for folder in dataset_folders_expected):
        raise RuntimeError(f"Expected folders {dataset_folders_expected} not found in {path}. Check your dataset download.")

    return path

if __name__ == "__main__":
    # Example usage
    path = download_dataset()
    print("Path to dataset files:", path)