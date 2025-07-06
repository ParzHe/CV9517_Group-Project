import kagglehub

def download_dataset(handle: str) -> str:
    """Download the dataset from Kaggle.
    Args:
        handle (str): The Kaggle dataset handle, e.g., "meteahishali/aerial-imagery-for-standing-dead-tree-segmentation".
    Returns:
        str: Path to the downloaded dataset files.
    """
    
    path = kagglehub.dataset_download(handle)
    
    return path
    
if __name__ == "__main__":
    # Example usage
    path = download_dataset("meteahishali/aerial-imagery-for-standing-dead-tree-segmentation")
    print("Path to dataset files:", path)