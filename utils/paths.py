# utils/paths.py
# This script defines the paths used in the project, including data directories, log directories, and checkpoint directories.

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class PathManager:
    def __init__(self):
        home_dir = os.path.expanduser("~")
        self.kaggle_datasets_dir = os.path.join(home_dir, ".cache", "kagglehub", "datasets")
        self.project_root = os.getenv("PROJECT_ROOT", "./")  # Default to current directory if not set)
        
        self.data_split_dir = os.getenv("DATA_SPLIT_DIR", "./data_splits")
        self.log_dir = os.getenv("LOG_DIR", "./logs")
        self.tensorboard_log_dir = os.getenv("TB_LOG_DIR", os.path.join(self.log_dir, "tensorboard"))
        self.checkpoint_dir = os.getenv("CHECKPOINT_DIR", "./checkpoints")

paths = PathManager()