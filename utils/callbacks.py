# utils/callbacks.py
# This script provides utility functions to create callbacks for PyTorch Lightning training and testing.
# It includes progress bars, model summaries, learning rate monitoring, early stopping, and model checkpointing.

import os
from .paths import paths
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, RichProgressBar, RichModelSummary, Timer

def callbacks(model_name, early_stop_patience, version, metric="per_image_iou/val", metric_mode="max"):
    progress_bar = RichProgressBar()
    model_sum_callback = RichModelSummary(max_depth=2)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stop_callback = EarlyStopping(
        monitor=metric,
        min_delta=0.001,
        patience=early_stop_patience,
        verbose=True,
        mode=metric_mode  # Maximize the metric
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(paths.checkpoint_dir, f"{model_name}", version),
        monitor=metric,
        filename="{epoch:02d}-{per_image_iou_val:.4f}",
        mode=metric_mode,
        save_top_k=2,
        enable_version_counter=True,
    )
    return [
        progress_bar,
        model_sum_callback,
        lr_monitor,
        early_stop_callback,
        checkpoint_callback
    ]