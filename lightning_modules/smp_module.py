import os

import torch
import lightning as L
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp
from segmentation_module import SegLitModule

class SMPLitModule(SegLitModule):
    def __init__(self, arch, encoder_name, encoder_weights="imagenet", in_channels=4, out_classes=1, lr=1e-3, use_scheduler=True, **kwargs):
        model = smp.create_model(
            arch=arch,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_classes,
            encoder_depth=5,
            **kwargs,
        )
        super().__init__(model=model, smp_model=True, lr=lr, use_scheduler=use_scheduler, **kwargs)
        self.save_hyperparameters(ignore=[encoder_weights, out_classes, lr, use_scheduler])
    