import os

import torch
import lightning as L
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp
from .segmentation_module import SegLitModule

class SMPLitModule(SegLitModule):
    def __init__(self, arch, encoder_name, encoder_weights="imagenet", in_channels=4, out_classes=1, 
                 loss1=smp.losses.JaccardLoss(mode='binary', from_logits=True),
                 loss2=smp.losses.FocalLoss(mode='binary'),
                 lr=1e-3, use_scheduler=True, **kwargs):
        model = smp.create_model(
            arch=arch,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_classes,
            dynamic_img_size=True,
        )
        super().__init__(model=model, in_channels=in_channels, loss1=loss1, loss2=loss2, lr=lr, use_scheduler=use_scheduler, **kwargs)
        self.save_hyperparameters(ignore=["encoder_weights", "out_classes", "loss1", "loss2", "lr", "use_scheduler"])