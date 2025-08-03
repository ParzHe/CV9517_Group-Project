import os

import torch
import lightning as L
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp
from .segmentation_module import SegLitModule
from models import U2net

class U2netLitModule(SegLitModule):
    def __init__(self, in_channels=4, out_classes=1, 
                 loss1=smp.losses.JaccardLoss(mode='binary', from_logits=True),
                 loss2=smp.losses.FocalLoss(mode='binary'),
                 lr=1e-3, use_scheduler=True, **kwargs):
        model = U2net(in_ch=in_channels, out_ch=out_classes)
        super().__init__(model=model, in_channels=in_channels, loss1=loss1, loss2=loss2, lr=lr, use_scheduler=use_scheduler, **kwargs)
        self.save_hyperparameters(ignore=["encoder_weights", "out_classes", "loss1", "loss2", "lr", "use_scheduler"])