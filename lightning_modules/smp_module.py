# lightning_modules/smp_module.py
# This module inherits from SegLitModule and is used for training Segmentation Models PyTorch (SMP) models with PyTorch Lightning.

import lightning as L
import segmentation_models_pytorch as smp
from .segmentation_module import SegLitModule

class SMPLitModule(SegLitModule):
    def __init__(self, arch, encoder_name, encoder_weights="imagenet", in_channels=4, out_classes=1, 
                 loss1=smp.losses.JaccardLoss(mode='binary', from_logits=True),
                 loss2=smp.losses.FocalLoss(mode='binary'),
                 lr=1e-3, use_scheduler=True, target_size=256, **kwargs):
        model = smp.create_model(
            arch=arch if arch != "Unet_scse" else "Unet",
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            decoder_attention_type="scse" if arch == "Unet_scse" else None,
            in_channels=in_channels,
            classes=out_classes,
            dynamic_img_size=True,
        )
        super().__init__(model=model, in_channels=in_channels, loss1=loss1, loss2=loss2, lr=lr, use_scheduler=use_scheduler, target_size=target_size, **kwargs)
        self.save_hyperparameters(ignore=["encoder_weights", "out_classes", "loss1", "loss2", "lr", "use_scheduler"])