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
            encoder_depth=5,
            **kwargs,
        )
        super().__init__(model=model, loss1=loss1, loss2=loss2, lr=lr, use_scheduler=use_scheduler, **kwargs)
        self.save_hyperparameters(ignore=["encoder_weights", "out_classes", "loss1", "loss2", "lr", "use_scheduler"])

    def on_fit_start(self):
        example_input = torch.randn(1, self.hparams.in_channels, 224, 224)
        self.logger.log_graph(self, example_input)

def freeze_smp_encoder_layers(model, encoder_name, layers_range=(1, 3)):
    layer_names = []
    for name, module in model.model.encoder.named_children():
        # print(f"Layer: {name}, Module: {module}")
        layer_names.append(name)

    if str(encoder_name).startswith("mit_b"):
        layers_list = [
            ['patch_embed1', 'block1', 'norm1'],
            ['patch_embed2', 'block2', 'norm2'],
            ['patch_embed3', 'block3', 'norm3'],
            ['patch_embed4', 'block4', 'norm4'],
        ]
        for name, module in model.model.encoder.named_children():
            if name in layers_list[layers_range[0]:layers_range[1]]:
                print(f"Freezing layer: {name}")
                for param in module.parameters():
                    param.requires_grad = False
    elif str(encoder_name) == "se_resnet50":
        for name, module in model.model.encoder.named_children():
            if name in layer_names[layers_range[0]:layers_range[1]]:
                print(f"Freezing layer: {name}")
                for param in module.parameters():
                    param.requires_grad = False