import segmentation_models_pytorch as smp
from rich import print
from lightning_modules import SMPLitModule

def modes_list():
    arch_list = smp.__all__.copy()  # List of available architectures in segmentation_models_pytorch
    arch_list.remove("datasets")  # Remove datasets as it's not an architecture
    arch_list.remove("encoders")  # Remove encoders as it's not an architecture
    arch_list.remove("decoders")  # Remove decoders as it's not an architecture
    arch_list.remove("losses")  # Remove losses as it's not an architecture
    arch_list.remove("metrics")  # Remove metrics as it's not an architecture
    arch_list.remove("DPT")  # DPT is not supported in this context
    arch_list.remove("from_pretrained")  # Remove from_pretrained as it's not an architecture
    arch_list.remove("create_model")  # Remove create_model as it's not an architecture
    arch_list.remove("__version__")  # Remove version info
    return arch_list

def encoders_list(model_name:str, only_available = "all"):
    encoder_list = []
    
    if model_name.upper() == "DPT":
        # Future work for DPT encoders
        if only_available == "all":
            encoder_list = [
                "tu-vit_small_patch16_224.augreg_in21k_ft_in1k",
                "tu-swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
                "tu-swinv2_tiny_window8_256.ms_in1k",
            ]
        else:
            encoder_list = [only_available]
    elif model_name.upper() == "UNETPLUSPLUS":
        if only_available == "all":
            encoder_list = [
                "resnet50",
                "resnext50_32x4d",
                "se_resnet50",
                "se_resnext50_32x4d",
                "densenet161",
                "efficientnet-b5",
            ]
        else:
            encoder_list = [only_available]
    else:
        if only_available == "all":
            encoder_list = [
                "resnet50",
                "resnext50_32x4d",
                "se_resnet50",
                "se_resnext50_32x4d",
                "densenet161",
                "efficientnet-b5",
                "mit_b2",
            ]
        else:
            encoder_list = [only_available]
        
    return encoder_list

class FreezeSMPEncoderUtils:
    def __init__(self):
        self._model = None
        self.encoder_name = None
        self._layers_range = (0, 1)
        self._frozen_layers = []

    def __call__(self, model, encoder_name: str, layers_range=(0, 1)):
        self._model = model
        self.encoder_name = encoder_name
        start, end = layers_range
        if start < 0 or end < start:
            raise ValueError("Invalid layers range. Must be within the range of available layers.")
        if start == end:
            print("[yellow]No layers to freeze. The model will train all layers.[/yellow]")
            return
        self._layers_range = layers_range
        # Dispatch to specific freeze logic
        fn = self._get_freeze_function(encoder_name)
        fn()

    def _get_freeze_function(self, name: str):
        mapping = {
            'resnet': self.resnet_freeze,
            'resnext': self.resnet_freeze,
            'se_resnet': self.se_resnet_freeze,
            'se_resnext': self.se_resnet_freeze,
            'densenet': self.densenet_freeze,
            'efficientnet': self.efficientnet_freeze,
            'mit_b': self.mit_b_freeze,
            'tu-vit': self.vit_freeze,
            'tu-swin': self.swin_freeze,
            'tu-swinv2': self.swin_freeze,
        }
        for prefix, fn in mapping.items():
            if name.startswith(prefix): return fn
        raise ValueError(f"Encoder {name} is not supported for freezing layers.")

    def _resolve_module(self, root, path: str):
        # support nested attributes and indexed module lists: e.g. '_blocks.3'
        parts = path.split('.')
        module = root
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module

    def _freeze_groups(self, root, groups: list[str] | list[list[str]]):
        start, end = self._layers_range
        flat = [g if isinstance(g, list) else [g] for g in groups]
        selected = [layer for grp in flat[start:end] for layer in grp]
        for name in selected:
            module = self._resolve_module(root, name)
            self._frozen_layers.append(f"{name}")
            for param in module.parameters(): param.requires_grad = False
        print(f"[bold]Frozen layers for {self.encoder_name}: {self._frozen_layers}[/bold]")

    def resnet_freeze(self):
        groups = [
            ['conv1','bn1','relu','maxpool'],
            ['layer1'], ['layer2'], ['layer3'], ['layer4']
        ]
        self._freeze_groups(self._model.model.encoder, groups)

    def se_resnet_freeze(self):
        groups = [
            ['layer0'], ['layer1'], ['layer2'], ['layer3'], ['layer4'], ['layer0_pool']
        ]
        self._freeze_groups(self._model.model.encoder, groups)

    def densenet_freeze(self):
        groups = [
            ['features.conv0','features.norm0','features.relu0','features.pool0'],
            ['features.denseblock1','features.transition1'],
            ['features.denseblock2','features.transition2'],
            ['features.denseblock3','features.transition3'],
            ['features.denseblock4','features.norm5'],
        ]
        self._freeze_groups(self._model.model.encoder, groups)

    def efficientnet_freeze(self):
        enc = self._model.model.encoder
        # define groups by attribute strings
        groups = [
            ['_conv_stem','_bn0'],
            *[[f'_blocks.{i}' for i in range(0,3)]],
            *[[f'_blocks.{i}' for i in range(3,8)]],
            *[[f'_blocks.{i}' for i in range(8,9)]],
            *[[f'_blocks.{i}' for i in range(9,13)]],
            *[[f'_blocks.{i}' for i in range(13,14)]],
            *[[f'_blocks.{i}' for i in range(14,20)]],
            *[[f'_blocks.{i}' for i in range(20,21)]],
            *[[f'_blocks.{i}' for i in range(21,27)]],
            *[[f'_blocks.{i}' for i in range(27,28)]],
            *[[f'_blocks.{i}' for i in range(28,36)]],
            *[[f'_blocks.{i}' for i in range(36,37)]],
            *[[f'_blocks.{i}' for i in range(37,39)]],
            ['_conv_head','_bn1'],
            ['_avg_pooling','_dropout','_swish'],
        ]
        self._freeze_groups(enc, groups)

    def mit_b_freeze(self):
        groups = [
            ['patch_embed.proj','norm'],
            ['block1','norm1'], ['block2','norm2'], ['block3','norm3'], ['block4','norm4'],
        ]
        self._freeze_groups(self._model.model.encoder, groups)

    def vit_freeze(self):
        enc = self._model.model.encoder.model
        groups = [['patch_embed.proj'], *[[f'blocks.{i}' ] for i in range(12)], ['norm']]
        self._freeze_groups(enc, groups)

    def swin_freeze(self):
        enc = self._model.model.encoder.model
        groups = [['patch_embed.proj'], *[[f'layers.{i}' ] for i in range(len(enc.layers))], ['norm']]
        self._freeze_groups(enc, groups)

if __name__ == "__main__":
    print("Available encoders:")
    smp_encoders = smp.encoders.get_encoder_names()
    for encoder in smp_encoders:
        print(f"- {encoder}")