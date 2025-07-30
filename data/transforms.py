# data/transforms.py

import albumentations as A
import numpy as np

def identity_image(x, **kwargs):
        return x.astype('float32')

def binarize_mask(x, **kwargs):
    return (x > 0.5).astype('float32')

class SegmentationTransform:
    def __init__(self, 
            target_size=224, 
            mode="train", 
            modality="rgb",
        ):
        
        assert mode in ["train", "val", "test"], "Mode must be one of 'train', 'val', or 'test'."
        assert modality in ["merged", "rgb", "nrg"], "Modality must be one of 'merged', 'rgb', or 'nrg'."
        
        target_size = (target_size, target_size) if isinstance(target_size, int) else target_size
        self.mode = mode
        self.modality = modality
        
        transforms = []

        if mode == "train":
            transforms.extend([
                A.SmallestMaxSize(max_size=target_size[0] * 2, p=1.0),
                A.CropNonEmptyMaskIfExists(height=target_size[0], width=target_size[1], p=1.0),
                A.SquareSymmetry(p=1.0),  # Replaces Horizontal/Vertical Flips
                A.GaussNoise(std_range=(0.02, 0.06), mean_range=(0.01,0.03), p=0.3),
            ])
        else:
            transforms.extend([
                A.SmallestMaxSize(max_size=target_size[0] * 2, p=1.0),
                A.CenterCrop(height=target_size[0] * 2, width=target_size[1] * 2, p=1.0),
            ])
            
        if modality == "merged":
            transforms.append(
                A.Normalize(mean=(0.485, 0.456, 0.406, 0.5), std=(0.229, 0.224, 0.225, 0.2), p=1.0)
            )
        else:
            transforms.append(
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0)
            )

        transforms.extend([
            A.Lambda(
                image=identity_image,
                mask=binarize_mask,
            ),
            A.ToTensorV2(),
        ])
        
        self.transform = A.Compose(transforms)

    def __call__(self, img, mask):
        transformed = self.transform(image=np.array(img), mask=np.array(mask))
        img, mask = transformed['image'], transformed['mask']
        
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)  # Ensure mask has a channel dimension
            
        return img, mask
        