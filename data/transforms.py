import albumentations as A
import numpy as np

def identity_image(x, **kwargs):
        return x

def binarize_mask(x, **kwargs):
    return (x > 0.5).astype('float32')

class SegmentationTransform:
    def __init__(self, 
            target_size=224, 
            mode="train", 
            modality="rgb",
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ):
        
        TARGET_SIZE = (target_size, target_size) if isinstance(target_size, int) else target_size
        self.mode = mode
        self.modality = modality
        
        assert mode in ["train", "val", "test"], "Mode must be one of 'train', 'val', or 'test'."
        assert modality in ["merged", "rgb", "nrg"], "Modality must be one of 'merged', 'rgb', or 'nrg'."
        
        transforms = []
        
        transforms.extend([
            A.LongestMaxSize(max_size=TARGET_SIZE[0], p=1.0),
            A.PadIfNeeded(min_height=TARGET_SIZE[0], min_width=TARGET_SIZE[1], p=1.0),
        ])

        if mode == "train":
            transforms.extend([
                A.SquareSymmetry(p=1.0),  # Replaces Horizontal/Vertical Flips
                A.ISONoise(
                    color_shift=(0.005, 0.03),
                    intensity=(0.02, 0.1),
                    p=0.2
                ),
            ])
        
        transforms.extend([
            A.Normalize(mean=mean, std=std),
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
        