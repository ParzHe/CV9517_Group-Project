import os
import argparse
from glob import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
from hydra.utils import instantiate
from tqdm import tqdm

from sam2.modeling.sam2_base import SAM2Base

# ImageNet normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class EvalDataset(Dataset):
    def __init__(self, rgb_dir, gt_dir, image_size):
        self.rgb_paths  = sorted(glob(os.path.join(rgb_dir, "*.png")))
        self.gt_dir     = gt_dir
        self.image_size = image_size

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb_path = self.rgb_paths[idx]
        suffix   = os.path.basename(rgb_path).split("_",1)[1]
        gt_path  = os.path.join(self.gt_dir, f"mask_{suffix}")

        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        gt  = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        gt  = cv2.resize(gt,  (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        # normalize to [0,1], then ImageNet standardize
        img = rgb.astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        img_tensor = torch.from_numpy(img).permute(2,0,1)

        # return binary mask as numpy
        gt_mask = gt > 127

        return img_tensor, gt_mask, os.path.basename(rgb_path)

def load_model(config_path, checkpoint_path, device):
    cfg = OmegaConf.load(config_path)
    model = instantiate(cfg.model)  # type: SAM2Base
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state = ckpt["model"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model

def evaluate(model, dataset, device):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    preds, gts, names = [], [], []

    with torch.no_grad():
        for img, gt_mask, name in tqdm(loader, desc="Eval"):
            # img: [1,3,H,W], gt_mask: numpy boolean [H,W]
            img = img.to(device)

            out = model.forward_image(img)
            fpn_feats = out["backbone_fpn"]
            image_embeddings = fpn_feats[-1]
            image_pe         = out["vision_pos_enc"][-1]

            # zero-shot prompt
            sparse_pe = torch.zeros(
                (img.size(0), 0, model.sam_prompt_encoder.embed_dim),
                dtype=image_embeddings.dtype, device=device
            )
            dense_pe = model.sam_prompt_encoder.get_dense_pe().to(device)

            high_res_feats = fpn_feats[-4:-1]

            masks_low_logits, _, _, _ = model.sam_mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_pe,
                dense_prompt_embeddings=dense_pe,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_feats
            )

            prob_map = F.interpolate(
                torch.sigmoid(masks_low_logits),
                size=(dataset.image_size, dataset.image_size),
                mode="nearest"
            )[0,0].cpu().numpy()

            preds.append(prob_map)
            gts.append(gt_mask[0] if isinstance(gt_mask, np.ndarray) else gt_mask.numpy()[0])
            names.append(name[0] if isinstance(name, list) else name)

    # threshold search
    best_iou, best_thr = 0.0, 0.5
    for thr in np.linspace(0.3, 0.7, num=9):
        ious = []
        for p_map, g_map in zip(preds, gts):
            p_bin = p_map > thr
            g_bin = g_map
            if not g_bin.any() and not p_bin.any():
                iou = 1.0
            else:
                inter = (p_bin & g_bin).sum()
                union = (p_bin | g_bin).sum()
                iou = float(inter) / float(union) if union > 0 else 0.0
            ious.append(iou)
        m = float(np.mean(ious))
        if m > best_iou:
            best_iou, best_thr = m, thr

    print(f"Best IoU = {best_iou:.4f} at threshold {best_thr:.2f}\n")

    # print per-image IoU at best_thr
    print("Per-image IoU:")
    for name, p_map, g_map in zip(names, preds, gts):
        p_bin = p_map > best_thr
        g_bin = g_map
        if not g_bin.any() and not p_bin.any():
            iou_i = 1.0
        else:
            inter = (p_bin & g_bin).sum()
            union = (p_bin | g_bin).sum()
            iou_i = float(inter) / float(union) if union > 0 else 0.0
        print(f"{name}: {iou_i:.4f}")

    return best_iou, best_thr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="sam2/sam2.1_hiera_b+.yaml")
    parser.add_argument("--checkpoint", default="SAM2_finetune/SAM2_finetune.pkl")
    parser.add_argument("--rgb_dir",    default="sam2/data/RGB_images")
    parser.add_argument("--gt_dir",     default="sam2/data/masks")
    parser.add_argument("--image_size", type=int,   default=1024)
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()

    dataset = EvalDataset(args.rgb_dir, args.gt_dir, args.image_size)
    model   = load_model(args.config, args.checkpoint, args.device)
    evaluate(model, dataset, args.device)

if __name__ == "__main__":
    main()

