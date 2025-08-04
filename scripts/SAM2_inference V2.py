
import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from glob import glob
from omegaconf import OmegaConf
from hydra.utils import instantiate
import matplotlib.pyplot as plt
import time  

from sam2.modeling.sam2_base import SAM2Base

# configuration
CONFIG = "sam2/sam2.1_hiera_b+.yaml"
CHECKPOINT = "SAM2_finetune/SAM2_finetune.pkl"  
RGB_DIR = "sam2/data/RGB_images"
GT_DIR = "sam2/data/masks"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THR = 0.7 

# Loading model (structure + weights)
def load_model(cfg_path, ckpt_path, device):
    cfg = OmegaConf.load(cfg_path)
    model = instantiate(cfg.model)  
    sd = torch.load(ckpt_path, map_location="cpu")

    if isinstance(sd, dict) and any(k.startswith("sam_mask_decoder") for k in sd.keys()):
        model.load_state_dict(sd, strict=False)
    else:
        raise RuntimeError("error state_dict。")
    model.to(device).eval()
    return model

# read + pre-processing
def preprocess_image(rgb_path, gt_path, image_size):
    rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    H0, W0 = rgb.shape[:2]
    rgb_resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    gt_resized  = cv2.resize(gt,  (image_size, image_size), interpolation=cv2.INTER_NEAREST)

    img_tensor = torch.from_numpy(rgb_resized.astype("float32")/255.).permute(2,0,1).unsqueeze(0)  # 1×3×H×W
    mask_tensor = torch.from_numpy((gt_resized>127).astype("float32"))[None,None,...]  # 1×1×H×W
    return img_tensor, mask_tensor, rgb, gt 

# Inference
def infer_single(model, img_tensor, dense_only=True):
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        out = model.forward_image(img_tensor)  # struct return backbone_fpn, vision_pos_enc, ...
        fpn = out["backbone_fpn"]
        pe = out["vision_pos_enc"]
        image_embeddings, image_pe = fpn[-1], pe[-1]  # (B,C,He,We), positional embedding for high-res

        B = img_tensor.size(0)
        # sparse prompt empty
        sparse_pe = torch.zeros((B,0,model.sam_prompt_encoder.embed_dim),
                                dtype=image_embeddings.dtype, device=image_embeddings.device)
        # dense prompt
        # only positional prior (zero-shot style)
        dense_pe = model.sam_prompt_encoder.get_dense_pe().to(device)

        # reuse same slice as train code
        high_res_feats = fpn[-4:-1]  

        logits, _, _, _ = model.sam_mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_pe,
            dense_prompt_embeddings=dense_pe,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_feats
        )
        prob = torch.sigmoid(logits)  # 1×1×H_feat×W_feat
        prob_upsampled = F.interpolate(prob, size=(image_embeddings.shape[-2]*16, image_embeddings.shape[-1]*16),  
                                       mode="nearest")  
        # resize same as model input
        prob_final = F.interpolate(prob, size=(image_embeddings.shape[-2]*16, image_embeddings.shape[-1]*16), mode="nearest")

        return prob.squeeze(0).squeeze(0).cpu().numpy()

# visualisation overlay
def visualize(rgb_orig, gt_orig, pred_mask_prob, thr=0.65, save_path=None):
    h, w = rgb_orig.shape[:2]
    pred_bin = (pred_mask_prob >= thr).astype(np.uint8) * 255
    pred_color = cv2.applyColorMap(pred_bin, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(rgb_orig.astype(np.uint8), 0.6, pred_color, 0.4, 0)

    # ground-truth contour
    gt_bin = (gt_orig > 127).astype(np.uint8) * 255
    contours, _ = cv2.findContours(gt_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0,255,0), 2)  

    plt.figure(figsize=(12,6))
    plt.subplot(1,3,1)
    plt.title("RGB Input"); plt.axis("off")
    plt.imshow(rgb_orig)
    plt.subplot(1,3,2)
    plt.title(f"Pred Mask (thr={thr})"); plt.axis("off")
    plt.imshow(pred_bin, cmap="gray")
    plt.subplot(1,3,3)
    plt.title("Overlay (GT)"); plt.axis("off")
    plt.imshow(overlay)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

def main():
    model = load_model(CONFIG, CHECKPOINT, DEVICE)
    rgb_paths = sorted(glob(os.path.join(RGB_DIR, "*.png")))
    # Select no N image (in 20% val set)
    N = int(len(rgb_paths)*0.85) 
    rgb_p = rgb_paths[N]
    suf = os.path.basename(rgb_p).split("_",1)[1]
    gt_p = os.path.join(GT_DIR, f"mask_{suf}")

    # preprocessing
    cfg = OmegaConf.load(CONFIG)
    IMAGE_SIZE = cfg.model.image_size
    img_tensor, gt_tensor, rgb_orig, gt_orig = preprocess_image(rgb_p, gt_p, IMAGE_SIZE)

    # inference with timing
    start_time = time.time()
    pred_prob = infer_single(model, img_tensor)  
    elapsed = time.time() - start_time
    print(f"Inference time (single image): {elapsed:.4f} seconds")

    pred_prob_resized = cv2.resize(pred_prob, (rgb_orig.shape[1], rgb_orig.shape[0]), interpolation=cv2.INTER_LINEAR)
    os.makedirs("inference_vis", exist_ok=True)
    out_path = os.path.join("inference_vis", f"vis_{os.path.basename(rgb_p)}.png")
    visualize(rgb_orig, gt_orig, pred_prob_resized, thr=THR, save_path=out_path)
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    main()
