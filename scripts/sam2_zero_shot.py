# scripts/sam2_zero_shot.py
# This script is for zero-shot inference using the SAM2 model on the whole Aerial Dead Tree Segmentation dataset.

import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import cv2
import torch
import tqdm
import numpy as np
from glob import glob
from sklearn.metrics import jaccard_score
from omegaconf import OmegaConf
from hydra.utils import instantiate
from sam2.sam2.sam2_image_predictor import SAM2ImagePredictor

from data import AerialDeadTreeSegDataModule
from utils import make_logger

from rich import print

import segmentation_models_pytorch as smp

data_module = AerialDeadTreeSegDataModule()
data_module.prepare_data()

##
SAM2_dir = os.path.join(project_root, "sam2")
CONFIG       = os.path.join(SAM2_dir, "sam2", "configs", "sam2.1", "sam2.1_hiera_b+.yaml")
CHECKPOINT   = os.path.join(SAM2_dir, "checkpoints", "sam2.1_hiera_base_plus.pt")
IMG_DIR      = os.path.join(data_module.dataset_path, "RGB_images")
GT_DIR       = os.path.join(data_module.dataset_path, "masks")
OUT_DIR      = os.path.join(project_root, "outputs", "SAM2_zs_inference")
USE_BOX      = False   
POINT_COUNT  = 5       

def load_sam2_model(config_path, checkpoint_path, device="cuda"):
    cfg = OmegaConf.load(config_path)
    # cfg.model ， _target instance model
    model = instantiate(cfg.model)
    # weight
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"], strict=False)
    # front inference
    model.to(device).eval()
    return model

def compute_iou(gt_mask, pred_mask):
    gt = (gt_mask.flatten() > 127).astype(np.uint8)
    pr = (pred_mask.flatten() > 127).astype(np.uint8)
    return jaccard_score(gt, pr)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log = make_logger("SAM2_zero_shot", log_path=os.path.join(OUT_DIR, "evaluation.log"))
    
    log.info(f"Zero-shot inference with SAM2")

    # load SAM2 model
    sam2 = load_sam2_model(CONFIG, CHECKPOINT, device)
    predictor = SAM2ImagePredictor(sam2, device=device)

    pred_masks = []
    gt_masks = []
    img_paths = sorted(glob(os.path.join(IMG_DIR, "*.png")))
    for img_path in tqdm.tqdm(img_paths, desc="Predicting images"):
        name = os.path.basename(img_path)

        # setting predictor（OpenCV read BGR → RGB）
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(img_rgb)

        h, w = img_rgb.shape[:2]
        #  prompt and pred
        if USE_BOX:
        
            box = np.array([0, 0, w, h])  # [x0,y0,x1,y1]
            masks, scores, logits = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                mask_input=None,
                multimask_output=False
            )
        else:
            # random notice points (x,y)
            ys = np.random.randint(0, h, size=POINT_COUNT)
            xs = np.random.randint(0, w, size=POINT_COUNT)
            pts = np.stack([xs, ys], axis=1)
            labels = np.ones(POINT_COUNT, dtype=int)
            masks, scores, logits = predictor.predict(
                point_coords=pts,
                point_labels=labels,
                box=None,
                mask_input=None,
                multimask_output=False
            )

        # mask
        pred_mask = (masks[0] * 255).astype(np.uint8)

        # 5cal IoU 
        suffix = name.split("_", 1)[1]
        gt_name = f"mask_{suffix}"
        gt_path = os.path.join(GT_DIR, gt_name)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            print(f"[Warning] No GT：{gt_path}")
            continue
        
        pred_masks.append(pred_mask)
        gt_masks.append(gt)
        
    # Find maximum dimensions to resize all masks to the same size
    max_h = max(mask.shape[0] for mask in pred_masks + gt_masks)
    max_w = max(mask.shape[1] for mask in pred_masks + gt_masks)
    
    # Resize all masks to the same size
    pred_masks_resized = []
    gt_masks_resized = []
    
    for pred_mask, gt_mask in zip(pred_masks, gt_masks):
        pred_resized = cv2.resize(pred_mask, (max_w, max_h), interpolation=cv2.INTER_NEAREST)
        gt_resized = cv2.resize(gt_mask, (max_w, max_h), interpolation=cv2.INTER_NEAREST)
        pred_masks_resized.append(pred_resized)
        gt_masks_resized.append(gt_resized)
    
    # Convert lists to tensors for smp metrics
    pred_masks_tensor = torch.from_numpy(np.stack(pred_masks_resized)).float() / 255.0  # Normalize to [0,1]
    gt_masks_tensor = torch.from_numpy(np.stack(gt_masks_resized)).long() // 255  # Convert to long type for smp metrics

    tp, fp, fn, tn = smp.metrics.get_stats(
        pred_masks_tensor, gt_masks_tensor, mode="binary", threshold=0.5
    )
    
    imagewise_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
    dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")
    f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro-imagewise")
    accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro-imagewise")
    precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro-imagewise")
    specificity = smp.metrics.specificity(tp, fp, fn, tn, reduction="micro-imagewise")
    sensitivity = smp.metrics.sensitivity(tp, fp, fn, tn, reduction="micro-imagewise")
    
    log.info(f"Image-wise IoU: {imagewise_iou:.4f}")
    log.info(f"Dataset IoU: {dataset_iou:.4f}")
    log.info(f"F1 Score: {f1_score:.4f}")
    log.info(f"F2 Score: {f2_score:.4f}")
    log.info(f"Accuracy: {accuracy:.4f}")
    log.info(f"Precision: {precision:.4f}")
    log.info(f"Specificity: {specificity:.4f}")
    log.info(f"Sensitivity: {sensitivity:.4f}")
    
    print("[green]Zero-shot inference completed successfully![/green]")
    print(f"Results saved to: [bold]{OUT_DIR}[/bold]")

if __name__ == "__main__":
    main()
