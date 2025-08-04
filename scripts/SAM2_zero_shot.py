import os
import cv2
import torch
import numpy as np
from glob import glob
from sklearn.metrics import jaccard_score
from omegaconf import OmegaConf
from hydra.utils import instantiate
from sam2.sam2_image_predictor import SAM2ImagePredictor

##
CONFIG       = "sam2/sam2.1_hiera_b+.yaml"
CHECKPOINT   = "sam2/sam2.1_hiera_base_plus.pt"
IMG_DIR      = "data/RGB_images"   
GT_DIR       = "data/masks"
OUT_DIR      = "predictions_sam2"
USE_BOX      = False   
POINT_COUNT  = 5       

def load_sam2_model(config_path, checkpoint_path, device="cuda"):
    cfg = OmegaConf.load(config_path)
    # cfg.model ， _target instance model
    model = instantiate(cfg.model)
    # weight
    ckpt = torch.load(checkpoint_path, map_location="cpu")
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

    # load SAM2 model
    sam2 = load_sam2_model(CONFIG, CHECKPOINT, device)
    predictor = SAM2ImagePredictor(sam2, device=device)

    ious = []
    for img_path in sorted(glob(os.path.join(IMG_DIR, "*.png"))):
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
        cv2.imwrite(os.path.join(OUT_DIR, name), pred_mask)

        # 5cal IoU 
        suffix = name.split("_", 1)[1]
        gt_name = f"mask_{suffix}"
        gt_path = os.path.join(GT_DIR, gt_name)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            print(f"[Warning] No GT：{gt_path}")
            continue

        iou = compute_iou(gt, pred_mask)
        print(f"{name} → IoU: {iou:.4f}")
        ious.append(iou)

# summary
    if ious:
        mean_iou = float(np.mean(ious))
        print(f"\n=== Mean IoU over {len(ious)} images: {mean_iou:.4f} ===")
    else:
        print("[Error]")

if __name__ == "__main__":
    main()
