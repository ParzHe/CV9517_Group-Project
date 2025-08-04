import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

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

from sam2.sam2.modeling.sam2_base import SAM2Base
from data import AerialDeadTreeSegDataModule
from utils import make_logger

import segmentation_models_pytorch as smp

config_dir = os.path.join(project_root, "sam2", "sam2", "configs", "sam2.1")
checkpoints_dir = os.path.join(project_root, "checkpoints", "SAM2_finetune")

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
    return model, cfg

def evaluate(model, dataloader, image_size, device, logger=None):
    if logger is None:
        logger = make_logger(name="SAM2_Evaluation", log_path=os.path.join(checkpoints_dir, "evaluation.log"))
    loader = dataloader
    preds, gts, names = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval"):
            # img: [1,3,H,W], gt_mask: numpy boolean [H,W]
            img = batch["image"].to(device)
            gt_mask = batch["mask"].to(device)
            name = batch["name"]

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
            B, C, H, W = image_embeddings.size()
            if dense_pe.shape[-2:] != (H, W):
                dense_pe = F.interpolate(dense_pe, size=(H, W), mode='bilinear', align_corners=False)

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
                size=(image_size, image_size),
                mode="nearest"
            )[0,0].cpu().numpy()

            if isinstance(gt_mask, torch.Tensor):
                # Remove batch dimension and squeeze extra dimensions if present
                gt_np = gt_mask[0].cpu().numpy()
                # Ensure gt_np is 2D
                while gt_np.ndim > 2:
                    gt_np = gt_np.squeeze()
                # Resize to match prediction size if needed
                if gt_np.shape != (image_size, image_size):
                    gt_np = cv2.resize(gt_np.astype(np.float32), (image_size, image_size), interpolation=cv2.INTER_NEAREST)
            else:
                gt_np = gt_mask[0]
                while gt_np.ndim > 2:
                    gt_np = gt_np.squeeze()
                if gt_np.shape != (image_size, image_size):
                    gt_np = cv2.resize(gt_np.astype(np.float32), (image_size, image_size), interpolation=cv2.INTER_NEAREST)

            preds.append(prob_map)
            gts.append(gt_np)
            names.append(name[0] if isinstance(name, list) else name)
    
    all_preds = np.stack(preds)
    all_gts = np.stack(gts)
    
    all_preds_tensor = torch.from_numpy(all_preds).unsqueeze(1).float()
    all_gts_tensor = torch.from_numpy(all_gts.astype(np.float32)).unsqueeze(1)

    # threshold search
    best_iou, best_thr = 0.0, 0.5
    logger.info("Searching for best IoU threshold...")
    for thr in np.linspace(0.3, 0.7, num=9):
        tp, fp, fn, tn = smp.metrics.get_stats(
            all_preds_tensor, 
            all_gts_tensor.long(), 
            mode='binary', 
            threshold=thr
        )
        
        # Calculate IoU
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        
        if iou > best_iou:
            best_iou, best_thr = iou.item(), thr
            
    logger.info(f"Best IoU = {best_iou:.4f} at threshold {best_thr:.2f}\n")

    # print per-image IoU at best_thr
    # Calculate all metrics at best threshold
    tp, fp, fn, tn = smp.metrics.get_stats(
        all_preds_tensor.long(), 
        all_gts_tensor.long(), 
        mode='binary', 
        threshold=best_thr
    )
    
    imagewise_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
    dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro-imagewise")
    f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")
    f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro-imagewise")
    precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro-imagewise")
    recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
    sensitivity = smp.metrics.sensitivity(tp, fp, fn, tn, reduction="micro-imagewise")
    specificity = smp.metrics.specificity(tp, fp, fn, tn, reduction="micro-imagewise")
    
    logger.info(f"Image-wise IoU: {imagewise_iou:.4f}")
    logger.info(f"Dataset IoU: {dataset_iou:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1 Score: {f1_score:.4f}")
    logger.info(f"F2 Score: {f2_score:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"Sensitivity: {sensitivity:.4f}")
    logger.info(f"Specificity: {specificity:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default=os.path.join(config_dir, "sam2.1_hiera_b+.yaml"))
    parser.add_argument("--checkpoint", default=os.path.join(checkpoints_dir, "SAM2_finetune.pkl"))
    parser.add_argument("--image_size", type=int,   default=1024)
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()
    
    data_module = AerialDeadTreeSegDataModule(
        val_split=0.1, test_split=0.2, seed=42,
        modality="rgb",  # in_channels=3 for RGB
        batch_size=1,
        num_workers=int(os.cpu_count() - 2) if os.cpu_count() is not None else 0,
        target_size=args.image_size
    )
    data_module.prepare_data()
    data_module.setup()

    test_dataloader = data_module.test_dataloader()
    model, _   = load_model(args.config, args.checkpoint, args.device)
    evaluate(model, test_dataloader, args.image_size, args.device)

if __name__ == "__main__":
    main()

