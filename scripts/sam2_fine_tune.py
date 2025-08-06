# scripts/sam2_fine_tune.py
# This script is for fine-tuning the SAM2 model on the Aerial Dead Tree Segmentation dataset.

import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import argparse
import random

from datetime import datetime

import torch
import torch.nn.functional as F
from torch import optim
from sklearn.metrics import jaccard_score
from omegaconf import OmegaConf
from hydra.utils import instantiate
from tqdm import tqdm

from sam2.sam2.modeling.sam2_base import SAM2Base
from data import AerialDeadTreeSegDataModule
from utils import paths, make_logger

config_dir = os.path.join(project_root, "sam2", "sam2", "configs", "sam2.1")
checkpoints_dir = os.path.join(project_root, "sam2", "checkpoints")
out_dir = os.path.join(project_root, "checkpoints", "SAM2_finetune")

def lovasz_grad(gt_sorted):
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if gt_sorted.numel() > 1:
        prev = jaccard[:-1].clone()
        jaccard[1:] = jaccard[1:] - prev
    return jaccard

def lovasz_softmax_flat(probas, labels):
    C = probas.size(0)
    losses = []
    for c in range(C):
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
        errors = (fg - probas[c]).abs()
        errs_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        grad = lovasz_grad(fg_sorted)
        losses.append(torch.dot(errs_sorted, grad))
    return sum(losses) / len(losses)

def lovasz_softmax(probas, labels):
    B, C, H, W = probas.size()
    losses = []
    for b in range(B):
        p_flat = probas[b].view(C, -1)
        l_flat = labels[b].view(-1)
        losses.append(lovasz_softmax_flat(p_flat, l_flat))
    return sum(losses) / len(losses)

# Parameter 
parser = argparse.ArgumentParser()
parser.add_argument("--config",     default=os.path.join(config_dir, "sam2.1_hiera_b+.yaml"))
parser.add_argument("--checkpoint", default=os.path.join(checkpoints_dir, "sam2.1_hiera_base_plus.pt"))
parser.add_argument("--out_dir",    default=out_dir)
parser.add_argument("--epochs",     type=int,   default=30)
parser.add_argument("--lr",         type=float, default=1e-5)
parser.add_argument("--device",     default="cuda")
args = parser.parse_args()

cfg = OmegaConf.load(args.config)
IMAGE_SIZE = cfg.model.image_size

# Load and freeze
def load_model(cfg_path, ckpt_path, device):
    cfg   = OmegaConf.load(cfg_path)
    model = instantiate(cfg.model)  
    sd    = torch.load(ckpt_path, map_location="cpu")["model"]
    model.load_state_dict(sd, strict=False)
    model.to(device).train()
    # freeze image_encoder、PromptEncoder、MaskDecoder
    for name, p in model.named_parameters():
        if name.startswith("image_encoder.") \
           or name.startswith("sam_prompt_encoder") \
           or name.startswith("sam_mask_decoder"):
            p.requires_grad = True
        else:
            p.requires_grad = False
    return model

# Training, Validation Functions
def train_epoch(model, loader, optimizer, epoch, warmup, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Train"):
        # Handle different data formats
        if isinstance(batch, dict) and len(batch) >= 2:
            img, gt = batch["image"], batch["mask"]
        else:
            raise ValueError(f"Unexpected batch format: {type(batch)}")
        
        img, gt = img.to(device), gt.to(device)
        B = img.size(0)

        out  = model.forward_image(img)
        fpn  = out["backbone_fpn"]
        pe   = out["vision_pos_enc"]
        emb, pe_h = fpn[-1], pe[-1]

        # 50% use GT prompt
        if random.random() < 0.5:
            Hm, Wm = model.sam_prompt_encoder.mask_input_size
            gt_low = F.interpolate(gt, size=(Hm,Wm), mode="nearest")
            dense_pe = model.sam_prompt_encoder._embed_masks(gt_low)
            dense_pe = dense_pe + model.sam_prompt_encoder.get_dense_pe().to(device)
        else:
            dense_pe = model.sam_prompt_encoder.get_dense_pe().to(device)

        sparse_pe = torch.zeros((B,0,model.sam_prompt_encoder.embed_dim),
                                dtype=emb.dtype, device=emb.device)

        high_res_feats = fpn[-4:-1]

        logits, _, _, _ = model.sam_mask_decoder(
            image_embeddings=emb,
            image_pe=pe_h,
            sparse_prompt_embeddings=sparse_pe,
            dense_prompt_embeddings=dense_pe,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_feats
        )

        prob = F.interpolate(torch.sigmoid(logits),
                             size=(IMAGE_SIZE,IMAGE_SIZE),
                             mode="nearest")

        # IoU Loss
        probas = torch.cat([1-prob, prob], dim=1)    # [B,2,H,W]
        labels = gt.squeeze(1).long()               # [B,H,W]
        lovasz = lovasz_softmax(probas, labels)

        intersect = (prob * gt).sum(dim=(2,3))
        union     = (prob + gt - prob*gt).sum(dim=(2,3))
        jaccard   = 1 - ((intersect + 1e-6)/(union + 1e-6)).mean()

        loss = lovasz + jaccard

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    # LR Warm-up
    if epoch <= warmup:
        scale = epoch / warmup
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr * scale

    return total_loss / len(loader)

def valid_epoch(model, loader, device):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Valid"):
            # Handle different data formats
            if isinstance(batch, dict) and len(batch) >= 2:
                img, gt = batch["image"], batch["mask"]
            else:
                raise ValueError(f"Unexpected batch format: {type(batch)}")
            
            img = img.to(device)
            out  = model.forward_image(img)
            fpn  = out["backbone_fpn"]
            pe   = out["vision_pos_enc"]
            emb, pe_h = fpn[-1], pe[-1]

            sparse_pe = torch.zeros((img.size(0),0,model.sam_prompt_encoder.embed_dim),
                                    dtype=emb.dtype, device=emb.device)
            dense_pe  = model.sam_prompt_encoder.get_dense_pe().to(device)
            B, C, H, W = emb.size()
            if dense_pe.shape[-2:] != (H, W):
                dense_pe = F.interpolate(dense_pe, size=(H, W), mode='bilinear', align_corners=False)
            
            high_res_feats = fpn[-4:-1]

            logits, _, _, _ = model.sam_mask_decoder(
                image_embeddings=emb,
                image_pe=pe_h,
                sparse_prompt_embeddings=sparse_pe,
                dense_prompt_embeddings=dense_pe,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_feats
            )

            prob_map = F.interpolate(torch.sigmoid(logits),
                                     size=(IMAGE_SIZE,IMAGE_SIZE),
                                     mode="nearest")[0,0].cpu().numpy()
            gt_resized = F.interpolate(gt, size=(IMAGE_SIZE,IMAGE_SIZE), mode="nearest")[0,0].cpu().numpy()
            preds.append(prob_map)
            gts.append(gt_resized)

    best_iou, best_thr = 0.0, 0.5
    for thr in [i/100 for i in range(30,71,5)]:
        ious = [
            jaccard_score((g>0.5).flatten(), (p>thr).flatten())
            for p,g in zip(preds,gts)
        ]
        m = sum(ious)/len(ious)
        if m > best_iou:
            best_iou, best_thr = m, thr

    return best_iou, best_thr

def main():
    os.makedirs(args.out_dir, exist_ok=True)
    
    data_module = AerialDeadTreeSegDataModule(
        val_split=0.1, test_split=0.2, seed=42,
        modality="rgb",  # in_channels=3 for RGB
        batch_size=1,
        num_workers=int(os.cpu_count() - 2) if os.cpu_count() is not None else 0,
        target_size=IMAGE_SIZE
    )
    data_module.prepare_data()
    data_module.setup()

    tr_loader = data_module.train_dataloader()
    va_loader = data_module.val_dataloader()

    model     = load_model(args.config, args.checkpoint, args.device)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
    )

    best_iou, best_thr = 0.0, 0.5
    warmup = 5
    
    logger = make_logger(
        name="SAM2_Finetune",
        log_path=os.path.join(args.out_dir, "train.log")
    )
    logger.info(f"Start training SAM2.1 b+ with config: {args.config}")
    start_time = datetime.now()
    fmt_start = start_time.strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Training started at {fmt_start}")

    for ep in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {ep}/{args.epochs}")
        tr_loss = train_epoch(model, tr_loader, optimizer, ep, warmup, args.device)
        val_iou, val_thr = valid_epoch(model, va_loader, args.device)
        logger.info(f"Train Loss: {tr_loss:.4f} | Val IoU: {val_iou:.4f} @thr {val_thr:.2f}")

        scheduler.step(val_iou)

        if val_iou > best_iou:
            best_iou, best_thr = val_iou, val_thr
            torch.save(
                model.state_dict(),
                os.path.join(args.out_dir, "SAM2_finetune.pkl")
            )
            logger.info(f"★ Saved best model (IoU {best_iou:.4f} @thr {best_thr:.2f})")
    
    end_time = datetime.now()
    fmt_end = end_time.strftime("%Y-%m-%d %H:%M:%S")
    elapsed = (end_time - start_time).total_seconds()
    logger.info(f"Training finished at {fmt_end}, elapsed time: {elapsed:.2f} seconds")
    logger.info(f"Best IoU: {best_iou:.4f} @thr {best_thr:.2f}")

if __name__ == "__main__":
    main()
