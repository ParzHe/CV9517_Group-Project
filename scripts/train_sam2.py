import os
import argparse
import random
from glob import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from sklearn.metrics import jaccard_score
from omegaconf import OmegaConf
from hydra.utils import instantiate
from tqdm import tqdm

from sam2.modeling.sam2_base import SAM2Base

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
parser.add_argument("--config",     default="sam2/sam2.1_hiera_b+.yaml")
parser.add_argument("--checkpoint", default="sam2/sam2.1_hiera_base_plus.pt")
parser.add_argument("--rgb_dir",    default="sam2/data/RGB_images")
parser.add_argument("--gt_dir",     default="sam2/data/masks")
parser.add_argument("--out_dir",    default="SAM2_finetune")
parser.add_argument("--epochs",     type=int,   default=30)
parser.add_argument("--lr",         type=float, default=1e-5)
parser.add_argument("--split",      type=float, default=0.8, help="train/val")
parser.add_argument("--device",     default="cuda")
args = parser.parse_args()

cfg = OmegaConf.load(args.config)
IMAGE_SIZE = cfg.model.image_size  

# Dataset
class TreeSegDataset(Dataset):
    def __init__(self, rgb_dir, gt_dir, image_size, augment=False):
        self.rgb_paths  = sorted(glob(os.path.join(rgb_dir, "*.png")))
        self.gt_dir     = gt_dir
        self.image_size = image_size
        self.augment    = augment

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb_p = self.rgb_paths[idx]
        suf   = os.path.basename(rgb_p).split("_",1)[1]
        gt_p  = os.path.join(self.gt_dir, f"mask_{suf}")

        rgb = cv2.cvtColor(cv2.imread(rgb_p), cv2.COLOR_BGR2RGB)
        gt  = cv2.imread(gt_p, cv2.IMREAD_GRAYSCALE)

        if self.augment:
            # random flip
            if random.random() < 0.5:
                rgb = np.fliplr(rgb).copy(); gt = np.fliplr(gt).copy()
            if random.random() < 0.5:
                rgb = np.flipud(rgb).copy(); gt = np.flipud(gt).copy()
            # Random cropping
            scale = random.uniform(0.8, 1.0)
            h0,w0 = rgb.shape[:2]
            nh,nw = int(h0*scale), int(w0*scale)
            top = random.randint(0, h0-nh); left = random.randint(0, w0-nw)
            rgb = rgb[top:top+nh, left:left+nw]; gt = gt[top:top+nh, left:left+nw]
            # HSV 
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[...,2] *= random.uniform(0.8,1.2); hsv[...,1] *= random.uniform(0.8,1.2)
            rgb = cv2.cvtColor(np.clip(hsv,0,255).astype(np.uint8), cv2.COLOR_HSV2RGB)

        # Resize
        rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        gt  = cv2.resize(gt,  (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        img  = torch.from_numpy(rgb.astype("float32")/255.).permute(2,0,1)
        mask = torch.from_numpy((gt>127).astype("float32"))[None,...]
        return img, mask

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
    for img, gt in tqdm(loader, desc="Train"):
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
        for img, gt in tqdm(loader, desc="Valid"):
            img = img.to(device)
            out  = model.forward_image(img)
            fpn  = out["backbone_fpn"]
            pe   = out["vision_pos_enc"]
            emb, pe_h = fpn[-1], pe[-1]

            sparse_pe = torch.zeros((img.size(0),0,model.sam_prompt_encoder.embed_dim),
                                    dtype=emb.dtype, device=emb.device)
            dense_pe  = model.sam_prompt_encoder.get_dense_pe().to(device)
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
            preds.append(prob_map)
            gts.append(gt[0,0].cpu().numpy())

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

    # Train augmentation, val not
    # 80/20 train/val
    ds_aug   = TreeSegDataset(args.rgb_dir, args.gt_dir, IMAGE_SIZE, augment=True)
    ds_noaug = TreeSegDataset(args.rgb_dir, args.gt_dir, IMAGE_SIZE, augment=False)

    n = len(ds_aug)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
    n_train = int(n * args.split)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    tr_loader = DataLoader(Subset(ds_aug, train_idx), batch_size=1, shuffle=True)
    va_loader = DataLoader(Subset(ds_noaug, val_idx), batch_size=1, shuffle=False)

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

    for ep in range(1, args.epochs + 1):
        print(f"\nEpoch {ep}/{args.epochs}")
        tr_loss = train_epoch(model, tr_loader, optimizer, ep, warmup, args.device)
        val_iou, val_thr = valid_epoch(model, va_loader, args.device)
        print(f"Train Loss: {tr_loss:.4f} | Val IoU: {val_iou:.4f} @thr {val_thr:.2f}")

        scheduler.step(val_iou)

        if val_iou > best_iou:
            best_iou, best_thr = val_iou, val_thr
            torch.save(
                model.state_dict(),
                os.path.join(args.out_dir, "SAM2_finetune.pkl")
            )
            print(f"★ Saved best model (IoU {best_iou:.4f} @thr {best_thr:.2f})")

if __name__ == "__main__":
    main()
