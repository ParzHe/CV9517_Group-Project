# utils.py
import torch
import torch.nn as nn

def iou_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    inter= (pred * target).sum(dim=(1,2,3))
    union= ((pred + target) >= 1).float().sum(dim=(1,2,3))
    return ((inter + eps) / (union + eps)).mean()

def pixel_accuracy(pred, target, eps=1e-6):
    pred    = (pred > 0.5).float()
    correct = (pred == target).float().sum()
    total   = torch.numel(target)
    return (correct + eps) / (total + eps)

class BCEDiceLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.bce   = nn.BCELoss()
        self.alpha = alpha
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        inter    = (pred * target).sum()
        union    = pred.sum() + target.sum()
        dice     = 1 - (2*inter + 1e-6) / (union + 1e-6)
        return self.alpha * bce_loss + (1 - self.alpha) * dice
