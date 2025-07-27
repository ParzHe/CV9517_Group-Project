# utils.py
import torch.nn as nn
import segmentation_models_pytorch as smp

def smp_metrics(pred, target):
    if pred.dim() == 4 and pred.shape[1] == 1:
        pred = pred.squeeze(1)
        
    target = target.long()  # Ensure target is in integer format
    pred = (pred > 0.5).long()  # Convert predictions to binary integers
    
    tp, fp, fn, tn = smp.metrics.get_stats(pred, target, mode='binary', threshold=0.5)
    
    return tp, fp, fn, tn

def iou_score(pred, target, eps=1e-6):
    tp, fp, fn, tn = smp_metrics(pred, target)
    
    iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro', zero_division=eps)
    return iou_score

def pixel_accuracy(pred, target, eps=1e-6):
    tp, fp, fn, tn = smp_metrics(pred, target)

    accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction='macro', zero_division=eps)
    return accuracy

class BCEDiceLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.bce   = nn.BCELoss()
        self.alpha = alpha
    def forward(self, pred, target):
        if pred.dim() == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
            
        bce_loss = self.bce(pred, target)
        inter    = (pred * target).sum()
        union    = pred.sum() + target.sum()
        dice     = 1 - (2*inter + 1e-6) / (union + 1e-6)
        return self.alpha * bce_loss + (1 - self.alpha) * dice
