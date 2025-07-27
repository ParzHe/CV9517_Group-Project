# train.py
import os
import torch
from torch import optim
from tqdm import tqdm
import config
from data_processing import make_dataloaders
from model import UNet
from util import iou_score, BCEDiceLoss, pixel_accuracy

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0
    running_acc  = 0
    for imgs, masks in tqdm(loader, desc="Train"):
        imgs, masks = imgs.to(config.DEVICE), masks.to(config.DEVICE)
        preds       = model(imgs)
        loss        = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_acc  += pixel_accuracy(preds, masks).item()
    return running_loss / len(loader), running_acc / len(loader)

def validate(model, loader, criterion):
    model.eval()
    val_loss, val_iou, val_acc = 0, 0, 0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Val"):
            imgs, masks= imgs.to(config.DEVICE), masks.to(config.DEVICE)
            preds      = model(imgs)
            val_loss  += criterion(preds, masks).item()
            val_iou   += iou_score(preds, masks).item()
            val_acc   += pixel_accuracy(preds, masks).item()
    n = len(loader)
    return val_loss/n, val_iou/n, val_acc/n

def run_training():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    train_loader, val_loader = make_dataloaders(
        config.RGB_DIR, config.NRG_DIR
    )
    model     = UNet(in_channels=config.IN_CHANNELS).to(config.DEVICE)
    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    best_iou = 0
    for epoch in range(1, config.NUM_EPOCHS+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_iou, val_acc = validate(model, val_loader, criterion)
        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f} | Val Acc: {val_acc:.4f}")
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(),
                       os.path.join(config.OUTPUT_DIR, "best_model.pth"))
    print(f"Best IoU: {best_iou:.4f}")

if __name__ == "__main__":
    run_training()
