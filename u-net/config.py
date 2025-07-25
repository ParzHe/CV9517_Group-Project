# config.py
import os
import torch

# 数据路径
DATA_DIR   = "data"
RGB_DIR    = os.path.join(DATA_DIR, "RGB_images")
NRG_DIR    = os.path.join(DATA_DIR, "NRG_images")
MASK_DIR   = os.path.join(DATA_DIR, "masks")
OUTPUT_DIR = "outputs"

# 预处理 & 划分
IMG_SIZE   = (256, 256)
VAL_SPLIT  = 0.2
BATCH_SIZE = 32

# 训练超参
NUM_EPOCHS  = 50
LR          = 1e-3
IN_CHANNELS = 4   # R, G, B, NIR 一起输入

# 设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
