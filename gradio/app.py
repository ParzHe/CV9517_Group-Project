# gradio/app.py
# This script is for creating a Gradio app for inference using Segmentation Models PyTorch (SMP) models.

import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import gradio as gr
import torch
import numpy as np
from PIL import Image

from lightning_modules import SMPLitModule
from models.smp_models_utils import archs_list, encoders_list
from utils import paths

from data.transforms import SegmentationTransform

# Load available architectures and encoders
ARCHS = archs_list()
CHECKPOINT_DIR = paths.checkpoint_dir

def get_encoders(arch):
    return encoders_list(arch)

def load_model(arch, encoder, image_size=256, modality="rgb"):
    # Example: load from checkpoint or instantiate
    # Replace with your actual checkpoint loading logic
    
    version = f"{modality}_{image_size}"
    ckpt_dir = os.path.join(CHECKPOINT_DIR, f"smp_{encoder}_{arch}", version)
    # Get ckpts list in the directory
    ckpts_name = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]

    if not ckpts_name:
        gr.Error(f"No checkpoints found in {ckpt_dir}. Skipping this model.")
    
    # Pick the best checkpoint based on the naming convention
    # e.g. "epoch=70-per_image_iou_val=0.4590.ckpt" and "epoch=80-per_image_iou_val=0.4720.ckpt" pick the one with the highest per_image_iou_val
    # if the iou is the same, pick the one with the highest epoch
    def extract_metrics(ckpt_name):
        try:
            iou = float(ckpt_name.split("per_image_iou_val=")[-1].split(".ckpt")[0])
            epoch = int(ckpt_name.split("epoch=")[-1].split("-")[0])
            return (iou, epoch)
        except (ValueError, IndexError):
            return (0.0, 0)  # fallback for malformed names
        
    ckpts_name.sort(key=extract_metrics, reverse=True)
    best_ckpt = ckpts_name[0]
    model = SMPLitModule.load_from_checkpoint(os.path.join(ckpt_dir, best_ckpt))
    return model

def infer(arch, encoder, image):
    model = load_model(arch, encoder)
    transform = SegmentationTransform(target_size=256, mode="val", modality="rgb")
    img_pil = Image.fromarray(image)
    img_tensor, _ = transform(img_pil, np.zeros_like(image))
    img_tensor = img_tensor.unsqueeze(0)
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    model.eval()
    model.freeze()
    y = model(img_tensor)
    # Postprocess: convert output to mask
    mask = y.squeeze().cpu().numpy()
    mask = (mask > 0).astype(np.uint8) * 255
    return mask

with gr.Blocks(title="Dead Tree Segmentation Demo") as demo:
    gr.Markdown("# Dead Tree Segmentation Demo")

    with gr.Column():
        gr.Markdown("Select an architecture and encoder, upload an image, and run inference to get the segmentation mask.")
        with gr.Row():
            arch = gr.Dropdown(label="Architecture", choices=ARCHS, value=ARCHS[0])
            encoder = gr.Dropdown(label="Encoder", choices=get_encoders(ARCHS[0]), value=get_encoders(ARCHS[0])[0])
        with gr.Row():
            image = gr.Image(type="numpy", label="Input Image", height=256, width=256)
            output = gr.Image(type="numpy", label="Output Prediction Mask", height=256, width=256)

    def update_encoders(selected_arch):
        return gr.Dropdown.update(choices=get_encoders(selected_arch), value=get_encoders(selected_arch)[0])

    arch.change(fn=update_encoders, inputs=arch, outputs=encoder)
    btn = gr.Button("Run Inference")
    btn.click(fn=infer, inputs=[arch, encoder, image], outputs=output)

demo.launch()