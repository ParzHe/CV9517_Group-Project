# lightning_modules/sam2_module.py
# (Future Work)
# This module defines the SAM2 model for semantic segmentation using PyTorch Lightning.

import random
import torch
import lightning as L
import torch.nn.functional as F
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp
from .segmentation_module import SegLitModule

from omegaconf import OmegaConf
from hydra.utils import instantiate

TRAIN_STAGE = "train"
VAL_STAGE = "val"
TEST_STAGE = "test"

class SAM2LitModule(SegLitModule):
    def __init__(self, cfg_path, ckpt_path,
                 loss1=smp.losses.JaccardLoss(mode='binary', from_logits=True),
                 loss2=smp.losses.FocalLoss(mode='binary'),
                 lr=1e-3, use_scheduler=True, **kwargs):
        cfg, model = self._load_cfg_model(cfg_path, ckpt_path, device="cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = cfg.model.image_size
        super().__init__(model=model, in_channels=3, loss1=loss1, loss2=loss2, lr=lr, use_scheduler=use_scheduler, target_size=1024, **kwargs)
        self.save_hyperparameters(ignore=["cfg_path", "ckpt_path", "loss1", "loss2", "lr", "use_scheduler"])
        self.automatic_optimization = False
        
    # Load and freeze
    def _load_cfg_model(self, cfg_path, ckpt_path, device):
        cfg   = OmegaConf.load(cfg_path)
        model = instantiate(cfg.model)  
        sd    = torch.load(ckpt_path, map_location=device)["model"]
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
        return cfg, model
    
    def forward(self, image):
        """
        Forward pass through the model.
        """
        out = self.model.forward_image(image)
        fpn = out["backbone_fpn"]
        pe   = out["vision_pos_enc"]
        
        emb, pe_h = fpn[-1], pe[-1]        
        emb = emb.to(self.device)
        pe_h = pe_h.to(self.device)
        
        B, C, H, W = emb.size()
        
        dense_pe = self.model.sam_prompt_encoder.get_dense_pe().to(self.device)
            
        if dense_pe.shape[-2:] != (H, W):
            dense_pe = F.interpolate(
                dense_pe,
                size=(H, W),
                mode="bilinear",
                align_corners=False
            )

        sparse_pe = torch.zeros((B,0,self.model.sam_prompt_encoder.embed_dim),
                                dtype=emb.dtype, device=self.device)
        high_res_feats = [f.to(self.device) for f in fpn[-4:-1]]
        
        logits_mask, _, _, _ = self.model.sam_mask_decoder(
            image_embeddings=emb,
            image_pe=pe_h,
            sparse_prompt_embeddings=sparse_pe,
            dense_prompt_embeddings=dense_pe,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_feats
        )
        
        return logits_mask
    
    def train_forward(self, image, mask):
        out = self.model.forward_image(image)
        fpn = out["backbone_fpn"]
        pe   = out["vision_pos_enc"]
        
        emb, pe_h = fpn[-1], pe[-1]
        emb = emb.to(self.device)
        pe_h = pe_h.to(self.device)
        
        B, C, H, W = emb.size()
        
        sparse_pe = torch.zeros((B,0,self.model.sam_prompt_encoder.embed_dim),
                                dtype=emb.dtype, device=self.device)

        if random.random() < 0.5:
            Hm, Wm = self.model.sam_prompt_encoder.mask_input_size
            mask_low = F.interpolate(mask, size=(Hm,Wm), mode="nearest")
            dense_pe = self.model.sam_prompt_encoder._embed_masks(mask_low)
            dense_pe = dense_pe + self.model.sam_prompt_encoder.get_dense_pe().to(self.device)
        else:
            dense_pe = self.model.sam_prompt_encoder.get_dense_pe().to(self.device)

        if dense_pe.shape[-2:] != (H, W):
            dense_pe = F.interpolate(
                dense_pe,
                size=(H, W),
                mode="bilinear",
                align_corners=False
            )

        high_res_feats = [f.to(self.device) for f in fpn[-4:-1]]

        logits_mask, _, _, _ = self.model.sam_mask_decoder(
            image_embeddings=emb,
            image_pe=pe_h,
            sparse_prompt_embeddings=sparse_pe,
            dense_prompt_embeddings=dense_pe,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_feats
        )
        
        return logits_mask
    
    def shared_step(self, batch, stage):
        self.stage = stage
        image = batch["image"]
        mask = batch["mask"]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0
        
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0
        
        image, mask = image.to(self.device), mask.to(self.device)
        
        logits_mask = self.train_forward(image, mask) if stage == TRAIN_STAGE else self.forward(image)
            
        logits_mask = F.interpolate(
            logits_mask,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False
        )
        mask = F.interpolate(
            mask,
            size=(self.image_size, self.image_size),
            mode="nearest"
        )

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self._loss_fn(logits_mask, mask)

        # Log the loss
        if stage == TRAIN_STAGE:
            self.log(f"loss/{stage}", loss, on_step=True, on_epoch=True, prog_bar=True)
        elif stage == VAL_STAGE:
            self.log(f"loss/{stage}", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="binary"
        )
        
        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
    
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        train_loss_info = self.shared_step(batch, TRAIN_STAGE)
        loss = train_loss_info["loss"]
        self.manual_backward(loss)
        opt.step()
        
        # append the metics of each step to the
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info
        
    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, TRAIN_STAGE)
        # empty set output list
        self.training_step_outputs.clear()
        
        sch = self.lr_schedulers()
        if isinstance(sch, lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics[f"per_image_iou/{VAL_STAGE}"])
        
        return
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=1e-4,
        )

        if not self.use_scheduler:
            return optimizer

        if hasattr(self.trainer, 'max_epochs') and int(self.trainer.max_epochs) > 0:
            scheduler = {
                'scheduler': lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="max",
                    factor=0.5,
                    patience=3,
                    min_lr=1e-6
                ),
                'interval': 'epoch',
            }
            return [optimizer], [scheduler]
        else:
            return [optimizer]