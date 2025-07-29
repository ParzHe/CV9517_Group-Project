import torch
import lightning as L
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp

TRAIN_STAGE = "train"
VAL_STAGE = "val"
TEST_STAGE = "test"

class SegLitModule(L.LightningModule):
    def __init__(self, model, 
                 loss = smp.losses.LovaszLoss(smp.losses.BINARY_MODE, from_logits=True, per_image=True),
                 lr=1e-3, use_scheduler=True, **kwargs):
        super().__init__()
        assert model is not None, "Model must be provided"

        self.save_hyperparameters(ignore=["model"])

        self.model = model

        # loss function
        self.loss_fn = loss

        # initialize step metics
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
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

        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)
        
        # Log the loss
        if stage == "train":
            self.log(f"metrics/loss/{stage}", loss, on_step=True, on_epoch=True, prog_bar=True)
        elif stage == "val":
            self.log(f"metrics/loss/{stage}", loss, on_step=False, on_epoch=True, prog_bar=True)
        else:
            self.log(f"metrics/loss/{stage}", loss, on_step=False, on_epoch=False, prog_bar=False)
        

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

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
        
        metrics = {
            f"metrics/accuracy/{stage}": accuracy,
            f"metrics/per_image_iou/{stage}": per_image_iou,
            f"metrics/dataset_iou/{stage}": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, TRAIN_STAGE)
        # append the metics of each step to the
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, TRAIN_STAGE)
        # empty set output list
        self.training_step_outputs.clear()
        return

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, VAL_STAGE)
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, VAL_STAGE)
        self.validation_step_outputs.clear()
        return

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, TEST_STAGE)
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, TEST_STAGE)
        # empty set output list
        self.test_step_outputs.clear()
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        
        if not self.hparams.use_scheduler:
            return optimizer

        if hasattr(self.trainer, 'max_epochs') and int(self.trainer.max_epochs) > 0:
            scheduler = {
                'scheduler': lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=self.hparams.lr,
                    total_steps=self.trainer.max_steps if self.trainer.max_steps > 0 else self.trainer.estimated_stepping_batches,
                    pct_start=0.1,
                anneal_strategy="cos",
                ),
                'interval': 'step', # 
            }
            return [optimizer], [scheduler]
        else:
            return [optimizer]