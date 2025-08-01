import torch
import lightning as L
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp

TRAIN_STAGE = "train"
VAL_STAGE = "val"
TEST_STAGE = "test"

class SegLitModule(L.LightningModule):
    def __init__(self, model, in_channels=3,
                 loss1 = smp.losses.JaccardLoss(mode='binary', from_logits=True),
                 loss2 = smp.losses.FocalLoss(mode='binary'),
                 lr=1e-3, use_scheduler=True, **kwargs):
        super().__init__()
        assert model is not None, "Model must be provided"

        self.save_hyperparameters(ignore=["model", "encoder_weights", "out_classes", "loss1", "loss2", "lr", "use_scheduler"])

        self.model = model
        self.loss_fn1 = loss1
        self.loss_fn2 = loss2
        self.lr = lr
        self.use_scheduler = use_scheduler

        # Pre-create default loss to avoid repeated instantiation
        self._default_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)
        
        # Determine loss strategy once during initialization
        if self.loss_fn1 is not None and self.loss_fn2 is not None:
            self._loss_strategy = "combined"
        elif self.loss_fn1 is not None:
            self._loss_strategy = "single"
        else:
            self._loss_strategy = "default"

        # initialize step metics
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        if self.hparams.in_channels is not None:
            self.example_input_array = torch.randn(1, self.hparams.in_channels, 256, 256)  # Example input tensor for logging
        else:
            self.example_input_array = torch.randn(1, 3, 256, 256)

    def _loss_fn(self, logits_mask, mask):
        """
        Optimized loss function computation.
        """
        if self._loss_strategy == "combined":
            return self.loss_fn1(logits_mask, mask) + self.loss_fn2(logits_mask, mask)
        elif self._loss_strategy == "single":
            return self.loss_fn1(logits_mask, mask)
        else:
            return self._default_loss(logits_mask, mask)

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
        self.log(f'per_image_iou/{stage}', per_image_iou, prog_bar=True, logger=True)

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro-imagewise")
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")
        f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro-imagewise")
        precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro-imagewise")
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
        sensitivity = smp.metrics.sensitivity(tp, fp, fn, tn, reduction="micro-imagewise")
        specificity = smp.metrics.specificity(tp, fp, fn, tn, reduction="micro-imagewise")

        # Except iou, all metrics are computed per image
        metrics = {
            f"dataset_iou/{stage}": dataset_iou,
            f"accuracy/{stage}": accuracy,
            f"f1_score/{stage}": f1_score,
            f"f2_score/{stage}": f2_score,
            f"precision/{stage}": precision,
            f"recall/{stage}": recall,
            f"sensitivity/{stage}": sensitivity,
            f"specificity/{stage}": specificity,
        }
        
        self.log_dict(metrics, prog_bar=False, logger=True)
        
        if stage == VAL_STAGE:
            # log the per_image_iou for model name
            self.log(f"per_image_iou_{stage}", per_image_iou, on_step=False, on_epoch=True, prog_bar=False)

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
        
        if not self.use_scheduler:
            return optimizer

        if hasattr(self.trainer, 'max_epochs') and int(self.trainer.max_epochs) > 0:
            scheduler = {
                'scheduler': lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=self.lr,
                    total_steps=self.trainer.max_steps if self.trainer.max_steps > 0 else self.trainer.estimated_stepping_batches,
                    pct_start=0.1,
                anneal_strategy="cos",
                ),
                'interval': 'step', 
            }
            return [optimizer], [scheduler]
        else:
            return [optimizer]