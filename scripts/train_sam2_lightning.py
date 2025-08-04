import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, RichProgressBar, RichModelSummary, Timer
from lightning.pytorch.loggers import TensorBoardLogger

from data import AerialDeadTreeSegDataModule
from lightning.pytorch.tuner import Tuner
from lightning_modules import SAM2LitModule
from utils import paths, make_logger, callbacks
import segmentation_models_pytorch as smp

from rich import print

from sam2.sam2.modeling.sam2_base import SAM2Base

TARGET_SIZE = 1024
BATCH_SIZE = 1  # Image_pe should have size 1 in batch dim
ACCUMULATE_GRAD_BATCHES = 1  
VERSION_SUFFIX = ""  # Suffix for the version, can be changed as needed
PRECISION = "bf16-mixed"  # SAM2 requires 32-bit precision
LOSS1 = smp.losses.LovaszLoss(mode='binary', from_logits=True)
LOSS2 = smp.losses.JaccardLoss(mode='binary', from_logits=True)
EARLY_STOP_PATIENCE = 30
MAX_EPOCHS = 100
MIN_LR = 1e-3 # Minimum learning rate for the learning rate finder
MAX_LR = 0.1  # Maximum learning rate for the learning rate finder

SAM2_prefix = "sam2.1"
sam2_dir = os.path.join(project_root, "sam2")
cfg_dir = os.path.join(sam2_dir, "sam2", "configs", f"{SAM2_prefix}")
ckpt_dir = os.path.join(sam2_dir, "checkpoints")

model_map = {
    "tiny": SAM2LitModule(
        cfg_path=os.path.join(cfg_dir, f"{SAM2_prefix}_hiera_t.yaml"),
        ckpt_path=os.path.join(ckpt_dir, f"{SAM2_prefix}_hiera_tiny.pt"),
        loss1=LOSS1,
        loss2=LOSS2,
    ),
    "small": SAM2LitModule(
        cfg_path=os.path.join(cfg_dir, f"{SAM2_prefix}_hiera_s.yaml"),
        ckpt_path=os.path.join(ckpt_dir, f"{SAM2_prefix}_hiera_small.pt"),
        loss1=LOSS1,
        loss2=LOSS2,
    ),
    "base_plus": SAM2LitModule(
        cfg_path=os.path.join(cfg_dir, f"{SAM2_prefix}_hiera_b+.yaml"),
        ckpt_path=os.path.join(ckpt_dir, f"{SAM2_prefix}_hiera_base_plus.pt"),
        loss1=LOSS1,
        loss2=LOSS2,
    ),
    "large": SAM2LitModule(
        cfg_path=os.path.join(cfg_dir, f"{SAM2_prefix}_hiera_l.yaml"),
        ckpt_path=os.path.join(ckpt_dir, f"{SAM2_prefix}_hiera_large.pt"),
        loss1=LOSS1,
        loss2=LOSS2,
    ),
}

modality_list = ["rgb", "nrg"]
models_list = list(model_map.keys())

def run_train():
    for modality in modality_list:
        for model_name in models_list:
            version = f"{modality}_{TARGET_SIZE}" if VERSION_SUFFIX == "" else f"{modality}_{TARGET_SIZE}_{VERSION_SUFFIX}"
            model_full_name = f"{SAM2_prefix}_{model_name}"
            data_module = AerialDeadTreeSegDataModule(
                val_split=0.1, test_split=0.2, seed=42,
                modality=modality, # in_channels=4. If modality is "merged", it will use 4 channels (RGB + NIR); Otherwise, it will use 3 channels (RGB).
                batch_size=BATCH_SIZE,
                num_workers= int(os.cpu_count() - 2) if os.cpu_count() is not None else 0,
                target_size=TARGET_SIZE
            )
            
            log_dir = os.path.join(paths.checkpoint_dir, f"{model_full_name}", version)
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "train.log")
            log = make_logger(
                name=f"{model_full_name}_{modality}",
                log_path=log_path
            )
            
            print()
            print("[dim]-[/dim]" * 60)
            log.info(f"Start Training [bold]{model_full_name} [/bold] on [bold] {modality} [/bold] modality")

            model = model_map[model_name]

            # Initialize callbacks
            callback_list = callbacks(model_full_name, EARLY_STOP_PATIENCE, version)
            
            tb_logger = TensorBoardLogger(paths.tensorboard_log_dir,  name=f"{model_full_name}_{modality}", version=version)
            
            # Initialize the trainer
            trainer = L.Trainer(
                accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
                precision=PRECISION,
                max_epochs=MAX_EPOCHS,
                enable_progress_bar=True,
                logger=tb_logger,
                log_every_n_steps=5,
                callbacks=callback_list,
            )
            timer = Timer()
            trainer.callbacks.append(timer)
            
            tuner = Tuner(trainer)
            lr_finder = tuner.lr_find(model, datamodule=data_module,
                                    min_lr=MIN_LR, max_lr=MAX_LR,
                                    num_training=100, early_stop_threshold=4)
            log.info(f"Learning rate finder suggestion: {lr_finder.suggestion()}")
            Cur_init_lr = model.hparams.lr if hasattr(model.hparams, 'lr') else model.lr
            log.info(f"Current initial learning rate update to suggested: [bold]{Cur_init_lr}[/bold]")

            trainer.fit(model, datamodule=data_module)

            log.info(f"[green]Training [bold]{model_full_name}[/bold] on {modality} modality completed[/green]")

            # Test the model
            log.info("")
            log.info(f"Testing [bold]{model_full_name}[/bold] on {modality} modality")
            trainer.test(datamodule=data_module,ckpt_path="best")
            log.info(f"[green]Testing [bold]{model_full_name}[/bold] on {modality} modality completed[/green]")

            log.info("")
            log.info(f"Total [bold]training[/bold] stage time: [bold]{timer.time_elapsed('train')} seconds[/bold].")
            log.info(f"Total [bold]validation[/bold] stage time: [bold]{timer.time_elapsed('validate')} seconds[/bold].")
            log.info(f"Total [bold]testing[/bold] time: [bold]{(timer.time_elapsed('test'))} seconds[/bold].")
            print("[dim]-[/dim]" * 60)

            del log
            del model
            del tb_logger
            del trainer
            del tuner
            
if __name__ == "__main__":
    print("[bold]Starting training SAM2 process for all configurations...[/bold]")
    print(f"Target size: {TARGET_SIZE}. Max epochs: {MAX_EPOCHS}. ")
    run_train()
    print("[green]Training completed for all configurations.[/green]")