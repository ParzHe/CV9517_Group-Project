import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from datetime import datetime

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, RichProgressBar, RichModelSummary, Timer
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from data import AerialDeadTreeSegDataModule
from lightning.pytorch.tuner import Tuner
from lightning_modules import SMPLitModule
from utils import paths
import segmentation_models_pytorch as smp
from models import FreezeSMPEncoderUtils, modes_list, encoders_list

from rich import print
from rich.logging import RichHandler
import logging

TARGET_SIZE = 256
BATCH_SIZE = 32
VERSION_SUFFIX = ""  # Suffix for the version, can be changed as needed
PRECISION = "bf16-mixed"  # Use bf16 mixed precision for training
LOSS1 = smp.losses.JaccardLoss(mode='binary', from_logits=True)
LOSS2 = smp.losses.FocalLoss(mode='binary')
EARLY_STOP_PATIENCE = 20
FREEZE_ENCODER_LAYERS = False  # Set to True if you want to freeze encoder layers
FREEZE_ENCODER_LAYERS_RANGE = (0, 1)  # Range of layers to freeze, if applicable
MAX_EPOCHS = 100
MIN_LR = 1e-3 # Minimum learning rate for the learning rate finder
MAX_LR = 0.1  # Maximum learning rate for the learning rate finder

arch_list = modes_list()
arch_list.remove("Unet")
arch_list.remove("UnetPlusPlus")
modality_list = ["merged", "rgb", "nrg"]

# Initialize callbacks

freeze_tool = FreezeSMPEncoderUtils()

def callbacks(encoder_name, arch, version):
    progress_bar = RichProgressBar()
    model_sum_callback = RichModelSummary(max_depth=2)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stop_callback = EarlyStopping(
        monitor="per_image_iou/val",
        patience=EARLY_STOP_PATIENCE,
        verbose=True,
        mode="max"  # Maximize the metric
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(paths.checkpoint_dir, f"smp_{encoder_name}_{arch}", version),
        monitor="per_image_iou/val",
        filename="{epoch:02d}-{per_image_iou_val:.4f}",
        mode="max",
        save_top_k=2,
        enable_version_counter=True,
    )
    return [
        progress_bar,
        model_sum_callback,
        lr_monitor,
        early_stop_callback,
        checkpoint_callback
    ]

def run_train():
    for modality in modality_list:
        version = f"{modality}_{TARGET_SIZE}" if VERSION_SUFFIX == "" else f"{modality}_{TARGET_SIZE}_{VERSION_SUFFIX}"
        data_module = AerialDeadTreeSegDataModule(
            val_split=0.1, test_split=0.2, seed=42,
            modality=modality, # in_channels=4. If modality is "merged", it will use 4 channels (RGB + NIR); Otherwise, it will use 3 channels (RGB).
            batch_size=BATCH_SIZE,
            num_workers= int(os.cpu_count() - 2) if os.cpu_count() is not None else 0,
            target_size=TARGET_SIZE
        )
        
        for arch in arch_list:
            for encoder_name in encoders_list(arch):
                print("\n\n")
                print("[dim]--------------------------------------------------[/dim]")
                print(f"Training [bold]{encoder_name}-{arch} [/bold] on [bold] {modality} [/bold] modality")
                start_time = datetime.now()
                print(f"Training started at [bold]{start_time.strftime('%Y-%m-%d %H:%M:%S')}[/bold].")
                
                model = SMPLitModule(
                    arch=arch,
                    encoder_name=encoder_name,
                    in_channels=data_module.in_channels,
                    loss1=LOSS1,
                    loss2=LOSS2,
                )
                if FREEZE_ENCODER_LAYERS:
                    freeze_tool(model, encoder_name, layers_range=FREEZE_ENCODER_LAYERS_RANGE)
                
                # Initialize callbacks
                callback_list = callbacks(encoder_name, arch, version)
                
                # Initialize logger
                logger = TensorBoardLogger(paths.tensorboard_log_dir,  name=f"smp_{encoder_name}_{arch}", version=version)

                # Initialize the trainer
                trainer = L.Trainer(
                    precision=PRECISION,
                    max_epochs=MAX_EPOCHS,
                    enable_progress_bar=True,
                    logger=logger,
                    log_every_n_steps=5,
                    callbacks=callback_list,
                )
                timer = Timer()
                trainer.callbacks.append(timer)
                
                tuner = Tuner(trainer)
                lr_finder = tuner.lr_find(model, datamodule=data_module,
                                        min_lr=MIN_LR, max_lr=MAX_LR,
                                        num_training=100, early_stop_threshold=4)
                suggested_lr = lr_finder.suggestion()

                print(f"\nSuggested learning rate: {suggested_lr}")
                Cur_init_lr = model.hparams.lr if hasattr(model.hparams, 'lr') else model.lr
                print(f"Current initial learning rate: [bold]{Cur_init_lr}[/bold]")

                trainer.fit(model, datamodule=data_module)
                
                end_time = datetime.now()
                print(f"[green]Training [bold]{encoder_name}-{arch}[/bold] on {modality} modality completed[/green]")
                print(f"Completion time is at [bold]{end_time.strftime('%Y-%m-%d %H:%M:%S')}[/bold].")
                print(f"Total training stage time: [bold]{timer.time_elapsed('train')} seconds[/bold].")
                print(f"Total validation stage time: [bold]{timer.time_elapsed('validate')} seconds[/bold].")


                # Test the model
                print(f"\nTesting [bold]{encoder_name}-{arch}[/bold] on {modality} modality")
                start_time = datetime.now()
                print(f"Testing started at [bold]{start_time.strftime('%Y-%m-%d %H:%M:%S')}[/bold].")
                trainer.test(model, datamodule=data_module)
                end_time = datetime.now()
                print(f"Testing completed at [bold]{end_time.strftime('%Y-%m-%d %H:%M:%S')}[/bold].")
                print(f"Total testing time: [bold]{(timer.time_elapsed('test'))} seconds[/bold].")
                print("[dim]--------------------------------------------------[/dim]")
                print("\n\n")

                del model
                del logger
                del trainer
                del tuner

if __name__ == "__main__":
    print("[bold]Starting training process for all configurations...[/bold]")
    print(f"Target size: {TARGET_SIZE}. Max epochs: {MAX_EPOCHS}. ")
    run_train()
    print("[green]Training completed for all configurations.[/green]")
