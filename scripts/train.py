import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import math
from datetime import datetime

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, RichProgressBar, RichModelSummary
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from data import AerialDeadTreeSegDataModule
from lightning.pytorch.tuner import Tuner
from lightning_modules import SMPLitModule
from utils import paths,TimerCallback
import segmentation_models_pytorch as smp
from models import FreezeSMPEncoderUtils, modes_list, encoders_list

from rich import print

TARGET_SIZE = 256
BATCH_SIZE = 48
PRECISION = "bf16-mixed"  # Use bf16 mixed precision for training
LOSS1 = smp.losses.JaccardLoss(mode='binary', from_logits=True)
LOSS2 = smp.losses.FocalLoss(mode='binary')
EARLY_STOP_PATIENCE = 20
FREEZE_ENCODER_LAYERS = False  # Set to True if you want to freeze encoder layers
FREEZE_ENCODER_LAYERS_RANGE = (0, 1)  # Range of layers to freeze, if applicable
MAX_EPOCHS = 100

arch_list = modes_list()
modality_list = ["merged", "rgb", "nrg"]

# Initialize callbacks
progress_bar = RichProgressBar()
freeze_tool = FreezeSMPEncoderUtils()

def run_train():
    for modality in modality_list:
        version = f"{modality}_{TARGET_SIZE}"
        data_module = AerialDeadTreeSegDataModule(
            val_split=0.1, test_split=0.2, seed=42,
            modality=modality, # in_channels=4. If modality is "merged", it will use 4 channels (RGB + NIR); Otherwise, it will use 3 channels (RGB).
            batch_size=BATCH_SIZE,
            num_workers= int(os.cpu_count() - 2) if os.cpu_count() is not None else 0,
            target_size=TARGET_SIZE
        )
        
        for arch in arch_list:
            for encoder_name in encoders_list(arch):
                print("[dim]--------------------------------------------------[/dim]")
                print(f"Training [bold]{arch} [/bold] with [bold] {modality} [/bold] modality and encoder [bold]{encoder_name}[/bold]")
                start_time = datetime.now()
                print(f"Training started at [bold]{start_time.strftime('%Y-%m-%d %H:%M:%S')}[/bold].")
                
                model = SMPLitModule(
                    arch=arch,
                    encoder_name=encoder_name,
                    in_channels=data_module.in_channels,
                    loss1=LOSS1,
                    loss2=LOSS2,
                    use_scheduler=False,
                )
                if FREEZE_ENCODER_LAYERS:
                    freeze_tool(model, encoder_name, layers_range=FREEZE_ENCODER_LAYERS_RANGE)
                
                trainer = L.Trainer(
                    log_every_n_steps=5,
                    precision=PRECISION,
                )
                
                tuner = Tuner(trainer)
                lr_finder = tuner.lr_find(model, datamodule=data_module,
                                        min_lr=1e-5, max_lr= 0.1,
                                        num_training=100, early_stop_threshold=4)
                suggested_lr = lr_finder.suggestion()

                # Round the suggested learning rate to 1 significant digit
                magnitude =  10 ** (math.floor(math.log10(suggested_lr)))
                suggested_lr = round(suggested_lr / magnitude) * magnitude

                print(f"Rounded suggested learning rate: {suggested_lr}")
                
                model = SMPLitModule(
                    arch=arch,
                    encoder_name=encoder_name,
                    in_channels=data_module.in_channels,
                    loss1=LOSS1,
                    loss2=LOSS2,
                    lr=suggested_lr,
                    use_scheduler=True
                )
                if FREEZE_ENCODER_LAYERS:
                    freeze_tool(model, encoder_name, layers_range=FREEZE_ENCODER_LAYERS_RANGE)
                
                model_sum_callback = RichModelSummary(max_depth=2)

                lr_monitor = LearningRateMonitor(logging_interval='step')

                early_stop_callback = EarlyStopping(
                    monitor="per_image_iou/val",
                    patience=EARLY_STOP_PATIENCE,
                    verbose=True,
                    mode="max"  # Maximize the metric
                )

                timer = TimerCallback()
                
                checkpoint_callback = ModelCheckpoint(
                    dirpath=os.path.join(paths.checkpoint_dir, f"smp_{encoder_name}_{arch}", version),
                    monitor="per_image_iou/val",
                    filename="{epoch:02d}-{per_image_iou_val:.4f}",
                    mode="max",
                    save_top_k=3,
                    enable_version_counter=True,
                )
                
                logger = TensorBoardLogger(paths.tensorboard_log_dir,  name=f"smp_{encoder_name}_{arch}", version=version)

                trainer = L.Trainer(
                    precision=PRECISION,
                    max_epochs=MAX_EPOCHS,
                    enable_progress_bar=True,
                    logger=logger,
                    log_every_n_steps=5,
                    callbacks=[
                        model_sum_callback,
                        lr_monitor,
                        early_stop_callback,
                        timer,
                        progress_bar,
                        checkpoint_callback
                    ],
                )
                trainer.fit(model, datamodule=data_module)
                
                end_time = datetime.now()
                print(f"[green]Training {arch} with {encoder_name} on {modality} modality completed[/green]")
                print(f"Completion time is at [bold]{end_time.strftime('%Y-%m-%d %H:%M:%S')}[/bold].")
                print(f"Total training time: [bold]{(end_time - start_time).total_seconds()} seconds[/bold].")
                print("[dim]--------------------------------------------------[/dim]")
                print("\n\n")
                
                # Test the model
                print(f"Testing model {arch} with encoder {encoder_name} on {modality} modality")
                start_time = datetime.now()
                print(f"Testing started at [bold]{start_time.strftime('%Y-%m-%d %H:%M:%S')}[/bold].")
                trainer.test(model, datamodule=data_module)
                end_time = datetime.now()
                print(f"Testing completed for {arch} with {encoder_name} on {modality} modality at [bold]{end_time.strftime('%Y-%m-%d %H:%M:%S')}[/bold].")
                print(f"Total testing time: [bold]{(end_time - start_time).total_seconds()} seconds[/bold].")

                del model
                del logger
                del trainer
                del tuner

if __name__ == "__main__":
    print("[bold]Starting training process for all configurations...[/bold]")
    print(f"Target size: {TARGET_SIZE}. Max epochs: {MAX_EPOCHS}. ")
    run_train()
    print("[green]Training completed for all configurations.[/green]")
