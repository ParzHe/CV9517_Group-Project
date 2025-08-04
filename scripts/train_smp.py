import os
import sys
import torch
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, RichProgressBar, RichModelSummary, Timer
from lightning.pytorch.loggers import TensorBoardLogger

from data import AerialDeadTreeSegDataModule
from lightning.pytorch.tuner import Tuner
from lightning_modules import SMPLitModule
from utils import paths, make_logger, callbacks
import segmentation_models_pytorch as smp
from models import FreezeSMPEncoderUtils, modes_list, encoders_list

from rich import print

torch.set_float32_matmul_precision('high')

TARGET_SIZE = 256
BATCH_SIZE = 32  # Default 32, if oom, try 16 or 8
ACCUMULATE_GRAD_BATCHES = 1  # Default 1, if oom, try 2 or 4
VERSION_SUFFIX = "test"  # Suffix for the version, can be changed as needed
PRECISION = "bf16-mixed"  # Use bf16 mixed precision for training
LOSS1 = smp.losses.JaccardLoss(mode='binary', from_logits=True)
LOSS2 = smp.losses.FocalLoss(mode='binary')
EARLY_STOP_PATIENCE = 30
FREEZE_ENCODER_LAYERS = False  # Set to True if you want to freeze encoder layers
FREEZE_ENCODER_LAYERS_RANGE = (0, 1)  # Range of layers to freeze, if applicable
MAX_EPOCHS = 100
MIN_LR = 1e-3 # Minimum learning rate for the learning rate finder
MAX_LR = 0.1  # Maximum learning rate for the learning rate finder

arch_list = modes_list()
encoder_only = "all"  # Set to "all" to use all available encoders, or specify a specific encoder name
modality_list = ["merged", "rgb", "nrg"]

freeze_tool = FreezeSMPEncoderUtils()

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
            for encoder_name in encoders_list(arch, only_available=encoder_only):
                log_dir = os.path.join(paths.checkpoint_dir, f"smp_{encoder_name}_{arch}", version)
                os.makedirs(log_dir, exist_ok=True)
                log_path = os.path.join(log_dir, "train.log")
                log = make_logger(
                    name=f"smp_{encoder_name}_{arch}_{modality}",
                    log_path=log_path
                )

                print()
                print("[dim]-[/dim]" * 60)
                log.info(f"Start Training [bold]{encoder_name}-{arch} [/bold] on [bold] {modality} [/bold] modality")

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
                model_name = f"smp_{encoder_name}_{arch}"
                callback_list = callbacks(model_name, EARLY_STOP_PATIENCE, version)

                # Initialize logger
                tb_logger = TensorBoardLogger(paths.tensorboard_log_dir,  name=model_name, version=version, log_graph=True)

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
                
                log.info(f"[green]Training [bold]{encoder_name}-{arch}[/bold] on {modality} modality completed[/green]")

                # Test the model
                log.info("")
                log.info(f"Testing [bold]{encoder_name}-{arch}[/bold] on {modality} modality")
                trainer.test(datamodule=data_module,ckpt_path="best")
                log.info(f"[green]Testing [bold]{encoder_name}-{arch}[/bold] on {modality} modality completed[/green]")

                log.info("")
                log.info(f"Total [bold]training[/bold] stage time: [bold]{timer.time_elapsed('train')} seconds[/bold].")
                log.info(f"Total [bold]validation[/bold] stage time: [bold]{timer.time_elapsed('validate')} seconds[/bold].")
                log.info(f"Total [bold]testing[/bold] time: [bold]{(timer.time_elapsed('test'))} seconds[/bold].")
                print("[dim]-[/dim]" * 60)

                del log
                del model
                del callback_list
                del tb_logger
                del trainer
                del tuner

if __name__ == "__main__":
    print("[bold]Starting training process for all configurations...[/bold]")
    print(f"Target size: {TARGET_SIZE}. Max epochs: {MAX_EPOCHS}. ")
    run_train()
    print("[green]Training completed for all configurations.[/green]")
