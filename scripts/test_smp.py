import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import torch
import lightning as L
from lightning.pytorch.callbacks import RichProgressBar, RichModelSummary, Timer
from lightning.pytorch.loggers import TensorBoardLogger

from data import AerialDeadTreeSegDataModule
from lightning_modules import SMPLitModule

from utils import paths, make_logger

import segmentation_models_pytorch as smp
from models import modes_list, encoders_list

from rich import print
import pandas as pd

torch.set_float32_matmul_precision('high')

TARGET_SIZE = 256
BATCH_SIZE = 1  # Test with batch size 1 for inference
VERSION_SUFFIX = ""  # Suffix for the version, can be changed as needed
PRECISION = "bf16-mixed"  # Use bf16 mixed precision
EPOCH_WARNING = 50  # If the model has not been trained for at least this many epochs, it will warn

arch_list = modes_list()
encoder_only = "all"
modality_list = ["merged", "rgb", "nrg"]

def run_test():
    # Initialize list to store all results
    all_results = []
    output_dir = os.path.join(paths.project_root, "outputs", "smp_test_results")
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    main_log = make_logger("smp_test", log_path=os.path.join(output_dir, "smp_test.log"), file_mode='w', show_level_name=True)

    for modality in modality_list:
        version = f"{modality}_{TARGET_SIZE}" if VERSION_SUFFIX == "" else f"{modality}_{TARGET_SIZE}_{VERSION_SUFFIX}"
        data_module = AerialDeadTreeSegDataModule(
            val_split=0.1, test_split=0.2, seed=42,
            modality=modality,  # in_channels=4. If modality is "merged", it will use 4 channels (RGB + NIR); Otherwise, it will use 3 channels (RGB).
            batch_size=BATCH_SIZE,
            num_workers=int(os.cpu_count() - 2) if os.cpu_count() is not None else 0,
            target_size=TARGET_SIZE
        )
        
        for arch in arch_list:
            for encoder_name in encoders_list(arch, only_available=encoder_only):
                log_dir = os.path.join(paths.checkpoint_dir, f"smp_{encoder_name}_{arch}", version)
                os.makedirs(log_dir, exist_ok=True)
                log_path = os.path.join(log_dir, "evaluate.log")
                log = make_logger(
                    name=f"smp_{encoder_name}_{arch}_{modality}",
                    log_path=log_path,
                    file_mode='w'
                )
                
                print()
                print("[dim]-[/dim]" * 60)
                log.info(f"Testing [bold]{encoder_name}-{arch} [/bold] on [bold] {modality} [/bold] modality")
                
                model = SMPLitModule(
                    arch=arch,
                    encoder_name=encoder_name,
                    in_channels=data_module.in_channels,
                )
                
                # Calculate model parameters
                total_params = sum(p.numel() for p in model.parameters())
                
                # Convert to millions for readability
                total_params_m = total_params / 1e6
                
                log.info("")
                log.info(f"Total Parameters: {total_params_m:.2f}M")
                log.info("")
                
                model_name = f"smp_{encoder_name}_{arch}"
                ckpt_dir = os.path.join(paths.checkpoint_dir, model_name, version)
                
                # Get ckpts list in the directory
                ckpts_name = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
                
                if not ckpts_name:
                    main_log.critical(f"No checkpoints found in {ckpt_dir}. Skipping this model.")
                    continue
                
                # Pick the best checkpoint based on the naming convention
                # e.g. "epoch=70-per_image_iou_val=0.4590.ckpt" and "epoch=80-per_image_iou_val=0.4720.ckpt" pick the one with the highest per_image_iou_val
                # if the iou is the same, pick the one with the highest epoch
                def extract_metrics(ckpt_name):
                    try:
                        iou = float(ckpt_name.split("per_image_iou_val=")[-1].split(".ckpt")[0])
                        epoch = int(ckpt_name.split("epoch=")[-1].split("-")[0])
                        if epoch < EPOCH_WARNING:
                            main_log.warning(f"Model {model_name} has not been trained for at least {EPOCH_WARNING} epochs. Training might not be sufficient.")
                        return (iou, epoch)
                    except (ValueError, IndexError):
                        return (0.0, 0)  # fallback for malformed names
                    
                ckpts_name.sort(key=extract_metrics, reverse=True)
                best_ckpt = ckpts_name[0]
                
                # Initialize logger
                tb_logger = TensorBoardLogger(paths.tensorboard_log_dir,  name=model_name, version=version)
                
                model_summary = RichModelSummary(max_depth=2)
                timer = Timer()
                progress_bar = RichProgressBar()
                
                trainer = L.Trainer(
                    accumulate_grad_batches=1,
                    precision=PRECISION,
                    max_epochs=1,  # For testing, we only need to run one epoch
                    enable_progress_bar=True,
                    logger=tb_logger,
                    log_every_n_steps=5,
                    callbacks=[model_summary, timer, progress_bar],
                )
                
                test_results = trainer.test(model=model, datamodule=data_module, ckpt_path=os.path.join(ckpt_dir, best_ckpt))
                log.info(f"[green]Testing [bold]{encoder_name}-{arch}[/bold] on {modality} modality completed[/green]")
                log.info(f"Test results for {model_name} on {modality}: \n{test_results}")
                
                # Extract metrics from test_results and add to all_results
                if test_results and len(test_results) > 0:
                    metrics = test_results[0]  # test_results is a list with one dict
                    result_row = {
                        'Architecture': arch,
                        'Encoder': encoder_name,
                        'Parameters': total_params_m,
                        'Modality': modality,
                        'Best_Checkpoint': best_ckpt,
                        'Per_Image_IoU': metrics.get('per_image_iou/test', None),
                        'Dataset_IoU': metrics.get('dataset_iou/test', None),
                        'Accuracy': metrics.get('accuracy/test', None),
                        'F1_Score': metrics.get('f1_score/test', None),
                        'F2_Score': metrics.get('f2_score/test', None),
                        'Precision': metrics.get('precision/test', None),
                        'Recall': metrics.get('recall/test', None),
                        'Sensitivity': metrics.get('sensitivity/test', None),
                        'Specificity': metrics.get('specificity/test', None),
                        'Test_Time_Seconds': timer.time_elapsed('test')
                    }
                    all_results.append(result_row)
                
                log.info("")
                log.info(f"Total [bold]testing[/bold] time: [bold]{(timer.time_elapsed('test'))} seconds[/bold].")
                print("[dim]-[/dim]" * 60)
                
                del log
                del model
                del tb_logger
                del trainer
                
    # Convert results to DataFrame and save to CSV
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Sort by Architecture and Encoder for better organization
        df = df.sort_values(['Architecture', 'Encoder', 'Modality'])
        
        # Save to CSV with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"smp_test_results_{TARGET_SIZE}_{timestamp}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        
        df.to_csv(csv_path, index=False, float_format='%.4f')
        main_log.info(f"[green]Test results saved to: [bold]{csv_path}[/bold][/green]")

        # Display summary statistics
        main_log.info("\n[bold]Test Results Summary of modalities:[/bold]")
        main_log.info(df.groupby(['Modality']).agg({
            'Per_Image_IoU': ['mean', 'std', 'max'],
            'Dataset_IoU': ['mean', 'std', 'max'],
        }).round(4))

        main_log.info("\n[bold]Test Results Summary of architectures:[/bold]")
        main_log.info(df.groupby(['Architecture']).agg({
            'Per_Image_IoU': ['mean', 'std', 'max'],
            'Dataset_IoU': ['mean', 'std', 'max'],
            'Test_Time_Seconds': ['mean', 'std', 'max']
        }).round(4))
        
        main_log.info("\n[bold]Test Results Summary of feature extraction backbones:[/bold]")
        main_log.info(df.groupby(['Encoder']).agg({
            'Per_Image_IoU': ['mean', 'std', 'max'],
            'Dataset_IoU': ['mean', 'std', 'max'],
            'Test_Time_Seconds': ['mean', 'std', 'max']
        }).round(4))

        return df
    else:
        main_log.info("[red]No test results to save![/red]")
        return None

if __name__ == "__main__":
    print("[bold]Starting testing process for all configurations...[/bold]")
    print(f"Target size: {TARGET_SIZE}")
    run_test()
    print("[green]Testing completed for all configurations.[/green]")