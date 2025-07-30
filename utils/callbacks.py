# utils/callbacks.py

from lightning.pytorch.callbacks import Callback
from rich import print
from datetime import datetime

# Record time taken for each epoch
# and the whole training/val/test process

class TimerCallback(Callback):
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.epoch_start_time = None

    def on_fit_start(self, trainer, pl_module):
        self.start_time = datetime.now()
        print(f"Training started at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}.")

    def on_fit_end(self, trainer, pl_module):
        total_duration = datetime.now() - self.start_time
        print(f"The model fit completed in {total_duration.total_seconds()} seconds at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
        trainer.logger.log_metrics({
            "total_duration": total_duration.total_seconds()
        })
    
    def on_test_start(self, trainer, pl_module):
        self.start_time = datetime.now()
        print(f"Testing started at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}.")

    def on_test_end(self, trainer, pl_module):
        total_duration = datetime.now() - self.start_time
        print(f"Testing completed in {total_duration.total_seconds()} seconds at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
        trainer.logger.log_metrics({
            "test_duration": total_duration.total_seconds()
        })