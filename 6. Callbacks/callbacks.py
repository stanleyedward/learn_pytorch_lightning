from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, EarlyStopping


class MyPrintingCallback(Callback):
    def __init(self):
        super().__init__()

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        print("Starting to train!")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        print("Training is done.")
