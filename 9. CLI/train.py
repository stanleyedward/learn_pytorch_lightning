import lightning as L
from model import NN
from dataset import MnistDataModule
import config
from callbacks import MyPrintingCallback, EarlyStopping
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.cli import LightningCLI
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64)
args = parser.parse_args()
if __name__ == "__main__":
    profiler = PyTorchProfiler()
    # strategy = DeepSpeedStrategy()
    datamodule = MnistDataModule(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    model = NN(
        input_size=config.INPUT_SIZE,
        learning_rate=config.LEARNING_RATE,
        num_classes=config.NUM_CLASSES,
    )

    trainer = L.Trainer(
        strategy="auto",  # Data Distributed Parallel for Multi-GPU from deep speed
        profiler=config.PROFILER,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=config.MIN_NUM_EPOCHS,
        max_epochs=config.MAX_NUM_EPOCHS,
        precision=config.PRECISION,
        callbacks=[MyPrintingCallback(), EarlyStopping(monitor="val_loss")],
        fast_dev_run=False,
    )
    # trainer.tune() to find out optimal hyperparams
    trainer.fit(model, datamodule=datamodule)
    trainer.validate(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
