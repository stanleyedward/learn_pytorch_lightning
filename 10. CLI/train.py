import lightning as L
from model import NN
from dataset import MnistDataModule
import config
from callbacks import MyPrintingCallback, EarlyStopping
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.strategies import DeepSpeedStrategy
import wandb
from lightning.pytorch.loggers import WandbLogger

if __name__ == "__main__":
    profiler = PyTorchProfiler()
    strategy = DeepSpeedStrategy()
    wandb_logger = WandbLogger(
        project="first MNIST using lightning", save_dir=config.LOGS_DIR
    )
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
        strategy="auto",
        profiler=config.PROFILER,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=config.MIN_NUM_EPOCHS,
        max_epochs=config.MAX_NUM_EPOCHS,
        precision=config.PRECISION,
        callbacks=[MyPrintingCallback(), EarlyStopping(monitor="val_loss")],
        logger=wandb_logger,
    )

    # trainer.tune() to find out optimal hyperparams
    trainer.fit(model, datamodule=datamodule)
    trainer.validate(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

    wandb.finish()
