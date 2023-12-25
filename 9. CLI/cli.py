from lightning.pytorch.cli import LightningCLI
from model import NN
from dataset import MnistDataModule
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule
import config


def cli_main():
    # cli = LightningCLI(model_class=NN(
    #     input_size=config.INPUT_SIZE,
    #     learning_rate=config.LEARNING_RATE,
    #     num_classes=config.NUM_CLASSES,
    # ),
    # datamodule_class=MnistDataModule(
    #     data_dir=config.DATA_DIR,
    #     batch_size=config.BATCH_SIZE,
    #     num_workers=config.NUM_WORKERS,
    # ))
    cli = LightningCLI(model_class=DemoModel, datamodule_class=BoringDataModule)


if __name__ == "__main__":
    cli_main()
