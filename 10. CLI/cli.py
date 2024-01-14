from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.boring_classes import BoringDataModule, BoringModel
from model import NN 
from dataset import MnistDataModule

def cli_main():
    cli = LightningCLI(model_class=NN, datamodule_class=MnistDataModule)
    
if __name__=='__main__':
    cli_main()