from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.boring_classes import BoringDataModule, BoringModel

def cli_main():
    cli = LightningCLI(model_class=BoringModel, datamodule_class=BoringDataModule)
    
if __name__=='__main__':
    cli_main()