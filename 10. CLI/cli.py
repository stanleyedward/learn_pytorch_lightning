from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.boring_classes import BoringDataModule, BoringModel
from model import NN 
from dataset import MnistDataModule
import torch
import model
import dataset


def cli_main():
    cli = LightningCLI(model_class=NN, datamodule_class=MnistDataModule)
     
# OPTIMIZERS
class LitAdam(torch.optim.Adam):
    def step(self, closure):
        print("⚡", "using LitAdam", "⚡")
        super().step(closure)


class FancyAdam(torch.optim.Adam):
    def step(self, closure):
        print("⚡", "using FancyAdam", "⚡")
        super().step(closure)

if __name__=='__main__':
    cli_main()