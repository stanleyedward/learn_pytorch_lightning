from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import random_split
import lightning as L
import torchmetrics
from torchmetrics import Metric

#making our own accuracy metric
class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        preds = torch.argmax(preds, dim=1)
        assert preds.shape == target.shape, 'preds and target must have the same shape'
        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
        
class NN(L.LightningModule):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.my_accuracy = MyAccuracy()
        self.f1_score = torchmetrics.F1Score(task='multiclass', num_classes=num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        loss, y_pred, y = self._common_step(batch, batch_idx)
        accuracy = self.my_accuracy(y_pred, y)
        f1_score = self.f1_score(y_pred, y)
        self.log_dict({'train_loss': loss, 'train_accuracy': accuracy, 'train_f1_score': f1_score},
                      on_step=False, on_epoch=True, prog_bar=True) 
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_pred, y = self._common_step(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_pred, y = self._common_step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss

    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y)
        return loss, y_pred, y

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        y_pred = self.forward(x)
        preds = torch.argmax(y_pred, dim=1)
        return preds

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)

class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def prepare_data(self) -> None:
        datasets.MNIST(root=self.data_dir, train=True)
        datasets.MNIST(root=self.data_dir, train=False)
        
    def setup(self, stage: str) -> None:
        entire_dataset = datasets.MNIST(
            root=self.data_dir,
            train=True,
            transform=transforms.ToTensor(),
            download=False
        )
        self.train_dataset, self.val_dataset = random_split(entire_dataset, [50000, 10000])
        
        self.test_dataset = datasets.MNIST(
            root=self.data_dir,
            train=False,
            transform=transforms.ToTensor(),
            download=False
        )
        
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

# Set device cuda for GPU if it's available otherwise run on the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 3

#Load the dataset and dataloaders
datamodule = MNISTDataModule(data_dir='dataset/', batch_size=batch_size, num_workers=4)

# Initialize network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

#Setup the trainer
trainer = L.Trainer(
    accelerator="gpu", 
    devices=1, 
    min_epochs=1, 
    max_epochs=3, 
    precision=32
)
#trainer.tune() to find out optimal hyperparams
trainer.fit(model, datamodule=datamodule)
trainer.validate(model, datamodule=datamodule)
trainer.test(model, datamodule=datamodule)