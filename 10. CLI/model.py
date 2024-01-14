import torch
import torch.nn.functional as F
from torch import nn, optim
import lightning as L
import torchmetrics
from metrics import MyAccuracy


class NN(L.LightningModule):
    def __init__(self, input_size:int = 784, learning_rate:float=0.001, num_classes:int =10):
        super().__init__()
        # log hparams
        self.save_hyperparameters()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.my_accuracy = MyAccuracy()
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        self.training_step_outputs = []

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, y_pred, y = self._common_step(batch, batch_idx)
        metric = {"loss": loss, "y_pred": y_pred, "y": y}
        self.training_step_outputs.append(metric)
        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.stack([x["loss"] for x in self.training_step_outputs]).mean()
        y_pred = torch.cat([x["y_pred"] for x in self.training_step_outputs])
        y = torch.cat([x["y"] for x in self.training_step_outputs])
        self.log_dict(
            {
                "avg_loss": avg_loss,
                "train_acc": self.my_accuracy(y_pred, y),
                "train_f1": self.f1_score(y_pred, y),
            },
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )  # sync dist for mutli-GPU
        self.training_step_outputs.clear()

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
        return optim.Adam(self.parameters(), lr=self.learning_rate)
