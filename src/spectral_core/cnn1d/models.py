import os
import numpy as np
import torch
from torch import nn
import torchmetrics
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning as L

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.cli import LightningCLI
from spectral_core.cnn1d.dataset import SpectraDataset, SpectraDataModule


class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=360)
        self.dropout1 = nn.Dropout(0.1)
        
        self.fc2 = nn.Linear(in_features=360, out_features=180)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(in_features=180, out_features=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class CNN1DModel(nn.Module):
    def __init__(self, input_size):
        super(CNN1DModel, self).__init__()
        self.input_size = input_size

        self.conv1d1 = nn.Conv1d(
            in_channels=1, 
            out_channels=64, 
            kernel_size=5, 
            stride=1, 
            padding=5
        )
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.max_pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv1d2 = nn.Conv1d(
            in_channels=64, 
            out_channels=64, 
            kernel_size=5, 
            stride=1, 
            padding=5
        )
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.max_pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        flatten_size = self._calculate_flatten_size()
        self.fc = MLP(flatten_size)

    def _calculate_flatten_size(self):
        dummy_input = torch.zeros(1, 1, self.input_size)
        with torch.no_grad():
            dummy_output = self.max_pool1(F.relu(self.batch_norm1(self.conv1d1(dummy_input))))
            dummy_output = self.max_pool2(F.relu(self.batch_norm2(self.conv1d2(dummy_output))))
        return dummy_output.numel()

    def forward(self, x):
        x = x.unsqueeze(1)

        x = F.relu(self.batch_norm1(self.conv1d1(x)))
        x = self.max_pool1(x)

        x = F.relu(self.batch_norm2(self.conv1d2(x)))
        x = self.max_pool2(x)

        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output


class Lightning1DCNNModel(L.LightningModule):
    def __init__(self, input_size, learning_rate=1e-3):
        super(Lightning1DCNNModel, self).__init__()
        self.save_hyperparameters()
        self.model = CNN1DModel(input_size)
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()

        self.r2_metric = torchmetrics.R2Score()
        self.rmse_metric = torchmetrics.MeanSquaredError(squared=False)
        self.pearson_corr = torchmetrics.PearsonCorrCoef()
        self.mae_metric = torchmetrics.MeanAbsoluteError()

    def forward(self, x):
        return self.model(x)
    
    def predict_step(self, batch, batch_idx):
        x, _ = batch
        y_pred = self.forward(x).squeeze(-1)
        return y_pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x).squeeze(-1)
        loss = self.criterion(y_pred, y)
        
        self.log(
            'train_loss', 
            loss, 
            on_epoch=True, 
            prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x).squeeze(-1)
        loss = self.criterion(y_pred, y)

        r2 = self.r2_metric(y_pred, y)
        rmse = self.rmse_metric(y_pred, y)

        self.log_dict(
            {
                'val_r2': r2,
                'val_loss': loss,
                'val_rmse': rmse,
            },
            on_epoch=True,
            prog_bar=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x).squeeze(-1)
        loss = self.criterion(y_pred, y)

        r2 = self.r2_metric(y_pred, y)
        rmse = self.rmse_metric(y_pred, y)
        r = self.pearson_corr(y_pred, y)
        mae = self.mae_metric(y_pred, y)

        self.log_dict(
            {
                'test_r2': r2,
                'test_loss': loss,
                'test_rmse': rmse,
                'test_pearson_r': r,
                'test_mae': mae,
            },
            on_epoch=True,
            prog_bar=True
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
