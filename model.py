from models.s4d import S4D
from models.networks import S4DModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt

class LitS4Model(L.LightningModule):
    def __init__(self, d_input, d_output, encoder: nn.Module, loss='MSELoss'):
        super().__init__()

        self.encoder = encoder
        self.loss = loss
        self.d_output = d_output
        if self.loss=='MSELoss':
            self.criterion = nn.MSELoss()
        elif self.loss=='GaussianNLLLoss':
            self.d_split = self.d_output
            self.d_output = 2 * self.d_output
            self.criterion = nn.GaussianNLLLoss(reduction='mean', full=False, eps=1e-6)
        else: raise ValueError(f'Unknown loss function {self.loss}')

        self.val_outputs = []

        self.save_hyperparameters()

    def __loss__(self, X, y):
        y_preds = self.forward(X)
        if self.loss=='MSELoss':
            return self.criterion(y, y_preds)
        elif self.loss=='GaussianNLLLoss':
            y_hat, y_hat_variance = y_preds[:,:self.d_split], y_preds[:,self.d_split:]
            return self.criterion(y_hat, y, y_hat_variance)

    def forward(self, x):
        x = self.encoder(x)  
        if self.loss=='GaussianNLLLoss':
            x_uncertainties = F.softplus(x[..., self.d_split:])
            x = torch.cat([x[..., :self.d_split], x_uncertainties], dim=-1)
        return x

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return optimizer, scheduler

    def training_step(self, batch, batch_idx, log=True):
        X, y, _ = batch
        loss = self.__loss__(X, y)

        if log:
            self.log("train/loss",
                    loss,
                    on_step=False,
                    on_epoch=True,
                    reduce_fx='mean',
                    logger=True,
                    prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx, log=True):
        X, y, _ = batch
        loss = self.__loss__(X, y)

        #self.val_outputs.append((y.cpu().numpy(), y_hat.cpu().numpy()))

        if log:
            self.log("val/loss",
                    loss,
                    on_step=False,
                    on_epoch=True,
                    reduce_fx='mean',
                    logger=True,
                    prog_bar=True)

        return loss
