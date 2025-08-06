from models.s4d import S4D
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
dropout_fn = nn.Dropout2d
import lightning as L
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt

class LitS4Model(L.LightningModule):
    def __init__(self, d_input, d_output, variables, d_model=256, n_layers=4,
                dropout=0.2, prenorm=False, loss='GaussianNLLLoss'):
        super().__init__()
        self.prenorm = prenorm
        self.encoder = nn.Linear(d_input, d_model)
        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, 0.01))
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))
        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

        self.loss = loss
        self.variables = variables
        # self.d_output = d_output
        self.d_output = len(self.variables)
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
            return self.criterion(X, y_preds)
        elif self.loss=='GaussianNLLLoss':
            y_hat, y_hat_variance = y_preds[:,:self.d_split], y_preds[:,self.d_split:]
            print(y_hat.shape)
            print(y_hat_variance.shape)
            return self.criterion(y_hat, y, y_hat_variance)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)
        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)
            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)
            # Dropout on the output of the S4 block
            z = dropout(z)
            # Residual connection
            x = z + x
            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(-1, -2)
        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)
        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)
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
