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
from losses import WeightedMSELoss
from models.curriculum_scheduler import NoiseScheduler
from data import LitDataModule

class LitS4Model(L.LightningModule):
    def __init__(self, d_input, d_output, encoder: nn.Module, loss='MSELoss',weights=None):
        super().__init__()

        self.encoder = encoder
        self.loss = loss
        self.d_output = d_output
        if self.loss=='MSELoss':
            self.criterion = nn.MSELoss()
        elif self.loss=='WeightedMSELoss':
            self.weights = weights
            self.criterion = WeightedMSELoss(weights=weights)
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
        elif self.loss=='WeightedMSELoss':
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

class LitS4Model_FFT(L.LightningModule):
    def __init__(self, 
                 encoder: nn.Module, 
                 loss='MSELoss', 
                 weights=None, 
                 use_curriculum_learning = False, 
                 max_noise_const = 1.0, 
                 noise_schedule_type = 'linear', 
                 trainer_max_epochs = 100, 
                 **kwargs):
        super().__init__()

        self.encoder = encoder
        self.loss = loss
        self.use_curriculum_learning = use_curriculum_learning
        self.trainer_max_epochs = trainer_max_epochs

        if self.loss=='MSELoss':
            self.criterion = nn.MSELoss()
        elif self.loss=='WeightedMSELoss':
            self.weights = weights
            self.criterion = WeightedMSELoss(weights=weights)
        else: 
            raise ValueError(f'Unknown loss function {self.loss}')

        if self.use_curriculum_learning:
            self.noise_scheduler = NoiseScheduler(
                schedule_type=noise_schedule_type,
                max_noise=max_noise_const,
                total_epochs=trainer_max_epochs
            )

        self.val_outputs = []

        self.save_hyperparameters(ignore=['encoder', 'max_noise_const'])

    def __loss__(self, fft_input, y):
        y_preds = self.forward(fft_input)
        if self.loss=='MSELoss':
            return self.criterion(y, y_preds)
        elif self.loss=='WeightedMSELoss':
            return self.criterion(y, y_preds)

    def forward(self, fft_input):
        x = self.encoder(fft_input)
        return x
   
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx, log=True):
        _, fft_input, y, _ = batch
        loss = self.__loss__(fft_input, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, log=True):
        _, fft_input, y, _ = batch
        loss = self.__loss__(fft_input, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_start(self):
        if not self.use_curriculum_learning:
            return

        current_epoch = self.trainer.current_epoch
        new_noise_const = self.noise_scheduler.get_noise_const(current_epoch)

        if hasattr(self.trainer, "datamodule"):
            self.trainer.datamodule.set_noise_const(new_noise_const)
            self.log("curriculum/noise_const", new_noise_const, on_epoch=True)

class LitS4_CNNModel(L.LightningModule):
    def __init__(self, 
                 encoder: nn.Module,
                 learning_rate = 1e-3, 
                 weight_decay = 0.0,
                 gamma = 0.99,
                 loss = 'MSELoss',
                 weights=None,
                 use_curriculum_learning = False,
                 max_noise_const = 1.0,
                 noise_schedule_type = 'linear',
                 trainer_max_epochs = 100,
                 **kwargs):
        super().__init__()
        
        self.encoder = encoder
        self.loss = loss 
        self.weights = weights
        self.use_curriculum_learning = use_curriculum_learning
        self.trainer_max_epochs = trainer_max_epochs
        self.d_output = self.encoder.output_dim
        
        if self.loss == 'MSELoss':
            self.criterion = nn.MSELoss()
        elif self.loss == 'WeightedMSELoss':
            self.criterion = WeightedMSELoss(weights=weights) 
        else: 
            raise ValueError(f'Unknown loss function {loss}')
            
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gamma = gamma
        
        if self.use_curriculum_learning:
            self.noise_scheduler = NoiseScheduler(
                schedule_type=noise_schedule_type,
                max_noise=max_noise_const,
                total_epochs=trainer_max_epochs
            )
            
        self.save_hyperparameters(ignore=['encoder', 'weights', 'max_noise_const'])

    def __loss__(self, ts_input, fft_input, y):
        y_preds = self.forward(ts_input, fft_input)
        y = y.squeeze(-1)
        y_preds = y_preds.squeeze(-1)
        if self.loss=='MSELoss':
            return self.criterion(y, y_preds)
        elif self.loss=='WeightedMSELoss':
            return self.criterion(y, y_preds)

    def forward(self, ts_input, fft_input):
        x = self.encoder(ts_input, fft_input)
        return x

    def configure_optimizers(self):
        p_wd, p_non_wd, optim_params = [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            
            if hasattr(param, "_optim"):
                optim_dict = param._optim
                group = {"params": [param]}
                if "lr" in optim_dict: group["lr"] = optim_dict["lr"]
                if "weight_decay" in optim_dict: group["weight_decay"] = optim_dict["weight_decay"]
                optim_params.append(group)
                continue
            
            if param.ndim == 1:
                p_non_wd.append(param)
            else:
                p_wd.append(param)
        
        optim_params.extend([
            {"params": p_wd, "weight_decay": self.weight_decay, "lr": self.learning_rate},
            {"params": p_non_wd, "weight_decay": 0.0, "lr": self.learning_rate},
            ])
        
        optimizer = optim.AdamW(optim_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx, log=True):
        ts_input, fft_input, y, _ = batch
        loss = self.__loss__(ts_input, fft_input, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, log=True):
        ts_input, fft_input, y, _ = batch
        loss = self.__loss__(ts_input, fft_input, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def on_train_epoch_start(self):
        if not self.use_curriculum_learning:
            return

        current_epoch = self.trainer.current_epoch 
        new_noise_const = self.noise_scheduler.get_noise_const(current_epoch)
        
        if hasattr(self.trainer, "datamodule"):
            self.trainer.datamodule.set_noise_const(new_noise_const)
            self.log("curriculum/noise_const", new_noise_const, on_epoch=True)
