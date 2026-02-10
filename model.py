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
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, CosineAnnealingLR

class BaseS4LightningModule(L.LightningModule):
    def __init__(self, 
                 encoder: nn.Module, 
                 loss='MSELoss', 
                 weights=None, 
                 use_curriculum_learning=False, 
                 max_noise_const=1.0, 
                 noise_schedule_type='linear', 
                 trainer_max_epochs=100,
                 learning_rate=1e-3, 
                 weight_decay=0.0,
                 gamma=0.99,
                 lr_schedule_type='exponential',
                 lr_patience=5,
                 lr_min=1e-6,
                 threshold=1e-4,
                 freeze_branches=False,
                 **kwargs):
        super().__init__()

        self.encoder = encoder
        self.loss = loss
        self.use_curriculum_learning = use_curriculum_learning
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.lr_schedule_type = lr_schedule_type
        self.lr_patience = lr_patience
        self.lr_min = lr_min
        self.threshold = threshold
        self.freeze_branches = freeze_branches

        # 1. Initialize Loss Functioni
        if self.loss == 'MSELoss':
            self.criterion = nn.MSELoss()
        elif self.loss == 'WeightedMSELoss':
            self.criterion = WeightedMSELoss(weights=weights)
        elif self.loss == 'GaussianNLLLoss':
            # NOTE: For GaussianNLLLoss, the encoder output must be 2*d_output
            self.criterion = nn.GaussianNLLLoss(reduction='mean', full=False, eps=1e-6)
        else: 
            raise ValueError(f'Unknown loss function {self.loss}')

        # 2. Setup Curriculum Learning Scheduler
        if self.use_curriculum_learning:
            self.noise_scheduler = NoiseScheduler(
                schedule_type=noise_schedule_type,
                max_noise=max_noise_const,
                total_epochs=trainer_max_epochs
            )

        #self.save_hyperparameters(ignore=['encoder', 'weights', 'max_noise_const'])
        self.save_hyperparameters(ignore=[])

    def __loss__(self, y, y_preds):
        """Calculates the loss based on the initialized criterion."""
        # Handle 1D vs 2D output targets consistently
        if y.ndim > 1 and y.shape[-1] == 1:
            y = y.squeeze(-1)
        if y_preds.ndim > 1 and y_preds.shape[-1] == 1:
            y_preds = y_preds.squeeze(-1)
            
        return self.criterion(y_preds, y) # PyTorch MSELoss is typically (input, target)

    def configure_optimizers(self):
        """
        Configures the optimizer and selected LR scheduler based on self.lr_schedule_type.
        """
        p_wd, p_non_wd, optim_params = [], [], []
        learning_rate = self.hparams.learning_rate
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
            if param.ndim == 1 or 'norm' in name.lower() or 'bias' in name.lower():
                p_non_wd.append(param)
            else:
                p_wd.append(param)
        
        optim_params.extend([
            {"params": p_wd, "weight_decay": self.weight_decay, "lr": learning_rate},
            {"params": p_non_wd, "weight_decay": 0.0, "lr": learning_rate},
        ])
        
        optimizer = optim.AdamW(optim_params, lr=learning_rate, weight_decay=self.weight_decay)
        if self.lr_schedule_type == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(
                optimizer, 
                mode='min',
                factor=self.gamma,
                patience=self.lr_patience,
                min_lr=self.lr_min,
                threshold=self.threshold
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                    "reduce_on_plateau": True,
                },
            }
        
        elif self.lr_schedule_type == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.trainer_max_epochs,
                eta_min=self.lr_min
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        else:
            scheduler = ExponentialLR(optimizer, gamma=self.gamma)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_train_epoch_start(self):
        if not self.use_curriculum_learning:
            return

        current_epoch = self.trainer.current_epoch
        new_noise_const = self.noise_scheduler.get_noise_const(current_epoch)
        
        if hasattr(self.trainer, "datamodule"):
            self.trainer.datamodule.set_noise_const(new_noise_const)
            self.log("curriculum/noise_const", new_noise_const, on_epoch=True)


class LitS4Model(BaseS4LightningModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        #self.save_hyperparameters(ignore=['encoder', 'weights', 'max_noise_const'])
        self.save_hyperparameters(ignore=[])

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_preds = self.forward(x) 
        loss = self.__loss__(y, y_preds)
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("lr", current_lr, on_step=False, on_epoch=True, logger=True)
        self.log(f"train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_preds = self.forward(x)
        loss = self.__loss__(y, y_preds)
        self.log(f"val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss


class LitS4DualModel(BaseS4LightningModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.freeze_branches:
            self.encoder.ts_branch.eval()
            self.encoder.fft_branch.eval()
            for param in self.encoder.ts_branch.parameters():
                param.requires_grad = False
            for param in self.encoder.fft_branch.parameters():
                param.requires_grad = False

    def forward(self, ts_input, fft_input):
        return self.encoder(ts_input, fft_input)

    def training_step(self, batch, batch_idx):
        ts_input, fft_input, y, _ = batch
        y_preds = self.forward(ts_input, fft_input)
        loss = self.__loss__(y, y_preds)
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("lr", current_lr, on_step=False, on_epoch=True, logger=True)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ts_input, fft_input, y, _ = batch
        y_preds = self.forward(ts_input, fft_input)
        loss = self.__loss__(y, y_preds)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
