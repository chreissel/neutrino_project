import torch.nn as nn
import numpy as np
import torch

class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.weights = torch.tensor(weights).to(self.device)

    def forward(self, inputs, targets):
        return ((inputs - targets)**2 * self.weights).mean()

class EfficiencyWeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs, targets, w):
        if w.dim() == 1:
            w = w.unsqueeze(1)
        per_sample_loss = (inputs - targets) ** 2
        weighted_loss = (w * per_sample_loss).sum() / w.sum()
        return weighted_loss
