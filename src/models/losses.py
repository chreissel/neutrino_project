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


class MixtureMSESpectralLoss(nn.Module):
    """Mixture of MSE loss (time domain) and spectral loss (frequency domain).

    The spectral component compares the magnitude of the real FFT of the
    predicted and target sequences along the sequence dimension (dim=1).

    Args:
        alpha: Weight for the MSE term.  The spectral term is weighted by
            ``(1 - alpha)``.  Defaults to ``0.5``.
    """

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(inputs, targets)

        # Spectral loss: compare magnitude spectra along the sequence dimension
        inputs_mag = torch.abs(torch.fft.rfft(inputs, dim=1))
        targets_mag = torch.abs(torch.fft.rfft(targets, dim=1))
        spectral_loss = self.mse(inputs_mag, targets_mag)

        return self.alpha * mse_loss + (1.0 - self.alpha) * spectral_loss
