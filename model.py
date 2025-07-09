from models.s4d import S4D
import torch.nn as nn
dropout_fn = nn.Dropout2d
import lightning as L

# --- Definition of S4D model: get columns, shapes, and total rows ---
class LitS4Model(L.LightningModule, **kwargs):

    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
    ):
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
        return x

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return optimizer, scheduler

    def training_step(self, batch, batch_idx, log=True):
        X, y = batch
        y_hat = self.forward(X)
        loss = nn.MSELoss(y_hat, y)

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
        X, y = batch
        y_hat = self.forward(X)
        loss = nn.MSELoss(y_hat, y)

        if log:
            self.log("val/loss",
                    loss,
                    on_step=False,
                    on_epoch=True,
                    reduce_fx='mean',
                    logger=True,
                    prog_bar=True)

        return loss

