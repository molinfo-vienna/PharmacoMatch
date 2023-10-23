import torchmetrics
from lightning import LightningModule
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class CustomLightningModule(LightningModule):
    def __init__(self, hyperparams, params):
        super(CustomLightningModule, self).__init__()
        self.save_hyperparameters()

    def on_train_start(self):
        self.logger.log_hyperparams(
            self.hparams,
            {
                "hp/train_loss": 1,
                "hp/val_loss": 1,
            },
        )

    def training_step(self, batch, batch_idx):
        out1 = self(batch)
        out2 = self(batch)
        batch_size, _ = out1.shape
        out = torch.cat((out1, out2), dim=1).reshape(batch_size * 2, -1)
        loss = self.nt_xent_loss(out)
        self.log(
            "hp/train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )

        return loss

    def validation_step(self, batch, batch_idx):
        out1 = self(batch)
        out2 = self(batch)
        batch_size, _ = out1.shape
        out = torch.cat((out1, out2), dim=1).reshape(batch_size * 2, -1)
        val_loss = self.nt_xent_loss(out)
        self.log(
            "hp/val_loss",
            val_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )
        return val_loss
