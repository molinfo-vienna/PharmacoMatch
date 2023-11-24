import sys
import os
import yaml

import torch
import torch_geometric
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, Callback

from utils import *
from dataset import *
from model import *


def training(device):
    PRETRAINING_ROOT = "/data/shared/projects/PhectorDB/chembl_data"
    VS_ROOT = "/data/shared/projects/PhectorDB/virtual_screening_cdk2"
    CONFIG_FILE_PATH = "/home/drose/git/PhectorDB/src/scripts/config.yaml"
    MODEL = PharmCLR

    params = yaml.load(open(CONFIG_FILE_PATH, "r"), Loader=yaml.FullLoader)

    torch.set_float32_matmul_precision("medium")
    torch_geometric.seed_everything(params["seed"])
    seed_everything(params["seed"])
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    datamodule = PharmacophoreDataModule(
        PRETRAINING_ROOT, VS_ROOT, batch_size=params["batch_size"], small_set_size=params["num_samples"])
    datamodule.setup("fit")
    model = MODEL(**params)
    tb_logger = TensorBoardLogger(
        "logs/", name=f"PharmCLR", default_hp_metric=False
    )
    callbacks = [ModelCheckpoint(monitor="hp/val_loss/dataloader_idx_0", mode="min"),
                 LearningRateMonitor("epoch"),
                 VirtualScreeningCallback()]

    trainer = Trainer(
        devices=device,
        max_epochs=params["epochs"],
        accelerator="auto",
        logger=tb_logger,
        log_every_n_steps=1,
        callbacks=callbacks,
        precision=16
    )

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    device = [int(i) for i in list(sys.argv[1])]
    training(device)
