import sys, os
import yaml

import torch
import torch_geometric
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)

from dataset import PharmacophoreDataModule
from model import PharmCLR, VirtualScreeningCallback
from utils import load_model_from_path, load_hparams_from_path


def training(device):
    # Path variables
    PRETRAINING_ROOT = "/data/shared/projects/PhectorDB/chembl_data"
    VS_ROOT = "/data/shared/projects/PhectorDB/virtual_screening_cdk2"
    CONFIG_FILE_PATH = "/home/drose/git/PhectorDB/src/scripts/config.yaml"
    MODEL = PharmCLR
    VERSION = None
    MODEL_PATH = f"logs/PharmCLR/version_{VERSION}/"

    # Check for pretrained model
    if os.path.exists(MODEL_PATH):
        load_model = True
        params = load_hparams_from_path(MODEL_PATH)
    else:
        load_model = False
        params = yaml.load(open(CONFIG_FILE_PATH, "r"), Loader=yaml.FullLoader)

    # Settings for determinism
    torch.set_float32_matmul_precision("medium")
    torch_geometric.seed_everything(params["seed"])
    seed_everything(params["seed"])
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    # Load dataset
    datamodule = PharmacophoreDataModule(
        PRETRAINING_ROOT,
        VS_ROOT,
        batch_size=params["batch_size"],
        small_set_size=params["num_samples"],
    )
    datamodule.setup("fit")

    # Initialize model or load if pretrained model exists
    if load_model:
        model = load_model_from_path(MODEL_PATH, PharmCLR)
    else:
        model = MODEL(**params)

    tb_logger = TensorBoardLogger("logs/", name=f"PharmCLR", default_hp_metric=False)
    callbacks = [
        ModelCheckpoint(monitor="hp/val_loss/dataloader_idx_0", mode="min"),
        LearningRateMonitor("epoch"),
        VirtualScreeningCallback(),
    ]

    # Model training
    trainer = Trainer(
        devices=device,
        max_epochs=params["epochs"],
        accelerator="auto",
        logger=tb_logger,
        log_every_n_steps=1,
        callbacks=callbacks,
        precision=params['precision'],
    )

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    device = [int(i) for i in list(sys.argv[1])]
    training(device)
