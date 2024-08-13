import os
import sys
import yaml

import torch
import torch_geometric
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor

from dataset import PharmacophoreDataModule
from model import (
    PharmacoMatch,
    CurriculumLearningScheduler,
)
from utils import load_model_from_path, load_hparams_from_path


def training(device):
    # Path variables
    PROJECT_ROOT = "/data/shared/projects/PhectorDB"
    PRETRAINING_ROOT = f"{PROJECT_ROOT}/training_data"
    CONFIG_FILE_PATH = "/home/drose/git/PhectorDB/pharmaco_match/scripts/config.yaml"
    MODEL = PharmacoMatch
    VERSION = None
    MODEL_PATH = f"{PROJECT_ROOT}/logs/{MODEL.__name__}/version_{VERSION}/"

    # Load or create new model parameters
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

    # Initialize model or load if model exists
    if load_model:
        model = load_model_from_path(MODEL_PATH, MODEL, device[0])
    else:
        model = MODEL(**params)

    tb_logger = TensorBoardLogger(
        f"{PROJECT_ROOT}/logs/", name=f"{MODEL.__name__}", default_hp_metric=False
    )
    callbacks = [
        LearningRateMonitor("epoch"),
        CurriculumLearningScheduler(4, 10),
    ]

    # Model training
    trainer = Trainer(
        devices=device,
        max_epochs=params["epochs"],
        accelerator="auto",
        logger=tb_logger,
        log_every_n_steps=1,
        callbacks=callbacks,
        precision=params["precision"],
        gradient_clip_val=1,
        reload_dataloaders_every_n_epochs=1,
    )

    # Load dataset
    datamodule = PharmacophoreDataModule(
        PRETRAINING_ROOT,
        batch_size=params["batch_size"],
        small_set_size=params["num_samples"],
    )

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    device = [int(i) for i in list(sys.argv[1])]
    training(device)
