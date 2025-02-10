import os
import sys
import yaml

import torch
import torch_geometric
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor

from pharmacomatch.dataset import PharmacophoreDataModule
from pharmacomatch.model import (
    PharmacoMatch,
    CurriculumLearningScheduler,
)


def training():
    # Path variables
    ROOT = os.getcwd()
    PRETRAINING_ROOT = os.path.join(ROOT, "data", "training_data")
    # PRETRAINING_ROOT = "/data/local/drose/PharmacoMatch/training_data"
    CONFIG_FILE_PATH = os.path.join(ROOT, "scripts", "config.yaml")
    MODEL = PharmacoMatch
    params = yaml.load(
        open(CONFIG_FILE_PATH, "r"),
        Loader=yaml.FullLoader,
    )

    # Settings for deterministic training
    torch.set_float32_matmul_precision("medium")
    torch_geometric.seed_everything(params["seed"])
    seed_everything(params["seed"])
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    # Model training
    model = MODEL(**params)
    tb_logger = TensorBoardLogger(
        os.path.join(ROOT, "trained_model", "logs/"),
        # "/data/sharedXL/projects/PharmacoMatch/logs",
        name=f"{MODEL.__name__}",
        default_hp_metric=False,
    )
    callbacks = [
        LearningRateMonitor("epoch"),
        CurriculumLearningScheduler(4, 10),
    ]
    trainer = Trainer(
        devices=1,
        max_epochs=params["epochs"],
        accelerator="auto",
        logger=tb_logger,
        log_every_n_steps=1,
        callbacks=callbacks,
        precision=params["precision"],
        gradient_clip_val=1,
        reload_dataloaders_every_n_epochs=1,
    )
    datamodule = PharmacophoreDataModule(
        PRETRAINING_ROOT,
        batch_size=params["batch_size"],
        small_set_size=params["num_samples"],
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    training()
