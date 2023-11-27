import sys
import os
import yaml

import torch
import torch_geometric
from lightning import Trainer, seed_everything

from utils import *
from dataset import *
from model import *
from .utils import load_model_from_path


def evaluation(device):
    PRETRAINING_ROOT = "/data/shared/projects/PhectorDB/chembl_data"
    VS_ROOT = "/data/shared/projects/PhectorDB/virtual_screening_cdk2"
    CONFIG_FILE_PATH = "/home/drose/git/PhectorDB/src/scripts/config.yaml"
    MODEL = PharmCLR
    VS_MODEL_NUMBER = 36
    MODEL_PATH = f"logs/PharmCLR/version_{VS_MODEL_NUMBER}/checkpoints/"

    params = yaml.load(open(CONFIG_FILE_PATH, "r"), Loader=yaml.FullLoader)

    torch.set_float32_matmul_precision("medium")
    torch_geometric.seed_everything(params["seed"])
    seed_everything(params["seed"])
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    datamodule = PharmacophoreDataModule(
        PRETRAINING_ROOT,
        VS_ROOT,
        batch_size=params["batch_size"],
        small_set_size=params["num_samples"],
    )

    model = load_model_from_path(MODEL_PATH, MODEL)
    datamodule.setup()

    trainer = Trainer(
        num_nodes=1,
        devices=device,
        max_epochs=params["epochs"],
        accelerator="auto",
        logger=False,
        log_every_n_steps=1,
    )

    vs = VirtualScreening(model, trainer)
    vs(datamodule)


if __name__ == "__main__":
    device = [int(i) for i in list(sys.argv[1])]
    evaluation(device)
