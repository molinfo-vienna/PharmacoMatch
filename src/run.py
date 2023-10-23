import sys
import os

import torch
import torch_geometric

from lightning import Trainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from utils import *
from dataset import *
from model import *


def run(device):
    ROOT = "/data/shared/projects/PhectorDB/pretraining_data/small"
    EPOCHS = 1000
    TEST_SET_EVAL = True
    # MODEL = GCN
    torch.set_float32_matmul_precision("medium")
    torch_geometric.seed_everything(42)
    datamodule = PharmacophoreDataModule(ROOT)
    datamodule.setup("fit")


if __name__ == "__main__":
    device = int(sys.argv[1])
    run(device)
