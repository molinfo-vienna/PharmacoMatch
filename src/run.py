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
    PRETRAINING_ROOT = "/data/shared/projects/PhectorDB/pretraining_data/small"
    VS_ROOT = "/data/shared/projects/PhectorDB/virtual_screening_cdk2"
    EPOCHS = 1000
    TRAINING = False
    BATCH_SIZE = 512
    MODEL = PharmCLR
    torch.set_float32_matmul_precision("medium")
    torch_geometric.seed_everything(42)
    datamodule = PharmacophoreDataModule(PRETRAINING_ROOT, VS_ROOT, batch_size=BATCH_SIZE)

    def training():
        datamodule.setup("fit")
        params = datamodule.params # dict(num_node_features=9, num_edge_features=5)
        hyperparams = MODEL.get_hyperparams()
        model = MODEL(hyperparams, params)

        tb_logger = TensorBoardLogger(
            "logs/", name=f"PharmCLR", default_hp_metric=False
        )

        callbacks = []

        trainer = Trainer(
            num_nodes=1,
            devices=[device],
            max_epochs=EPOCHS,
            accelerator="gpu",
            logger=tb_logger,
            log_every_n_steps=1,
            callbacks=callbacks,
        )

        trainer.fit(model=model, datamodule=datamodule)

    def testing():
        # load the trained model
        def load_model(path):
            for file in os.listdir(path):
                if file.endswith(".ckpt"):
                    path = os.path.join(path, file)
            return MODEL.load_from_checkpoint(path)
        
        path = f'logs/PharmCLR/version_4/checkpoints/'
        model = load_model(path)
        datamodule.setup('virtual_screening')
        trainer = Trainer(num_nodes=1,
                    devices=[device],
                    max_epochs=EPOCHS,
                    accelerator='auto',
                    logger=False,
                    log_every_n_steps=1)
        
        vs = VirtualScreening(model, trainer)
        vs(datamodule)

    if TRAINING:
        training()
    else:
        testing()


if __name__ == "__main__":
    device = int(sys.argv[1])
    run(device)
