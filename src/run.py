import sys
import os

import torch
import torch_geometric
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from utils import *
from dataset import *
from model import *


def run(device):
    PRETRAINING_ROOT = "/data/shared/projects/PhectorDB/chembl_data"
    VS_ROOT = "/data/shared/projects/PhectorDB/virtual_screening_cdk2"
    EPOCHS = 100
    TRAINING = True
    BATCH_SIZE = 512
    SMALL_SET_SIZE = 100000
    MODEL = PharmCLR
    VS_MODEL_NUMBER = 19
    SEED = 42
    torch.set_float32_matmul_precision("medium")
    torch_geometric.seed_everything(SEED)
    seed_everything(SEED)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    datamodule = PharmacophoreDataModule(PRETRAINING_ROOT, VS_ROOT, batch_size=BATCH_SIZE, small_set_size=SMALL_SET_SIZE)

    def training():
        datamodule.setup("fit")
        params = datamodule.params # dict(num_node_features=9, num_edge_features=5)
        params["batch_size"] = BATCH_SIZE
        hyperparams = MODEL.get_hyperparams()
        params.update(hyperparams)
        model = MODEL(**params)

        tb_logger = TensorBoardLogger(
            "logs/", name=f"PharmCLR", default_hp_metric=False
        )

        callbacks = [#ModelCheckpoint(monitor="hp/rank_me", mode="max"), 
                     LearningRateMonitor("epoch")]

        trainer = Trainer(
            num_nodes=1,
            devices=device,
            max_epochs=EPOCHS,
            accelerator="auto",
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
        
        path = f'logs/PharmCLR/version_{VS_MODEL_NUMBER}/checkpoints/'
        model = load_model(path)
        datamodule.setup('virtual_screening')
        trainer = Trainer(num_nodes=1,
                    devices=device,
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
    device = [int(i) for i in list(sys.argv[1])]
    run(device)
