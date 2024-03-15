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
from virtual_screening import VirtualScreeningEmbedder
from scripts import VirtualScreeningExperiment, SelfSimilarityEvaluation


def training(device):
    # Path variables
    PROJECT_ROOT = "/data/shared/projects/PhectorDB"
    PRETRAINING_ROOT = f"{PROJECT_ROOT}/training_data"
    VS_ROOT = f"{PROJECT_ROOT}/litpcba/ESR1_ant"
    CONFIG_FILE_PATH = "src/scripts/config.yaml"
    MODEL = PharmCLR
    VERSION = None
    MODEL_PATH = f"{PROJECT_ROOT}/logs/{MODEL.__name__}/version_{VERSION}/"

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

    tb_logger = TensorBoardLogger(
        f"{PROJECT_ROOT}/logs/", name=f"{MODEL.__name__}", default_hp_metric=False
    )
    callbacks = [
        ModelCheckpoint(monitor="val/val_loss/dataloader_idx_0", mode="min"),
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
        precision=params["precision"],
    )

    trainer.fit(model=model, datamodule=datamodule)
    model = PharmCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    embedder = VirtualScreeningEmbedder(model, datamodule, trainer)
    vs = VirtualScreeningExperiment(embedder, trainer.logger.version)
    vs()
    eval = SelfSimilarityEvaluation(model, datamodule.create_val_dataloader(), device)
    eval.calculate_mean_similarities(trainer.logger.version)


if __name__ == "__main__":
    device = [int(i) for i in list(sys.argv[1])]

    # for value in [0.05, 0.1, 0.2]:
    #     key = 'temperature'
    #     training(device, key, value)

    # for value in [0.01, 0.025, 0.1]:
    #     key = 'learning_rate'
    #     training(device, key, value)

    # for value in [64, 1024, 4096]:
    #     key = 'batch_size'
    #     training(device, key, value)

    # for value in [32, 64, 256]:
    #     key = 'hidden_dim_encoder'
    #     training(device, key, value)

    # for value in [0.1]:
    #    key = 'node_masking'
    training(device)

    # for value in [1, 2, 4]:
    #     key = 'n_layers_conv'
    #     training(device, key, value)
