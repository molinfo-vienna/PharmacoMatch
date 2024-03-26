import os
import sys
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
from model import (
    PharmCLR,
    VirtualScreeningCallback,
    PhectorMatch,
    CurriculumLearningScheduler,
)
from scripts import VirtualScreeningExperiment, SelfSimilarityEvaluation
from utils import load_model_from_path, load_hparams_from_path
from virtual_screening import VirtualScreeningEmbedder


def training(device):
    # Path variables
    PROJECT_ROOT = "/data/shared/projects/PhectorDB"
    PRETRAINING_ROOT = f"{PROJECT_ROOT}/training_data"
    VS_ROOT = f"{PROJECT_ROOT}/litpcba/ESR1_ant"
    CONFIG_FILE_PATH = "phector_db/scripts/config.yaml"
    MODEL = PhectorMatch
    VERSION = None
    MODEL_PATH = f"{PROJECT_ROOT}/logs/{MODEL.__name__}/version_{VERSION}/"
    PRETRAINED_MODEL = PharmCLR
    PRETRAINED_VERSION = 23
    PRETRAINED_MODEL_PATH = (
        f"{PROJECT_ROOT}/archived/old_logs_2/{PRETRAINED_MODEL.__name__}/version_{PRETRAINED_VERSION}/"
        # f"{PROJECT_ROOT}/logs/{PRETRAINED_MODEL.__name__}/version_{PRETRAINED_VERSION}/"
    )

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
        model = load_model_from_path(MODEL_PATH, MODEL)
    else:
        model = MODEL(**params)

    # Check for pretrained model
    if os.path.exists(PRETRAINED_MODEL_PATH):
        pretrained_model = load_model_from_path(PRETRAINED_MODEL_PATH, PRETRAINED_MODEL)
        model.encoder = pretrained_model.encoder
        print("Using pretrained encoder")

    tb_logger = TensorBoardLogger(
        f"{PROJECT_ROOT}/logs/", name=f"{MODEL.__name__}", default_hp_metric=False
    )
    callbacks = [
        # ModelCheckpoint(monitor="val/val_loss", mode="min"),
        LearningRateMonitor("epoch"),
        CurriculumLearningScheduler(4, 20),
        # VirtualScreeningCallback(),
    ]

    # for i in range(5, 20):
    # Model training
    trainer = Trainer(
        devices=device,
        max_epochs=params["epochs"],
        accelerator="auto",
        logger=tb_logger,
        log_every_n_steps=1,
        callbacks=callbacks,
        precision=params["precision"],
        gradient_clip_val=0.5,
        reload_dataloaders_every_n_epochs=1,
    )

    # Load dataset
    datamodule = PharmacophoreDataModule(
        PRETRAINING_ROOT,
        VS_ROOT,
        batch_size=params["batch_size"],
        small_set_size=params["num_samples"],
        graph_size_upper_bound=4,
    )
    datamodule.setup("fit")

    trainer.fit(model=model, datamodule=datamodule)
    # model = trainer.model
    # model = PharmCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    # embedder = VirtualScreeningEmbedder(model, datamodule, trainer)
    # vs = VirtualScreeningExperiment(embedder, trainer.logger.version)
    # vs()
    # eval = SelfSimilarityEvaluation(model, datamodule.create_val_dataloader(), device)
    # eval.calculate_mean_similarities(trainer.logger.version)


if __name__ == "__main__":
    device = [int(i) for i in list(sys.argv[1])]
    training(device)
