import os
import yaml
import sys

sys.path.append("/data/shared/software/CDPKit-head-RH9/Python")


from lightning import Trainer, seed_everything
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
)
import torch
import torch_geometric
from torch_geometric.nn import global_max_pool

from pharmacomatch.dataset import VirtualScreeningDataModule
from pharmacomatch.model import PharmacoMatch
from pharmacomatch.utils import (
    UmapEmbeddingPlotter,
    PcaEmbeddingPlotter,
    bedroc_score,
    enrichment_factor,
    bootstrap_metric,
    load_model_from_path,
)
from pharmacomatch.virtual_screening import (
    VirtualScreeningEmbedder,
    FingerprintEmbedder,
    VirtualScreener,
    ClassicalVirtualScreener,
)

Define path variables
ROOT = os.getcwd()
DATASET_ROOT = os.path.join(ROOT, "data", "DEKOIS20")
RESULTS_LOCATION = os.path.join(ROOT, "results")
if not os.path.exists(RESULTS_LOCATION):
    os.mkdir(RESULTS_LOCATION)
DEVICE = 0
MODEL = PharmacoMatch
MODEL_PATH = os.path.join(ROOT, "trained_model", "trained_model.ckpt")
params = yaml.load(
    open(os.path.join(ROOT, "trained_model", "hparams.yaml"), "r"),
    Loader=yaml.FullLoader,
)

# Path variables
# ROOT = os.getcwd()
# DATASET_ROOT = "/data/sharedXL/projects/PharmacoMatch/LIT-PCBA"
# RESULTS_LOCATION = os.path.join(ROOT, "results_updated", "results_LIT-PCBA_n_point")
# if not os.path.exists(RESULTS_LOCATION):
#     os.mkdir(RESULTS_LOCATION)
# DEVICE = 0
# MODEL = PharmacoMatch
# VERSION = 328
# PROJECT_ROOT = "/data/sharedXL/projects/PharmacoMatch"
# MODEL_PATH = f"{PROJECT_ROOT}/logs/{MODEL.__name__}/version_{VERSION}/"
# params = yaml.load(
#     open(os.path.join(MODEL_PATH, "hparams.yaml"), "r"),
#     Loader=yaml.FullLoader,
# )

# Deterministic flags (should not be necessary for inference, but just in case)
torch.set_float32_matmul_precision("medium")
torch_geometric.seed_everything(params["seed"])
seed_everything(params["seed"])
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

# Load the model and set up trainer for inference
model = MODEL.load_from_checkpoint(
    MODEL_PATH, map_location=torch.device(f"cuda:{DEVICE}")
)
#model = load_model_from_path(MODEL_PATH, MODEL)
trainer = Trainer(
    num_nodes=1,
    devices=1,
    max_epochs=params["epochs"],
    accelerator="auto",
    logger=False,
    log_every_n_steps=2,
)

i = 0
results = []

for TARGET in sorted(os.listdir(DATASET_ROOT)):
    try:
        VS_ROOT = os.path.join(DATASET_ROOT, TARGET)
        print(f"Processing {TARGET}")
        # Setup datamodule
        datamodule = VirtualScreeningDataModule(
            VS_ROOT,
            batch_size=params["batch_size"],
        )
        datamodule.setup()

        # Create embeddings of the VS dataset
        embedder = VirtualScreeningEmbedder(model, datamodule, trainer)
        screener = VirtualScreener(embedder)
        metadata = datamodule.metadata

        experiment_data = dict()
        experiment_data["target"] = TARGET
        experiment_data["embedding_time"] = screener.embedding_time
        experiment_data["matching_time"] = screener.matching_time
        experiment_data["query_num_features"] = metadata.query["num_features"].sum()
        experiment_data["active_ligands"] = (
            torch.max(screener.active_mol_ids).item() + 1
        )
        experiment_data["inactive_ligands"] = (
            torch.max(screener.inactive_mol_ids).item() + 1
        )
        experiment_data["active_conformations"] = len(screener.active_mol_ids)
        experiment_data["inactive_conformations"] = len(screener.inactive_mol_ids)

        # Calc hitlist metrics - order embedding score
        y_true = screener.ligand_label
        y_pred = -screener.ligand_score
        auroc = roc_auc_score(y_true, y_pred)
        auroc_mean, auroc_std = bootstrap_metric(y_true, y_pred, roc_auc_score, 100)
        experiment_data["mean_order_embedding_auroc"] = auroc_mean
        experiment_data["std_order_embedding_auroc"] = auroc_std

        bedroc = bedroc_score(y_true, y_pred)
        bedroc_mean, bedroc_std = bootstrap_metric(y_true, y_pred, bedroc_score, 100)
        experiment_data["mean_order_embedding_bedroc"] = bedroc_mean
        experiment_data["std_order_embedding_bedroc"] = bedroc_std

        bedroc = bedroc_score(y_true, y_pred)
        bedroc_mean, bedroc_std = bootstrap_metric(
            y_true, y_pred, bedroc_score, 100, alpha=80.5
        )
        experiment_data["mean_order_embedding_bedroc80"] = bedroc_mean
        experiment_data["std_order_embedding_bedro80c"] = bedroc_std

        alphas = [0.005, 0.01, 0.05, 0.1]
        for alpha in alphas:
            enrichment_mean, enrichment_std = bootstrap_metric(
                y_true, y_pred, enrichment_factor, 100, alpha=alpha
            )
            experiment_data[f"mean_order_embedding_ef{alpha}"] = enrichment_mean
            experiment_data[f"std_order_embedding_ef{alpha}"] = enrichment_std

        results.append(experiment_data)
        i += 1

    except Exception as e:
        print(e)
        continue

final_results = pd.DataFrame(results)
final_results.to_csv(os.path.join(RESULTS_LOCATION, "prescreening_results.csv"))
