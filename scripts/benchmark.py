import os
import yaml

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
    VirtualScreener,
    ClassicalVirtualScreener,
)

# Define path variables
ROOT = os.getcwd()
DATASET_ROOT = os.path.join(ROOT, "data", "DUD-E")
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

# # Path variables
# ROOT = os.getcwd()
# DATASET_ROOT = "/data/sharedXL/projects/PharmacoMatch/DUD-E"
# VERSION = 328
# RESULTS_LOCATION = os.path.join(ROOT, "results_updated", f"results_{VERSION}")
# if not os.path.exists(RESULTS_LOCATION):
#     os.mkdir(RESULTS_LOCATION)
# DEVICE = 0
# MODEL = PharmacoMatch
# PROJECT_ROOT = "/data/sharedXL/projects/PharmacoMatch"
# MODEL_PATH = f"{PROJECT_ROOT}/logs/{MODEL.__name__}/version_{VERSION}/"
# params = yaml.load(
#     open(os.path.join(MODEL_PATH, "hparams.yaml"), "r"),
#     Loader=yaml.FullLoader,
# )

N_BOOTSTRAPS = 100

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
    devices=[DEVICE],
    max_epochs=params["epochs"],
    accelerator="auto",
    logger=False,
    log_every_n_steps=2,
)

plt.rcParams.update({"figure.max_open_warning": 0})
fig1, axes1 = plt.subplots(2, 5, figsize=(25, 10), sharex=True, sharey=True)
fig2, axes2 = plt.subplots(2, 5, figsize=(25, 10), sharex=True, sharey=True)
i = 0
results = []

for TARGET in sorted(os.listdir(DATASET_ROOT)):
    try:
        VS_ROOT = os.path.join(DATASET_ROOT, TARGET)

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
        classical_screener = ClassicalVirtualScreener(datamodule)

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

        # Comparison of order embedding with classical screener
        y_true = classical_screener.matches
        y_pred = -screener.conformation_score

        # Calc metric
        auroc = roc_auc_score(y_true, y_pred)
        auroc_mean, auroc_std = bootstrap_metric(
            y_true, y_pred, roc_auc_score, N_BOOTSTRAPS
        )
        experiment_data["mean_auroc_comparison"] = auroc_mean
        experiment_data["std_auroc_comparison"] = auroc_std
        fpr, tpr, threshold = roc_curve(y_true, y_pred)

        axes1[i // 5][i % 5].plot(fpr, tpr)
        axes1[i // 5][i % 5].set_title(f"{TARGET} (AUROC = {auroc:.2f})")
        axes1[i // 5][i % 5].set_xlabel("False positive rate")
        axes1[i // 5][i % 5].set_ylabel("True positive rate")

        # Calc hitlist metrics - order embedding score
        y_true = screener.ligand_label
        y_pred = -screener.ligand_score
        auroc = roc_auc_score(y_true, y_pred)
        auroc_mean, auroc_std = bootstrap_metric(
            y_true, y_pred, roc_auc_score, N_BOOTSTRAPS
        )
        bedroc = bedroc_score(y_true, y_pred)
        bedroc_mean, bedroc_std = bootstrap_metric(
            y_true, y_pred, bedroc_score, N_BOOTSTRAPS
        )

        experiment_data["mean_order_embedding_auroc"] = auroc_mean
        experiment_data["std_order_embedding_auroc"] = auroc_std
        experiment_data["mean_order_embedding_bedroc"] = bedroc_mean
        experiment_data["std_order_embedding_bedroc"] = bedroc_std

        alphas = [0.01, 0.05, 0.1]
        for alpha in alphas:
            enrichment_mean, enrichment_std = bootstrap_metric(
                y_true, y_pred, enrichment_factor, N_BOOTSTRAPS, alpha=alpha
            )
            experiment_data[f"mean_order_embedding_ef{alpha}"] = enrichment_mean
            experiment_data[f"std_order_embedding_ef{alpha}"] = enrichment_std

        fpr, tpr, threshold = roc_curve(y_true, y_pred)
        axes2[i // 5][i % 5].plot(fpr, tpr)
        axes2[i // 5][i % 5].set_xlabel("False positive rate")
        axes2[i // 5][i % 5].set_ylabel("True positive rate")

        # Calc hitlist metrics - CDPKit score
        y_true = screener.ligand_label
        y_pred = global_max_pool(classical_screener.alignment_score, screener.mol_ids)
        auroc_cdp = roc_auc_score(y_true, y_pred)
        auroc_mean, auroc_std = bootstrap_metric(
            y_true, y_pred, roc_auc_score, N_BOOTSTRAPS
        )
        bedroc_cdp = bedroc_score(y_true, y_pred)
        bedroc_mean, bedroc_std = bootstrap_metric(
            y_true, y_pred, bedroc_score, N_BOOTSTRAPS
        )
        experiment_data["mean_cdpkit_auroc"] = auroc_mean
        experiment_data["std_cdpkit_auroc"] = auroc_std
        experiment_data["mean_cdpkit_bedroc"] = bedroc_mean
        experiment_data["std_cdpkit_bedroc"] = bedroc_std

        for alpha in alphas:
            enrichment_mean, enrichment_std = bootstrap_metric(
                y_true, y_pred, enrichment_factor, N_BOOTSTRAPS, alpha=alpha
            )
            experiment_data[f"mean_cdpkit_ef{alpha}"] = enrichment_mean
            experiment_data[f"std_cdpkit_ef{alpha}"] = enrichment_std

        fpr, tpr, threshold = roc_curve(y_true, y_pred)
        axes2[i // 5][i % 5].plot(fpr, tpr)
        axes2[i // 5][i % 5].set_title(f"{TARGET}")
        axes2[i // 5][i % 5].legend(
            [
                f"Order Embedding (AUROC = {auroc:.2f})",
                f"CDPKit (AUROC = {auroc_cdp:.2f})",
            ],
            loc="lower right",
        )
        axes2[i // 5][i % 5].set_xlabel("False positive rate")
        axes2[i // 5][i % 5].set_ylabel("True positive rate")

        results.append(experiment_data)
        i += 1

        # PCA order embedding space plot
        pca_plotter = PcaEmbeddingPlotter(screener, metadata)
        fig3 = pca_plotter.create_pca_plot()
        fig3.savefig(os.path.join(RESULTS_LOCATION, f"PCA_{TARGET}.png"), dpi=150)

        # UMAP embedding space visualization
        umap_plotter = UmapEmbeddingPlotter(screener, metadata)
        fig4 = umap_plotter.create_umap_plot()
        fig4.savefig(
            os.path.join(RESULTS_LOCATION, f"UMAP_{TARGET}.png"),
            dpi=150,
            bbox_inches="tight",
        )

    except Exception as e:
        print(e)
        continue

fig1.savefig(
    os.path.join(RESULTS_LOCATION, "comparison.png"), dpi=150, bbox_inches="tight"
)
fig2.savefig(
    os.path.join(RESULTS_LOCATION, "hitlist.png"), dpi=150, bbox_inches="tight"
)
final_results = pd.DataFrame(results)
final_results.to_csv(os.path.join(RESULTS_LOCATION, "results_dude.csv"))
