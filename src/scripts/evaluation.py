import sys, os
from typing import Any
import yaml

from lightning import Trainer, seed_everything
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from torch_geometric.nn import global_mean_pool
import torch
import torch_geometric
import umap
import numpy as np

from dataset import PharmacophoreDataModule
from model import PharmCLR
from virtual_screening import VirtualScreener, VirtualScreeningEmbedder
from utils import load_model_from_path


class VirtualScreeningExperiment:
    def __init__(self, embedder, model_number) -> None:
        self.screener = VirtualScreener(embedder)
        self.model_number = model_number

    def __call__(self) -> None:
        # plot UMAP of highest scoring active and inactive embeddings
        self._plot_UMAP(
            self.screener.query_embedding,
            self.screener.top_active_embeddings,
            self.screener.top_inactive_embeddings,
            self.screener.active_embeddings,
            self.screener.inactive_embeddings,
            self.screener.active_mol_ids,
            self.screener.inactive_mol_ids,
        )

        # Map similarity [-1, 1] --> [0, 1] and print AUC statistics
        y_true = self.screener.y_true.cpu().numpy()
        y_pred = self.screener.y_pred.cpu().numpy()

        self._plot_auc(y_true, y_pred)

    def _plot_auc(self, y_true, y_pred):
        # print metrics & plot figures
        print(roc_auc_score(y_true, y_pred))
        fpr, tpr, thr = roc_curve(y_true, y_pred)
        fig2 = plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC LigandScout Tutorial")
        plt.savefig("plots/auroc.png")
        fig3 = plt.figure()
        precision, recall, thr = precision_recall_curve(y_true, y_pred)
        plt.plot(precision, recall)
        plt.savefig("plots/prcurve.png")

    def _plot_UMAP(
        self,
        query_embedding,
        top_active_embeddings,
        top_inactive_embeddings,
        active_embeddings,
        inactive_embeddings,
        active_mol_ids,
        inactive_mol_ids,
    ):
        mean_actives = global_mean_pool(active_embeddings, active_mol_ids)
        mean_inactives = global_mean_pool(inactive_embeddings, inactive_mol_ids)

        reducer = umap.UMAP(metric="cosine")
        reducer.fit(torch.cat((mean_actives, mean_inactives, query_embedding)))
        reduced_inactive_embeddings = reducer.transform(inactive_embeddings)
        reduced_active_embeddings = reducer.transform(active_embeddings)
        reduced_top_inactive_embeddings = reducer.transform(top_inactive_embeddings)
        reduced_top_active_embeddings = reducer.transform(top_active_embeddings)
        reduced_query_embedding = reducer.transform(query_embedding)

        fig = plt.figure(figsize=(15, 15))
        plt.scatter(
            reduced_inactive_embeddings[:, 0],
            reduced_inactive_embeddings[:, 1],
            c="cornflowerblue",
            marker="o",
            s=10,
        )
        plt.scatter(
            reduced_active_embeddings[:, 0],
            reduced_active_embeddings[:, 1],
            c="lightcoral",
            marker="o",
            s=10,
        )
        plt.scatter(
            reduced_top_inactive_embeddings[:, 0],
            reduced_top_inactive_embeddings[:, 1],
            c="blue",
            marker="o",
            edgecolor="darkblue",
            s=20,
        )
        plt.scatter(
            reduced_top_active_embeddings[:, 0],
            reduced_top_active_embeddings[:, 1],
            c="red",
            marker="o",
            edgecolor="darkred",
            s=20,
        )
        plt.scatter(
            reduced_query_embedding[:, 0],
            reduced_query_embedding[:, 1],
            c="yellow",
            marker="*",
            s=300,
            edgecolor="black",
        )
        plt.legend(
            [
                "Inactive Conformation (CDK2)",
                "Active Conformation (CDK2)",
                "Inactive Conformation, compound-wise highest query similarity",
                "Active Conformation, compound-wise highest query similarity",
                "Query (Structure-based pharamcophore of 1ke7)",
            ]
        )
        plt.title("UMAP of PharmCLR Embedding Space")
        plt.savefig(f"umap{self.model_number}.png", dpi=250)


def evaluation(device):
    PRETRAINING_ROOT = "/data/shared/projects/PhectorDB/chembl_data"
    VS_ROOT = "/data/shared/projects/PhectorDB/virtual_screening_cdk2"
    MODEL = PharmCLR
    VS_MODEL_NUMBER = 23
    MODEL_PATH = f"logs/PharmCLR/version_{VS_MODEL_NUMBER}/"

    params = yaml.load(
        open(os.path.join(MODEL_PATH, "hparams.yaml"), "r"), Loader=yaml.FullLoader
    )

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
    datamodule.setup()

    model = load_model_from_path(MODEL_PATH, MODEL)
    device = [model.device.index]

    trainer = Trainer(
        num_nodes=1,
        devices=device,
        max_epochs=params["epochs"],
        accelerator="auto",
        logger=False,
        log_every_n_steps=1,
    )

    embedder = VirtualScreeningEmbedder(model, datamodule, trainer)
    vs = VirtualScreeningExperiment(embedder, VS_MODEL_NUMBER)
    vs()


if __name__ == "__main__":
    device = [int(i) for i in list(sys.argv[1])]
    evaluation(device)
