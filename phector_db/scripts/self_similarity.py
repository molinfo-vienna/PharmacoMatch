import os
import sys
import yaml

from lightning import Trainer, seed_everything
import numpy as np
from torch.nn.functional import cosine_similarity
import torch
import torch_geometric
from matplotlib import cm
import matplotlib.pyplot as plt

from dataset import PharmacophoreDataModule
from model import PharmCLR, ValidationDataTransformSetter, PhectorMatch
from utils import load_model_from_path


class SelfSimilarityEvaluation:
    def __init__(self, model, dataloader, device) -> None:
        self.model = model
        self.dataloader = dataloader
        self.device = device

        self.max_node_masking = 0.9
        steps_node_masking = 10
        self.max_radius = 5
        steps_radius = 6

        self.node_masking_range = [
            float(i)
            for i in torch.linspace(0, self.max_node_masking, steps_node_masking)
        ]
        self.radius_range = [
            float(i) for i in torch.linspace(0, self.max_radius, steps_radius)
        ]
        self.self_similarity = np.zeros((steps_node_masking, steps_radius))
        self.subgraph_isomorphism = np.zeros(steps_radius)
        self.subgraph_isomorphism_target_query = np.zeros(steps_radius)

    def subgraph_isomorphism_evaluation(self, num_version):
        target = self._create_embeddings(0, 0, None)
        for i, radius in enumerate(self.radius_range):
            queries = self._create_embeddings(
                node_masking=1, radius=radius, node_to_keep_lower_bound=3
            )
            threshold = 0.1
            self.subgraph_isomorphism[i] = torch.sum(
                torch.sum(
                    torch.max(
                        torch.zeros_like(target),
                        queries - target,
                    )
                    ** 2,
                    dim=1,
                )
                <= threshold
            ) / len(queries)

            self.subgraph_isomorphism_target_query[i] = torch.sum(
                torch.sum(
                    torch.max(
                        torch.zeros_like(target),
                        target - queries,
                    )
                    ** 2,
                    dim=1,
                )
                <= threshold
            ) / len(queries)

        fig = plt.figure()
        plt.plot(self.radius_range, self.subgraph_isomorphism)
        plt.plot(self.radius_range, self.subgraph_isomorphism_target_query)
        plt.xlabel(r"Displacement Radius / $\AA$")
        plt.ylabel("Subgraph Isomorphism Positive Rate")
        plt.legend(["Query-Target", "Target-Query"])
        plt.savefig(f"subgraph_isomorphism_{num_version}.png")

    def calculate_mean_similarities(self, num_version):
        reference = self._create_embeddings(0, 0, None)
        for j, radius in enumerate(self.radius_range):
            for i, node_masking in enumerate(self.node_masking_range):
                embeddings = self._create_embeddings(
                    node_masking=node_masking,
                    radius=radius,
                    node_to_keep_lower_bound=None,
                )
                self.self_similarity[i, j] = torch.mean(
                    cosine_similarity(reference, embeddings)
                )

        X, Y = np.meshgrid(self.node_masking_range, self.radius_range)
        Z = self.self_similarity.T

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(
            X,
            Y,
            self.self_similarity.T,
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=False,
        )
        ax.plot_wireframe(X, Y, Z, cmap=cm.coolwarm)
        ax.set_zlim(0, 1.0)
        ax.set_xlabel("Node Deletion Ratio")
        ax.set_ylabel(r"Displacement Radius / $\AA$")
        ax.set_ylim(self.max_radius, 0)
        ax.set_xlim(0, self.max_node_masking)
        ax.set_zlabel("Mean Cosine Similarity")
        plt.savefig(f"self-similarity{num_version}.png")

    def _create_embeddings(self, node_masking, radius, node_to_keep_lower_bound):
        callbacks = [
            ValidationDataTransformSetter(
                node_masking=node_masking,
                radius=radius,
                node_to_keep_lower_bound=node_to_keep_lower_bound,
            )
        ]
        trainer = Trainer(
            num_nodes=1,
            devices=self.device,
            callbacks=callbacks,
            accelerator="auto",
            logger=False,
            log_every_n_steps=1,
        )
        return torch.cat(trainer.predict(model=self.model, dataloaders=self.dataloader))


def run(device):
    PROJECT_ROOT = "/data/shared/projects/PhectorDB"
    PRETRAINING_ROOT = f"{PROJECT_ROOT}/training_data"
    VS_ROOT = f"{PROJECT_ROOT}/litpcba/ESR1_ant"
    MODEL = PhectorMatch
    VERSION = 214
    MODEL_PATH = f"{PROJECT_ROOT}/logs/{MODEL.__name__}/version_{VERSION}/"

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

    model = load_model_from_path(MODEL_PATH, MODEL)
    device = [model.device.index]

    datamodule.setup()
    eval = SelfSimilarityEvaluation(model, datamodule.val_dataloader()[1], device)
    eval.subgraph_isomorphism_evaluation(VERSION)


if __name__ == "__main__":
    device = [int(i) for i in list(sys.argv[1])]
    run(device)
