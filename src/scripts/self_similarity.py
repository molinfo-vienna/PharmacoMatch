import sys, yaml

from lightning import Trainer, seed_everything
import numpy as np
from torch.nn.functional import cosine_similarity
import torch
import torch_geometric
from matplotlib import cm
import matplotlib.pyplot as plt

from dataset import *
from model import *
from utils import load_model_from_path


class SelfSimilarityEvaluation:
    def __init__(self, model, dataloader, device) -> None:
        self.model = model
        self.dataloader = dataloader
        self.device = device

        self.max_node_masking = 0.8
        steps_node_masking = 9
        self.max_radius = 10
        steps_radius = 11

        self.node_masking_range = [
            float(i)
            for i in torch.linspace(0, self.max_node_masking, steps_node_masking)
        ]
        self.radius_range = [
            float(i) for i in torch.linspace(0, self.max_radius, steps_radius)
        ]
        self.self_similarity = np.zeros((steps_node_masking, steps_radius))

    def calculate_mean_similarities(self, num_version):
        reference = self._create_embeddings(0, 0)
        for j, radius in enumerate(self.radius_range):
            for i, node_masking in enumerate(self.node_masking_range):
                embeddings = self._create_embeddings(
                    node_masking=node_masking, radius=radius
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
        ax.set_xlabel("Node Masking Ratio")
        ax.set_ylabel("Noise radius / Angstrom")
        ax.set_ylim(self.max_radius, 0)
        ax.set_xlim(0, self.max_node_masking)
        ax.set_zlabel("Batch-wise Mean Cosine Similarity")
        plt.savefig(f"self-similarity{num_version}.png")

    def _create_embeddings(self, node_masking, radius):
        callbacks = [
            ValidationDataTransformSetter(node_masking=node_masking, radius=radius)
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
    PRETRAINING_ROOT = "/data/shared/projects/PhectorDB/chembl_data"
    VS_ROOT = "/data/shared/projects/PhectorDB/virtual_screening_cdk2"
    CONFIG_FILE_PATH = "/home/drose/git/PhectorDB/src/scripts/config.yaml"
    MODEL = PharmCLR
    VS_MODEL_NUMBER = 49
    MODEL_PATH = f"logs/PharmCLR/version_{VS_MODEL_NUMBER}/"

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

    eval = SelfSimilarityEvaluation(model, datamodule.create_val_dataloader(), device)
    eval.calculate_mean_similarities(VS_MODEL_NUMBER)


if __name__ == "__main__":
    device = [int(i) for i in list(sys.argv[1])]
    run(device)
