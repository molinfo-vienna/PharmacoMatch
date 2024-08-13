import os
import sys
import yaml

from lightning import Trainer, seed_everything
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric
from torch_geometric.loader import DataLoader

from dataset import PharmacophoreDataModule
from model import ValidationDataTransformSetter, PharmacoMatch
from utils import load_model_from_path


class PositionalPerceptionAssessor:
    """Assessment of the positional perception of the model.

    This class implements the experiment for positional perception as described in the
    paper.

    Args:
        model (PharmacoMatch): The trained model.
        dataloader (DataLoader): Dataloader to use for the evaluation. We used the outer
            validation set in this experiment.
        device (list[int]): Index of the device to use for the evaluation.
        max_threshold(int): Upper bound on the threshold for the decision function.
    """

    def __init__(
        self,
        model: PharmacoMatch,
        dataloader: DataLoader,
        device: list[int],
        max_threshold: int = 20000,
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.max_threshold = max_threshold
        self.max_radius = 10
        steps_radius = 41
        steps_threshold = 50

        self.radius_range = [
            float(i) for i in torch.linspace(0, self.max_radius, steps_radius)
        ]
        self.threshold_range = [
            float(i) for i in torch.linspace(self.max_threshold, 0, steps_threshold)
        ]
        self.mean_matching_decision = np.zeros((steps_threshold, steps_radius))
        self.mean_matching_decision_target_query = np.zeros(
            (steps_threshold, steps_radius)
        )

    def subgraph_isomorphism_evaluation(self, num_version):
        """Evaluation of the matching decision function with different displacement
        radii and thresholds, and plotting of the results."""
        target = self._create_embeddings(0, 0, None)
        for j, radius in enumerate(self.radius_range):
            queries = self._create_embeddings(
                node_masking=1, radius=radius, node_to_keep_lower_bound=3
            )
            for i, threshold in enumerate(self.threshold_range):
                self.mean_matching_decision[i, j] = torch.sum(
                    self.model.penalty(queries, target) <= threshold
                ) / len(queries)

                self.mean_matching_decision_target_query[i, j] = torch.sum(
                    self.model.penalty(target, queries) <= threshold
                ) / len(queries)

        fig1 = self._create2Dplot(self.mean_matching_decision)
        fig2 = self._create3Dplot(self.mean_matching_decision)
        fig3 = self._create3Dplot(self.mean_matching_decision_target_query)

        fig1.savefig(f"{num_version}_positional_perception.jpg", bbox_inches="tight")
        fig2.savefig(f"{num_version}_Query-Target.png")
        fig3.savefig(f"{num_version}_Target_Query.png")

    def _create_embeddings(self, node_masking, radius, node_to_keep_lower_bound):
        """Create embeddings for the given parameters."""
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

    def _create3Dplot(self, values):
        """Plot decision function values against displacement radius and threshold."""
        X, Y = np.meshgrid(self.threshold_range, self.radius_range)
        Z = values.T

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(
            X,
            Y,
            Z,
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=False,
        )
        ax.plot_wireframe(X, Y, Z, cmap=cm.coolwarm)
        ax.set_zlim(0, 1.0)
        ax.set_xlabel("Threshold")
        ax.set_ylabel(r"Displacement Radius / $\AA$")
        ax.set_ylim(self.max_radius, 0)
        ax.set_xlim(self.max_threshold, 0)
        ax.set_zlabel("Subgraph Positive Rate")

        return fig

    def _create2Dplot(self, values):
        """Plot decision function values against displacement radius for a given
        threshold."""
        fig = plt.figure(figsize=(3, 5))
        plt.plot(self.radius_range, values[33])
        plt.xlabel(r"Displacement radius $r_D$ / $\AA$", fontsize=12)
        plt.ylabel("Mean matching decision function", fontsize=12)
        plt.xlim([0, 10])
        plt.ylim([0, 1])
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        return fig


def run(device):
    PROJECT_ROOT = "/data/shared/projects/PhectorDB"
    PRETRAINING_ROOT = f"{PROJECT_ROOT}/training_data"
    MODEL = PharmacoMatch
    VERSION = 328
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
        batch_size=params["batch_size"],
    )

    model = load_model_from_path(MODEL_PATH, MODEL)
    device = [model.device.index]

    datamodule.setup()
    eval = PositionalPerceptionAssessor(model, datamodule.val_dataloader()[1], device)
    eval.subgraph_isomorphism_evaluation(VERSION)


if __name__ == "__main__":
    device = [int(i) for i in list(sys.argv[1])]
    run(device)
