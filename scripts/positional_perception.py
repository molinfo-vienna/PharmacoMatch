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

from pharmacomatch.dataset import PharmacophoreDataModule
from pharmacomatch.model import ValidationDataTransformSetter, PharmacoMatch


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

    def subgraph_isomorphism_evaluation(self, results_path: str) -> None:
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

        fig1 = self._create2Dplot(self.mean_matching_decision)
        fig2 = self._create3Dplot(self.mean_matching_decision)

        fig1.savefig(
            os.path.join(results_path, "positional_perception.jpg"), bbox_inches="tight"
        )
        fig2.savefig(os.path.join(results_path, "Query-Target.png"), dpi=150)

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
    # Path variables
    ROOT = os.getcwd()
    PRETRAINING_ROOT = os.path.join(ROOT, "data", "training_data")
    RESULTS_LOCATION = os.path.join(ROOT, "results")
    if not os.path.exists(RESULTS_LOCATION):
        os.mkdir(RESULTS_LOCATION)
    MODEL = PharmacoMatch
    MODEL_PATH = os.path.join(ROOT, "trained_model", "trained_model.ckpt")
    params = yaml.load(
        open(os.path.join(ROOT, "trained_model", "hparams.yaml"), "r"),
        Loader=yaml.FullLoader,
    )

    # Deterministic flags
    torch.set_float32_matmul_precision("medium")
    torch_geometric.seed_everything(params["seed"])
    seed_everything(params["seed"])
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    datamodule = PharmacophoreDataModule(
        PRETRAINING_ROOT,
        batch_size=params["batch_size"],
    )

    model = MODEL.load_from_checkpoint(
        MODEL_PATH, map_location=torch.device(f"cuda:{device[0]}")
    )
    device = [model.device.index]

    datamodule.setup()
    eval = PositionalPerceptionAssessor(model, datamodule.val_dataloader()[1], device)
    eval.subgraph_isomorphism_evaluation(RESULTS_LOCATION)


if __name__ == "__main__":
    device = [int(i) for i in list(sys.argv[1])]
    run(device)
