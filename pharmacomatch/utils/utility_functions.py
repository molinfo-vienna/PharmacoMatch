import os
import yaml
from typing import Callable, Tuple

from lightning import LightningModule
import numpy as np
import torch
from torch import Tensor
from rdkit.ML.Scoring.Scoring import CalcBEDROC


def bootstrap_metric(
    y_true: Tensor, y_pred: Tensor, metric: Callable, num_bootstrap: int = 1000
) -> Tuple[float, float]:
    """Bootstrap the metric calculation."""
    vals = []
    for _ in range(num_bootstrap):
        idx = (np.random.uniform(size=len(y_true)) * len(y_true)).astype(int)
        vals.append(metric(y_true[idx], y_pred[idx]))

    return np.mean(vals), np.std(vals)


def bedroc_score(y_true: Tensor, y_pred: Tensor, alpha: float = 20) -> float:
    """
    Calculate BEDROC score with the RDKit implementation.

    Parameters:
    - y_true: true binary labels (0 or 1)
    - y_score: predicted scores or probabilities
    - alpha: parameter controlling the degree of early retrieval emphasis

    Returns:
    - BEDROC score
    """
    scores = np.expand_dims(y_pred, axis=1)
    y_true = np.expand_dims(y_true, axis=1)
    scores = np.concatenate((scores, y_true), axis=1)
    # inverse sort scores based on first column
    scores = scores[scores[:, 0].argsort()[::-1]]
    bedroc = CalcBEDROC(scores, 1, alpha)

    return bedroc


def load_hparams_from_path(folder_path: str) -> dict:
    """Load the hyperparameters from the checkpoint path."""
    if not os.path.exists(folder_path):
        return None
    else:
        path = os.path.join(folder_path, "hparams.yaml")
        return yaml.load(open(path, "r"), Loader=yaml.FullLoader)


def load_model_from_path(
    folder_path: str, model_class: LightningModule, device: int = 0
) -> LightningModule:
    """Load the model from the checkpoint path."""
    if not os.path.exists(folder_path):
        return None
    else:
        folder_path = os.path.join(folder_path, "checkpoints")

    model_path = None
    for file in os.listdir(folder_path):
        if file.endswith(".ckpt"):
            model_path = os.path.join(folder_path, file)

    if model_path:
        return model_class.load_from_checkpoint(
            model_path, map_location=torch.device(f"cuda:{device}")
        )
    else:
        return None


# def visualize_pharm(data_list):
#     # plot configurations
#     fig = plt.figure()

#     ax = fig.add_subplot(111, projection="3d")
#     ax.axes.xaxis.set_ticklabels([])
#     ax.axes.yaxis.set_ticklabels([])
#     ax.axes.zaxis.set_ticklabels([])

#     colors = [
#         "blue",
#         "orange",
#         "green",
#         "red",
#         "purple",
#         "brown",
#         "pink",
#         "gray",
#         "olive",
#         "black",
#     ]

#     def plot_data(data, edge_color):
#         pos = data.pos.cpu().numpy()
#         x = data.x.cpu().numpy()

#         if data.edge_index is not None:
#             edge_index = data.edge_index.cpu()
#             for src, dst in edge_index.t().tolist():
#                 src = pos[src].tolist()
#                 dst = pos[dst].tolist()
#                 ax.plot(
#                     [src[0], dst[0]],
#                     [src[1], dst[1]],
#                     [src[2], dst[2]],
#                     linewidth=0.3,
#                     color=edge_color,
#                 )

#         # List of feature types
#         features = [(np.argmax(x, axis=1) == 0) * (np.sum(x, axis=1) != 0)]
#         for i in range(1, 9):
#             features.append(np.argmax(x, axis=1) == i)
#         features.append(np.sum(x, axis=1) == 0)  # --> masked features

#         for feature, c in zip(features, colors):
#             ax.scatter(
#                 pos[feature, 0],
#                 pos[feature, 1],
#                 pos[feature, 2],
#                 s=50,
#                 color=c,
#             )

#     for i, data in enumerate(data_list):
#         plot_data(data, colors[i % 10])

#     plt.savefig("pharm.png")
