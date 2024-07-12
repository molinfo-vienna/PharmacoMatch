import os
import sys
import yaml

from lightning import LightningModule
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

import CDPL.Pharm as Pharm


def enrichment_factor(y_true: Tensor, y_pred: Tensor, alpha: float) -> float:
    sorted, indices = torch.sort(y_pred, descending=True)
    high_ranked_actives = sum(y_true[indices][: int(len(y_true) * alpha)])
    actives = sum(y_true)
    return high_ranked_actives / (actives * alpha)


# Retrieve meta data from screening data base
def getReaderByFileExt(filename: str) -> Pharm.PharmacophoreReader:
    name_and_ext = os.path.splitext(filename)

    if name_and_ext[1] == "":
        sys.exit(
            "Error: could not determine pharmacophore input file format (file extension missing)"
        )

    # get input handler for the format specified by the input file's extension
    ipt_handler = Pharm.PharmacophoreIOManager.getInputHandlerByFileExtension(
        name_and_ext[1][1:].lower()
    )

    if not ipt_handler:
        sys.exit(
            "Error: unsupported pharmacophore input file format '%s'" % name_and_ext[1]
        )

    # create and return file reader instance
    return ipt_handler.createReader(filename)


def load_hparams_from_path(folder_path: str) -> dict:
    if not os.path.exists(folder_path):
        return None
    else:
        path = os.path.join(folder_path, "hparams.yaml")
        return yaml.load(open(path, "r"), Loader=yaml.FullLoader)


def load_model_from_path(
    folder_path: str, model_class: LightningModule, device: int = 0
) -> LightningModule:
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


def visualize_pharm(data_list):
    # plot configurations
    fig = plt.figure()

    ax = fig.add_subplot(111, projection="3d")
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])

    colors = [
        "blue",
        "orange",
        "green",
        "red",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "black",
    ]

    def plot_data(data, edge_color):
        pos = data.pos.cpu().numpy()
        x = data.x.cpu().numpy()

        if data.edge_index is not None:
            edge_index = data.edge_index.cpu()
            for src, dst in edge_index.t().tolist():
                src = pos[src].tolist()
                dst = pos[dst].tolist()
                ax.plot(
                    [src[0], dst[0]],
                    [src[1], dst[1]],
                    [src[2], dst[2]],
                    linewidth=0.3,
                    color=edge_color,
                )

        # List of feature types
        features = [(np.argmax(x, axis=1) == 0) * (np.sum(x, axis=1) != 0)]
        for i in range(1, 9):
            features.append(np.argmax(x, axis=1) == i)
        features.append(np.sum(x, axis=1) == 0)  # --> masked features

        for feature, c in zip(features, colors):
            ax.scatter(
                pos[feature, 0],
                pos[feature, 1],
                pos[feature, 2],
                s=50,
                color=c,
            )

    for i, data in enumerate(data_list):
        plot_data(data, colors[i % 10])

    plt.savefig("pharm.png")
