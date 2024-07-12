import os
import sys
import yaml

from lightning import LightningModule
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


class PharmacophoreMetaData:
    def __init__(self, path: str):
        inactive_path = os.path.join(path, "raw", "inactives.psd")
        active_path = os.path.join(path, "raw", "actives.psd")
        query_path = os.path.join(path, "raw", "query.pml")
        self.inactive_metadata = self.create_metadata(inactive_path)
        self.active_metadata = self.create_metadata(active_path)
        self.query_metadata = self.create_metadata(query_path)

    # Create metadata from pharmacophore file
    def create_metadata(self, path: str) -> pd.DataFrame:
        reader = getReaderByFileExt(path)
        ph4 = Pharm.BasicPharmacophore()
        names = []
        features = []
        index = []
        conf_index = []
        num_features = []
        conf = 0
        i = 0
        name = ""

        while reader.read(ph4):
            if ph4.getNumFeatures() == 0:
                continue
            feature_types = Pharm.generateFeatureTypeHistogramString(ph4)
            if name == Pharm.getName(ph4):
                conf += 1
            else:
                conf = 0
                name = Pharm.getName(ph4)
            conf_index.append(conf)
            features.append(feature_types)
            names.append(name)
            index.append(i)
            num_features.append(ph4.getNumFeatures())
            i += 1

        metadata = pd.DataFrame(
            {
                "index": index,
                "name": names,
                "conf_idx": conf_index,
                "features": features,
                "num_features": num_features,
            }
        )

        return metadata


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
