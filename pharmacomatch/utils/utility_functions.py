import os
import yaml
from typing import Callable, Tuple

from lightning import LightningModule
import numpy as np
import torch
from torch import Tensor
from rdkit.ML.Scoring.Scoring import CalcBEDROC


def bootstrap_metric(
    y_true: Tensor,
    y_pred: Tensor,
    metric: Callable,
    num_bootstrap: int = 100,
    **kwargs,
) -> Tuple[float, float]:
    """Bootstrap the metric calculation."""
    vals = []
    for _ in range(num_bootstrap):
        idx = (np.random.uniform(size=len(y_true)) * len(y_true)).astype(int)
        if "alpha" in kwargs.keys():
            vals.append(metric(y_true[idx], y_pred[idx], alpha=kwargs["alpha"]))
        else:
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


def enrichment_factor(y_true: Tensor, y_pred: Tensor, alpha: float = 0.1) -> float:
    """
    Calculate EF score.

    Parameters:
    - y_true: true binary labels (0 or 1)
    - y_score: predicted scores or probabilities
    - alpha: parameter controlling the degree of early retrieval emphasis

    Returns:
    - EF score
    """
    predicted_ranking = torch.argsort(y_pred, descending=True)
    bound = int(len(y_pred) * alpha)
    actives_above_bound = y_true[predicted_ranking][:bound]
    ef = sum(actives_above_bound) / sum(y_true) / alpha

    return ef.item()


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
