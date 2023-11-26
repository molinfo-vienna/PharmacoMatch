import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from torch.nn.functional import cosine_similarity
from torch_geometric.nn import global_max_pool
import sys
import os
import yaml

import torch
import torch_geometric
from matplotlib import cm
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, Callback
import numpy as np

from utils import *
from dataset import *
from model import *


class SelfSimilarityEvaluation:
    def __init__(self, model, dataloader, device) -> None:
        self.model = model
        self.dataloader = dataloader
        self.device = device

        max_node_masking = 0.8
        steps_node_masking = 9
        max_std = 2
        steps_std = 11

        self.node_masking_range = [float(i) for i in torch.linspace(
            0, max_node_masking, steps_node_masking)]
        self.std_range = [float(i)
                          for i in torch.linspace(0, max_std, steps_std)]
        self.self_similarity = np.zeros((steps_node_masking, steps_std))

    def create_embeddings(self, node_masking, std):
        callbacks = [ValidationDataTransformSetter(
            node_masking=node_masking, std=std)]
        self.trainer = Trainer(num_nodes=1,
                               devices=self.device,
                               callbacks=callbacks,
                               accelerator='auto',
                               logger=False,
                               log_every_n_steps=1)
        return torch.cat(self.trainer.predict(
            model=self.model, dataloaders=self.dataloader))

    def calculate_mean_similarities(self):
        reference = self.create_embeddings(0, 0)
        for i, node_masking in enumerate(self.node_masking_range):
            for j, std in enumerate(self.std_range):
                embeddings = self.create_embeddings(node_masking, std)
                self.self_similarity[i, j] = torch.mean(
                    cosine_similarity(reference, embeddings))
                
        X, Y = np.meshgrid(self.node_masking_range, self.std_range)
        Z = self.self_similarity.T

        # fig1 = plt.figure()
        # plt.figure()
        # CS = plt.contourf(X, Y, Z, levels=np.linspace(0,1,11))
        # plt.clabel(CS, inline=1, fontsize=10)
        # plt.title('Simplest default with labels')
        # plt.savefig('contour')

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, self.self_similarity.T, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.plot_wireframe(X, Y, Z, cmap=cm.coolwarm)
        ax.set_zlim(0, 1.)
        ax.set_xlabel('Node Masking Ratio')
        ax.set_ylabel('Gaussian Noise std / Angstrom')
        ax.set_zlabel('Batch-wise Mean Cosine Similarity')
        #fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig('self-similarity.png')


def run(device):
    PRETRAINING_ROOT = "/data/shared/projects/PhectorDB/chembl_data"
    VS_ROOT = "/data/shared/projects/PhectorDB/virtual_screening_cdk2"
    CONFIG_FILE_PATH = "/home/drose/git/PhectorDB/src/scripts/config.yaml"
    MODEL = PharmCLR
    VS_MODEL_NUMBER = 35

    params = yaml.load(open(CONFIG_FILE_PATH, "r"), Loader=yaml.FullLoader)

    torch.set_float32_matmul_precision("medium")
    torch_geometric.seed_everything(params["seed"])
    seed_everything(params["seed"])
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    datamodule = PharmacophoreDataModule(
        PRETRAINING_ROOT, VS_ROOT, batch_size=params["batch_size"], small_set_size=params["num_samples"])

    # load the trained model
    def load_model(path):
        for file in os.listdir(path):
            if file.endswith(".ckpt"):
                path = os.path.join(path, file)
        return MODEL.load_from_checkpoint(path)

    path = f'logs/PharmCLR/version_{VS_MODEL_NUMBER}/checkpoints/'
    model = load_model(path)
    datamodule.setup('fit')

    eval = SelfSimilarityEvaluation(
        model, datamodule.create_val_dataloader(), device)
    eval.calculate_mean_similarities()


if __name__ == "__main__":
    device = [int(i) for i in list(sys.argv[1])]
    run(device)


# from sklearn.manifold import TSNE
