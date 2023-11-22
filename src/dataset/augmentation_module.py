import torch
from torch_geometric import transforms as T
from lightning import LightningModule

from .dataset_transforms import DistanceOHE, DistanceRDF, RandomMasking, RandomGaussianNoise, RandomNodeDeletion, CompleteGraph


class AugmentationModule(torch.nn.Module):
    def __init__(self, train=True) -> None:
        super(AugmentationModule, self).__init__()
        self.is_training = train
        self.node_masking = 0.5
        self.knn = 50
        self.num_edge_features = 5

        # Data Augmentation
        # if self.train:
        self.transform = T.Compose(
            [
                # RandomMasking(), # with mask token, or better deletion? Try both.
                RandomNodeDeletion(self.node_masking),
                # Random masking mit bis zu 70%
                # RandomGaussianNoise(), # two different tolerance radii
                T.KNNGraph(k=self.knn, force_undirected=True),
                T.Distance(norm=False),
                DistanceRDF(num_bins=self.num_edge_features),
            ]
        )
    # else:
        self.val_transform = T.Compose(
            [
                T.KNNGraph(k=self.knn, force_undirected=True),
                T.Distance(norm=False),
                DistanceRDF(num_bins=self.num_edge_features),
            ]
        )

    @torch.no_grad()
    def forward(self, data):
        if self.is_training:
            return self.transform(data.clone())
        else:
            return self.val_transform(data.clone())
