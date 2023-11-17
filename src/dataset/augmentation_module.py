import torch
from torch_geometric import transforms as T
from lightning import LightningModule

from .dataset_transforms import DistanceOHE, DistanceRDF, RandomMasking, RandomGaussianNoise, RandomNodeDeletion

class AugmentationModule(LightningModule):
    def __init__(self, train=True) -> None:
        super(AugmentationModule, self).__init__()
        self.is_training = train

        # Data Augmentation
        #if self.train:
        self.transform = T.Compose(
            [
                #RandomMasking(), # with mask token, or better deletion? Try both.
                RandomNodeDeletion(0.5),
                # Random masking mit bis zu 70%
                RandomGaussianNoise(), # two different tolerance radii
                T.KNNGraph(k=50, force_undirected=True),
                T.Distance(norm=False),
                DistanceRDF(num_bins=5),
            ]
        )
    #else: 
        self.val_transform = T.Compose(
            [
                T.KNNGraph(k=50, force_undirected=True),
                T.Distance(norm=False),
                DistanceRDF(num_bins=5),
            ]
        )

    @torch.no_grad()
    def forward(self, data):
        if self.is_training:
            return self.transform(data.clone()), self.transform(data.clone())
        else: 
            return self.val_transform(data.clone())