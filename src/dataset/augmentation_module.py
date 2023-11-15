from torch_geometric import transforms as T

from .dataset_transforms import DistanceOHE, DistanceRDF, RandomMasking, RandomGaussianNoise

class AugmentationModule:
    def __init__(self, train=True) -> None:
        self.train = train

        # Data Augmentation
        #if self.train:
        self.transform = T.Compose(
            [
                RandomMasking(), # with mask token, or better deletion? Try both.
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

    def __call__(self, data):
        if self.train:
            return self.transform(data.clone()), self.transform(data.clone())
        else: 
            return self.transform(data.clone()), self.transform(data.clone()), self.val_transform(data.clone())