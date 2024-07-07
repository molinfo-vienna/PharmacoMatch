import torch
from torch_geometric import transforms as T
from torch_geometric.data import Data

from .dataset_transforms import (
    DistanceRDF,
    RandomNodeDeletion,
    RandomSphericalNoise,
    RandomSphericalSurfaceNoise,
    FurthestSphericalSurfaceDisplacement,
)

# TODO: The augmentation module signature is quite unsound. I need to rethink the design.
class AugmentationModule(torch.nn.Module):
    """Module for augmenting the input data.

    The module applies a series of transformations to the input data and creates a
    complete graph from the feature positions.

    Args:
        train (bool, optional): If set to False, the module creates the complete graph
            representation without prior augmentation. Defaults to True.
        node_masking (float, optional): Percentage of nodes to be deleted.
            Defaults to 0.3.
        radius (float, optional): Features are displaced by adding uniform random noise
            from a sphere of the given radius. Defaults to 0.75.
        sphere_surface_sampling (bool, optional): If set to True, uniform random noise
            gets sampled from a sphere surface. Otherwise, noise gets sampled from
            within the sphere. Defaults to False.
        num_edge_features (int, optional): Number of bins of the created edge attribute
            vectors. Defaults to 5.
        node_to_keep_lower_bound (int, optional): Minimum number of nodes that shall
            remain after random node deletion. Defaults to 3.
    """

    def __init__(
        self,
        train: bool = True,
        node_masking: float = 0.3,
        radius: float = 0.75,
        sphere_surface_sampling: bool = False,
        num_edge_features: int = 5,
        node_to_keep_lower_bound: int = 3,
    ) -> None:
        super(AugmentationModule, self).__init__()
        self.is_training = train
        self.node_masking = node_masking
        self.radius = radius
        self.num_edge_features = num_edge_features
        self.node_to_keep_lower_bound = node_to_keep_lower_bound
        self.knn = 50

        if sphere_surface_sampling:
            node_displacement = FurthestSphericalSurfaceDisplacement(self.radius)
        else:
            node_displacement = RandomSphericalNoise(self.radius)

        self.transform = T.Compose(
            [
                RandomNodeDeletion(self.node_to_keep_lower_bound, node_masking),
                node_displacement,
                T.KNNGraph(k=self.knn, force_undirected=True),
                T.Distance(norm=False),
                DistanceRDF(num_bins=self.num_edge_features),
            ]
        )
        self.val_transform = T.Compose(
            [
                T.KNNGraph(k=self.knn, force_undirected=True),
                T.Distance(norm=False),
                DistanceRDF(num_bins=self.num_edge_features),
            ]
        )

    @torch.no_grad()
    def forward(self, data: Data) -> Data:
        if self.is_training:
            return self.transform(data.clone())
        else:
            return self.val_transform(data.clone())
