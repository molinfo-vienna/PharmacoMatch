import torch
from torch_geometric import transforms as T
from torch_geometric.data import Data

from .dataset_transforms import (
    DistanceRDF,
    RandomNodeDeletion,
    RandomSphericalNoise,
    RandomSphericalSurfaceNoise,
)


class AugmentationModule(torch.nn.Module):
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
            node_displacement = RandomSphericalSurfaceNoise(self.radius)
        else:
            node_displacement = RandomSphericalNoise(self.radius)

        self.transform = T.Compose(
            [
                RandomNodeDeletion(self.node_masking, self.node_to_keep_lower_bound),
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
