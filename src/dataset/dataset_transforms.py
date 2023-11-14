import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform("distance_ohe")
class DistanceOHE(BaseTransform):
    def __call__(self, data: Data) -> Data:
        _, num_features = data.edge_attr.shape
        if num_features == 1:
            idx = (
                torch.round(torch.where(data.edge_attr * 2 < 9, data.edge_attr * 2, 9))
                .cpu()
                .numpy()
                .astype(int)
                .flatten()
            )
            features = np.zeros((data.num_edges, 10))
            features[np.arange(data.num_edges), idx] = 1
            data.edge_attr = torch.tensor(features, dtype=torch.float)  # .cuda()

        return data


@functional_transform("distance_rdf")
class DistanceRDF(BaseTransform):
    def __call__(self, data: Data) -> Data:
        _, num_features = data.edge_attr.shape
        if num_features == 1:
            r = data.edge_attr
            gamma = 0.5
            mu = torch.linspace(0, 10, 5).cuda()
            rdf = torch.exp(-gamma * (r - mu) ** 2)
            data.edge_attr = torch.tensor(rdf, dtype=torch.float)

        return data


@functional_transform("random_gaussian_noise")
class RandomGaussianNoise(BaseTransform):
    def __call__(self, data: Data) -> Data:
        size = data.pos.shape
        random_noise = torch.normal(mean=0, std=0.28, size=size).cuda()
        data.pos += random_noise

        return data


@functional_transform("random_masking")
class RandomMasking(BaseTransform):
    def __call__(self, data: Data) -> Data:
        size = data.x.shape
        mask = torch.rand(size).cuda() < 0.2
        data.x[mask] = 0
        return data


@functional_transform("random_node_deletion")
class RandomNodeDeletion(BaseTransform):
    def __call__(self, data: Data) -> Data:
        lst = data.to_data_list()
        for data in lst:
            n_nodes, _ = data.size()
            idx = torch.rand(n_nodes).cuda() > 0.3
            if torch.sum(idx) >= 3:
                data.x = data.x[idx]
                data.pos = data.pos[idx]

        return Batch.from_data_list(lst)