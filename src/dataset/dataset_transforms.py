import numpy as np
import torch
from torch_geometric.data import Data
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
            mu = np.linspace(0, 10, 5)
            rdf = np.exp(-gamma * (r - mu) ** 2)
            data.edge_attr = torch.tensor(rdf, dtype=torch.float)

        return data
