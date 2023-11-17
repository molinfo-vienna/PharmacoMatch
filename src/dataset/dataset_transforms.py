import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.data.lightning import LightningDataset
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform("distance_ohe")
class DistanceOHE(BaseTransform):
    def __init__(self, num_bins=10) -> None:
        self.num_bins = num_bins

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
            features = np.zeros((data.num_edges, self.num_bins))
            features[np.arange(data.num_edges), idx] = 1
            data.edge_attr = torch.tensor(features, dtype=torch.float)

        return data


@functional_transform("distance_rdf")
class DistanceRDF(BaseTransform):
    def __init__(self, num_bins=5, max_dist=10, gamma=0.5) -> None:
        self.num_bins = num_bins
        self.max_dist = max_dist
        self.gamma = gamma

    def __call__(self, data: Data) -> Data:
        _, num_features = data.edge_attr.shape
        device = data.edge_attr.device
        if num_features == 1:
            r = data.edge_attr
            gamma = self.gamma
            mu = torch.linspace(0, self.max_dist, self.num_bins, device=device)
            rdf = torch.exp(-gamma * (r - mu) ** 2)
            data.edge_attr = rdf# torch.tensor(rdf, dtype=torch.float)

        return data


@functional_transform("random_gaussian_noise")
class RandomGaussianNoise(BaseTransform):
    def __init__(self, std=0.28) -> None:
        self.std = std

    def __call__(self, data: Data) -> Data:
        size = data.pos.shape
        device = data.pos.device
        random_noise = torch.normal(mean=0, std=self.std, size=size, device=device)
        data.pos += random_noise

        return data


@functional_transform("random_masking")
class RandomMasking(BaseTransform):
    def __init__(self, mask_ratio=0.2) -> None:
        self.mask_ratio = mask_ratio

    def __call__(self, data: Data) -> Data:
        size = data.x.shape
        device = data.x.device
        mask = torch.rand(size, device=device) < self.mask_ratio
        data.x[mask] = 0
        return data


@functional_transform("random_node_deletion")
class RandomNodeDeletion(BaseTransform):
    def __init__(self, delete_ratio=0.3) -> None:
        self.delete_ratio = delete_ratio

    def __call__(self, data: Data) -> Data:
        device = data.x.device
        n_nodes_per_graph = data.ptr[1:] - data.ptr[:-1]
        n_nodes_to_delete = (n_nodes_per_graph * 0.3).int()
        n_nodes_to_keep = n_nodes_per_graph - n_nodes_to_delete

        idx = torch.cat([(torch.randperm(i + j, device=device) + k)[:i] for i, j, k in zip(n_nodes_to_keep, n_nodes_to_delete, data.ptr[0:-1])])

        data.x = data.x[idx]
        data.pos = data.pos[idx]
        data.batch = data.batch[idx]
        data.ptr = torch.cat((torch.zeros(1, device=device, dtype=torch.long), torch.cumsum(n_nodes_to_keep, dim=0)))

        return data

        # lst = data.to_data_list()
        # device = data.x.device
        # for data in lst:
        #     n_nodes, _ = data.size()
        #     idx = torch.rand(n_nodes, device=device) > self.delete_ratio
        #     if torch.sum(idx) >= 3:
        #         data.x = data.x[idx]
        #         data.pos = data.pos[idx]

        # return Batch.from_data_list(lst)