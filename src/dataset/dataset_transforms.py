import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_undirected


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
            data.edge_attr = rdf  # torch.tensor(rdf, dtype=torch.float)

        return data


@functional_transform("random_gaussian_noise")
class RandomGaussianNoise(BaseTransform):
    def __init__(self, radius=1.5) -> None:
        self.radius = radius

    def __call__(self, data: Data) -> Data:
        if self.radius == None or self.radius == 0:
            return data

        std = torch.sqrt(self.radius / 27)

        size = data.pos.shape
        device = data.pos.device
        random_noise = torch.normal(mean=0, std=std, size=size, device=device)
        data.pos += random_noise

        return data


@functional_transform("random_spherical_noise")
class RandomSphericalNoise(BaseTransform):
    def __init__(self, radius=1) -> None:
        self.radius = radius

    def __call__(self, data: Data) -> Data:
        if self.radius == None or self.radius == 0:
            return data

        data.pos += self.make_spherical_noise(data.pos)

        return data

    def make_spherical_noise(self, pos):
        size, _ = pos.shape
        device = pos.device

        # Generate all random values in one call
        rand_vals = torch.rand((size, 3), device=device)

        phi = rand_vals[:, 0] * 6.28
        costheta = rand_vals[:, 1] * 2 - 1
        u = rand_vals[:, 2]

        theta = torch.arccos(costheta)

        # Calculate sin and cos once to avoid redundant computations
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)

        r = self.radius * torch.pow(u, 1 / 3)
        x = r * sin_theta * torch.cos(phi)
        y = r * sin_theta * torch.sin(phi)
        z = r * cos_theta

        return torch.vstack((x, y, z)).T


@functional_transform("random_masking")
class RandomMasking(BaseTransform):
    def __init__(self, mask_ratio=0.2) -> None:
        self.mask_ratio = mask_ratio

    def __call__(self, data: Data) -> Data:
        if self.mask_ratio == None or self.mask_ratio == 0:
            return Data

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
        if self.delete_ratio == None or self.delete_ratio == 0:
            return data

        device = data.x.device
        n_nodes_per_graph = data.ptr[1:] - data.ptr[:-1]
        n_nodes_to_delete = (n_nodes_per_graph * self.delete_ratio).int()
        n_nodes_to_keep = n_nodes_per_graph - n_nodes_to_delete

        idx = torch.cat(
            [
                (torch.randperm(i + j, device=device) + k)[:i]
                for i, j, k in zip(n_nodes_to_keep, n_nodes_to_delete, data.ptr[0:-1])
            ]
        )

        data.x = data.x[idx]
        data.pos = data.pos[idx]
        data.batch = data.batch[idx]
        data.ptr = torch.cat(
            (
                torch.zeros(1, device=device, dtype=torch.long),
                torch.cumsum(n_nodes_to_keep, dim=0),
            )
        )

        return data


@functional_transform("complete_graph")
class CompleteGraph(BaseTransform):
    def __call__(self, data: Data) -> Data:
        n_nodes_per_graph = data.ptr[1:] - data.ptr[:-1]
        device = data.ptr.device

        def create_edges(num_nodes, ptr):
            idx = torch.combinations(torch.arange(num_nodes, device=device), r=2)
            edge_index = to_undirected(idx.t(), num_nodes=num_nodes)
            return edge_index + ptr

        data.edge_index = torch.cat(
            (
                [
                    create_edges(num_nodes, ptr)
                    for num_nodes, ptr in zip(n_nodes_per_graph, data.ptr[:-1])
                ]
            ),
            dim=1,
        )

        return data
