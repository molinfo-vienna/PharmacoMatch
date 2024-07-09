import math
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.nn import global_mean_pool
from torch.nn.functional import normalize
from torch_geometric import transforms as T


@functional_transform("distance_rdf")
class DistanceRDF(BaseTransform):
    """Transform to calculate RDF distance encodings from pair-wise edge distances.

    Pair-wise distances are represented by radial distance functions (RDF):

        e_k(r) = exp(-gamma * (r - mu_k)^2)

    where r is the pair-wise distance, mu_k is the k-th bin center, and gamma is a
    smoothing factor.

    Args:
        num_bins (int, optional): Number of bins with uniform spacing between bin
            centers. Defaults to 5.
        max_dist (float, optional): Maximum distance of the uniform grid of distance
            bins. Defaults to 10 Angstrom.
        gamma (float, optional): Smoothing factor, determines the width of the Gaussian
            shape. Defaults to 0.5.
    """

    def __init__(
        self, num_bins: int = 5, max_dist: float = 10, gamma: float = 0.5
    ) -> None:
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
            data.edge_attr = rdf

        return data


@functional_transform("random_spherical_noise")
class RandomSphericalNoise(BaseTransform):
    """Transform to add random noise sampled from a sphere to the node positions.

    Random displacements are sampled uniformly at random from a sphere with radius r.

    Args:
        radius (float, optional): radius of the sphere. Defaults to 1.
    """

    def __init__(self, radius: float = 1) -> None:
        self.radius = radius

    def __call__(self, data: Data) -> Data:
        if self.radius is None or self.radius == 0:
            return data

        data.pos += self.make_spherical_noise(data.pos)

        return data

    def make_spherical_noise(self, pos: Tensor) -> Tensor:
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


@functional_transform("random_spherical_surface_noise")
class RandomSphericalSurfaceNoise(BaseTransform):
    """Transform to add random noise sampled from a sphere surface to node positions.

    Random displacements are sampled uniformly at random from the surface of a sphere
    with radius r.

    Args:
        radius (float, optional): radius of the sphere. Defaults to 1.
    """

    def __init__(self, radius: float = 1) -> None:
        self.radius = radius

    def __call__(self, data: Data) -> Data:
        if self.radius is None or self.radius == 0:
            return data

        data.pos += self.make_sphere_surface_noise(data.pos)

        return data

    def make_sphere_surface_noise(self, pos: Tensor) -> Tensor:
        size, _ = pos.shape
        device = pos.device

        # Generate all random values in one call
        rand_vals = torch.rand((size, 3), device=device)
        phi = rand_vals[:, 0] * 6.28
        costheta = rand_vals[:, 1] * 2 - 1
        theta = torch.arccos(costheta)

        # Calculate sin and cos once to avoid redundant computations
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)

        x = self.radius * sin_theta * torch.cos(phi)
        y = self.radius * sin_theta * torch.sin(phi)
        z = self.radius * cos_theta

        return torch.vstack((x, y, z)).T


@functional_transform("furthest_spherical_surface_displacement")
class FurthestSphericalSurfaceDisplacement(BaseTransform):
    """Transform to displace features into the opposite direction of the center of mass.

    Displacements of features on the surface of a sphere with radius r s.t. they are
    displaced in the opposite direction of the center of mass. This shall prevent the
    creation of negative samples that might actually align to the initial pharmacophore.

    Args:
        radius (float, optional): radius of the sphere. Defaults to 1.
    """

    def __init__(self, radius: float = 1) -> None:
        self.radius = radius

    def __call__(self, data: Data) -> Data:
        if self.radius is None or self.radius == 0:
            return data

        data.pos += self.calculate_displacement(data.pos, data.batch)

        return data

    def calculate_displacement(self, pos: Tensor, batch: Tensor) -> Tensor:
        size, _ = pos.shape
        device = pos.device

        positional_mean = global_mean_pool(pos, batch)  # center of mass
        displacement = pos - positional_mean[batch]
        displacement = normalize(displacement, dim=1) * self.radius

        return displacement


@functional_transform("random_gaussian_noise")
class RandomGaussianNoise(BaseTransform):
    """Transform to add random Gaussian noise to the node positions.

    Will be removed in the future.

    Args:
        radius (float, optional): Width of the Gaussian, radius corresponds to 3*sigma.
            Defaults to 1.5.
    """

    def __init__(self, radius: float = 1.5) -> None:
        self.radius = radius

    def __call__(self, data: Data) -> Data:
        if self.radius is None or self.radius == 0:
            return data

        std = math.sqrt(self.radius**2 / 27)
        size = data.pos.shape
        device = data.pos.device
        random_noise = torch.normal(mean=0, std=std, size=size, device=device)
        data.pos += random_noise

        return data


@functional_transform("random_node_deletion")
class RandomNodeDeletion(BaseTransform):
    """Transform that deletes a random subset of nodes from the input data.

    Random node deletion involved removing at least one node, with the upper bound
    determined by the cardinality of the set of nodes V_i of graph G_i. The number of
    nodes to delete was drawn uniformly at random.

    Args:
        node_to_keep_lower_bound (int, optional): Minimum number of nodes that shall
            remain after node deletion. Defaults to 3.
    """

    def __init__(self, node_to_keep_lower_bound: int = 3) -> None:
        self.node_to_keep_lower_bound = node_to_keep_lower_bound

    def __call__(self, data: Data) -> Data:
        if len(data) == 0 or data is None:
            return data

        device = data.x.device
        n_nodes_per_graph = data.num_ph4_features
        n_nodes_to_delete = n_nodes_per_graph - self.node_to_keep_lower_bound
        uniform_distribution = (
            torch.rand(n_nodes_per_graph.shape, device=device) * 0.999
        )
        n_nodes_to_delete = (n_nodes_to_delete * uniform_distribution).int() + 1
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
        data.num_ph4_features = n_nodes_to_keep

        return data


@functional_transform("random_node_deletion_by_ratio")
class RandomNodeDeletionByRatio(BaseTransform):
    """Transform that deletes a random subset of nodes from the input data.

    Args:
        delete_ratio (float, optional): The number of nodes to delete is determined by
            this ratio. Defaults to 0.3.
    """

    def __init__(self, delete_ratio: float = 0.3) -> None:
        self.delete_ratio = delete_ratio

    def __call__(self, data: Data) -> Data:
        if self.delete_ratio is None or self.delete_ratio == 0:
            return data

        device = data.x.device
        n_nodes_per_graph = data.num_ph4_features
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
        data.num_ph4_features = n_nodes_to_keep

        return data


class PositionsToGraphTransform(torch.nn.Module):
    def __init__(
        self,
        num_edge_features: int = 5,
    ) -> None:
        super(PositionsToGraphTransform, self).__init__()
        self.num_edge_features = num_edge_features
        self.knn = 50

        self.transform = T.Compose(
            [
                T.KNNGraph(k=self.knn, force_undirected=True),
                T.Distance(norm=False),
                DistanceRDF(num_bins=self.num_edge_features),
            ]
        )

    @torch.no_grad()
    def forward(self, data: Data) -> Data:
        return self.transform(data.clone())
