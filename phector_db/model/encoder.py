import torch
from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GINEConv, MLP
from torch_geometric.nn import global_mean_pool, BatchNorm, global_add_pool
from torch_geometric.nn import (
    MLP,
    PointTransformerConv,
    fps,
    knn,
    knn_graph,
)
from torch_geometric.utils import scatter


class PositiveLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.log_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.log_weight)

    def forward(self, input):
        return torch.nn.functional.linear(input, self.log_weight.exp())


class GATEncoder(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        node_embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_conv_layers: int,
        num_edge_features: int,
        dropout: float,
        residual_connection: str,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_conv_layers = n_conv_layers
        self.num_edge_features = num_edge_features
        self.dropout = dropout
        self.residual_connection = residual_connection
        self.num_heads = 10

        # node_embedding_dim = hidden_dim
        self.node_embedding = Linear(input_dim, node_embedding_dim)
        input_dim = node_embedding_dim

        # Convolutional layers
        # try out spectral layers, maybe 3-6 layers
        self.convolution = torch.nn.ModuleList()
        self.convolution_batch_norm = torch.nn.ModuleList()

        # input and outputdimension maybe bigger
        # input to say 32-64, output to maybe 25
        for _ in range(self.n_conv_layers):
            self.convolution.append(
                GATConv(  # GAT is maybe not most suitable for our problem
                    input_dim,
                    hidden_dim,
                    edge_dim=self.num_edge_features,
                    heads=self.num_heads,
                    concat=False,
                )
                # GINEConv(
                #     MLP([input_dim, hidden_dim]),
                #     edge_dim=self.num_edge_features,
                #     train_eps=False,
                # )
            )
            self.convolution_batch_norm.append(BatchNorm(hidden_dim))
            if self.residual_connection == "dense":
                input_dim += hidden_dim
            if self.residual_connection == "res":
                input_dim = hidden_dim

        # Expand dimensionality before node pooling
        self.linear = Linear(input_dim, output_dim)
        self.pooling = global_mean_pool

    def forward(self, data: Data) -> Tensor:
        # Embedding of OHE features
        x = data.x
        x = self.node_embedding(x)

        # Graph convolution via message passing
        for i, conv in enumerate(self.convolution):
            x_conv = conv(x, data.edge_index, data.edge_attr)
            x_conv = self.convolution_batch_norm[i](x_conv)
            x_conv = torch.nn.functional.gelu(x_conv)
            if self.residual_connection == "dense":
                x = torch.cat((x, x_conv), dim=1)
            if self.residual_connection == "res":
                x = x + x_conv
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Dimensionality expansion and read-out
        x = self.linear(x)
        representation = self.pooling(x, data.batch)

        return representation


class TransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_in = Linear(in_channels, in_channels)
        self.lin_out = Linear(out_channels, out_channels)

        self.pos_nn = MLP([3, 64, out_channels], norm=None, plain_last=False)

        self.attn_nn = MLP(
            [out_channels, 64, out_channels], norm=None, plain_last=False
        )

        self.transformer = PointTransformerConv(
            in_channels, out_channels, pos_nn=self.pos_nn, attn_nn=self.attn_nn
        )

    def forward(self, x, pos, edge_index):
        x = self.lin_in(x).relu()
        x = self.transformer(x, pos, edge_index)
        x = self.lin_out(x).relu()
        return x


class TransitionDown(torch.nn.Module):
    """Samples the input point cloud by a ratio percentage to reduce
    cardinality and uses an mlp to augment features dimensionnality.
    """

    def __init__(self, in_channels, out_channels, ratio=0.5, k=3):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = MLP([in_channels, out_channels], plain_last=False)

    def forward(self, x, pos, batch):
        # FPS sampling
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)

        # compute for each cluster the k nearest points
        sub_batch = batch[id_clusters] if batch is not None else None

        # beware of self loop
        id_k_neighbor = knn(
            pos, pos[id_clusters], k=self.k, batch_x=batch, batch_y=sub_batch
        )

        # transformation of features through a simple MLP
        x = self.mlp(x)

        # Max pool onto each cluster the features from knn in points
        x_out = scatter(
            x[id_k_neighbor[1]],
            id_k_neighbor[0],
            dim=0,
            dim_size=id_clusters.size(0),
            reduce="max",
        )

        # keep only the clusters and their max-pooled features
        sub_pos, out = pos[id_clusters], x_out
        return out, sub_pos, sub_batch


class PointTransformerEncoder(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_conv_layers: int,
        dropout: float,
        k: int = 3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_conv_layers = n_conv_layers
        self.dropout = dropout
        self.k = k

        # first block
        self.mlp_input = MLP([self.input_dim, self.hidden_dim], plain_last=False)

        self.transformer_input = TransformerBlock(
            in_channels=hidden_dim, out_channels=hidden_dim
        )

        # backbone layers
        self.transformers_down = torch.nn.ModuleList()
        self.transition_down = torch.nn.ModuleList()

        for _ in range(n_conv_layers - 1):
            # Add Transition Down block followed by a Transformer block
            self.transition_down.append(
                TransitionDown(
                    in_channels=hidden_dim, out_channels=hidden_dim * 2, k=self.k
                )
            )

            hidden_dim *= 2

            self.transformers_down.append(
                TransformerBlock(in_channels=hidden_dim, out_channels=hidden_dim)
            )

        self.mlp_output = MLP([hidden_dim, output_dim], norm=None)

    def forward(self, data):
        x = data.x
        pos = data.pos
        batch = data.batch

        # first block
        x = self.mlp_input(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_input(x, pos, edge_index)

        # backbone
        for i in range(len(self.transformers_down)):
            x, pos, batch = self.transition_down[i](x, pos, batch=batch)

            edge_index = knn_graph(pos, k=self.k, batch=batch)
            x = self.transformers_down[i](x, pos, edge_index)

        # GlobalAveragePooling
        x = global_mean_pool(x, batch)

        return self.mlp_output(x)
