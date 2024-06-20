import torch
from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, MLP, GATv2Conv, NNConv
from torch_geometric.nn import global_mean_pool, BatchNorm, global_add_pool
from torch_geometric.nn import MLP


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
        pooling: str = "mean",
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

        self.node_embedding = Linear(input_dim, node_embedding_dim)
        input_dim = node_embedding_dim

        self.convolution = torch.nn.ModuleList()
        self.convolution_batch_norm = torch.nn.ModuleList()

        for _ in range(self.n_conv_layers):
            self.convolution.append(
                GATv2Conv(
                    input_dim,
                    hidden_dim,
                    edge_dim=self.num_edge_features,
                    heads=self.num_heads,
                    concat=False,
                )
            )
            self.convolution_batch_norm.append(BatchNorm(hidden_dim))
            if self.residual_connection == "dense":
                input_dim += hidden_dim
            if self.residual_connection == "res":
                input_dim = hidden_dim

        self.linear = Linear(input_dim, output_dim)

        if pooling == "mean":
            self.pooling = global_mean_pool
        elif pooling == "add":
            self.pooling = global_add_pool

    def forward(self, data: Data) -> Tensor:
        x = data.x
        x = self.node_embedding(x)

        for i, conv in enumerate(self.convolution):
            x_conv = conv(x, data.edge_index, data.edge_attr)
            x_conv = self.convolution_batch_norm[i](x_conv)
            x_conv = torch.nn.functional.gelu(x_conv)
            if self.residual_connection == "dense":
                x = torch.cat((x, x_conv), dim=1)
            if self.residual_connection == "res":
                x = x + x_conv
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.linear(x)
        representation = self.pooling(x, data.batch)

        return representation


class GINEncoder(torch.nn.Module):
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
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_conv_layers = n_conv_layers
        self.num_edge_features = num_edge_features
        self.dropout = dropout
        self.residual_connection = residual_connection

        node_embedding_dim = hidden_dim
        self.node_embedding = Linear(input_dim, node_embedding_dim)
        input_dim = node_embedding_dim

        self.convolution = torch.nn.ModuleList()
        self.convolution_batch_norm = torch.nn.ModuleList()

        for _ in range(self.n_conv_layers):
            self.convolution.append(
                GINEConv(
                    MLP([input_dim, hidden_dim]),
                    edge_dim=self.num_edge_features,
                    train_eps=False,
                )
            )
            self.convolution_batch_norm.append(BatchNorm(hidden_dim))
            if self.residual_connection == "dense":
                input_dim += hidden_dim
            if self.residual_connection == "res":
                input_dim = hidden_dim

        self.linear = Linear(input_dim, output_dim)
        if pooling == "mean":
            self.pooling = global_mean_pool
        elif pooling == "add":
            self.pooling = global_add_pool

    def forward(self, data: Data) -> Tensor:
        x = data.x
        x = self.node_embedding(x)

        for i, conv in enumerate(self.convolution):
            x_conv = conv(x, data.edge_index, data.edge_attr)
            x_conv = self.convolution_batch_norm[i](x_conv)
            x_conv = torch.nn.functional.gelu(x_conv)
            if self.residual_connection == "dense":
                x = torch.cat((x, x_conv), dim=1)
            if self.residual_connection == "res":
                x = x + x_conv
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.linear(x)
        representation = self.pooling(x, data.batch)

        return representation


class NNConvEncoder(torch.nn.Module):
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
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_conv_layers = n_conv_layers
        self.num_edge_features = num_edge_features
        self.dropout = dropout
        self.residual_connection = residual_connection

        node_embedding_dim = hidden_dim
        self.node_embedding = Linear(input_dim, node_embedding_dim)
        input_dim = node_embedding_dim

        self.convolution = torch.nn.ModuleList()
        self.convolution_batch_norm = torch.nn.ModuleList()

        for _ in range(self.n_conv_layers):
            self.convolution.append(
                NNConv(
                    in_channels=input_dim,
                    out_channels=hidden_dim,
                    nn=MLP([self.num_edge_features, input_dim * hidden_dim]),
                )
            )
            self.convolution_batch_norm.append(BatchNorm(hidden_dim))
            if self.residual_connection == "dense":
                input_dim += hidden_dim
            if self.residual_connection == "res":
                input_dim = hidden_dim

        self.linear = Linear(input_dim, output_dim)
        if pooling == "mean":
            self.pooling = global_mean_pool
        elif pooling == "add":
            self.pooling = global_add_pool

    def forward(self, data: Data) -> Tensor:
        x = data.x
        x = self.node_embedding(x)

        for i, conv in enumerate(self.convolution):
            x_conv = conv(x, data.edge_index, data.edge_attr)
            x_conv = self.convolution_batch_norm[i](x_conv)
            x_conv = torch.nn.functional.gelu(x_conv)
            if self.residual_connection == "dense":
                x = torch.cat((x, x_conv), dim=1)
            if self.residual_connection == "res":
                x = x + x_conv
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.linear(x)
        representation = self.pooling(x, data.batch)

        return representation
