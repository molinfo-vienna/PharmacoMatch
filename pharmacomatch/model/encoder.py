from math import pi as PI

import torch
from torch import Tensor
from torch.nn import Linear, Sequential
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import (
    BatchNorm,
    MLP,
    NNConv,
    GINEConv,
    global_mean_pool,
    global_add_pool,
    MessagePassing,
    GATv2Conv,
)


class GINEncoder(torch.nn.Module):
    """Implementation of the encoder model based on the NNConv layer.

    Args:
        input_dim (int): Input feature dimension of the node features. Equals the number
            of pharmacophore feature types.
        node_embedding_dim (int): Initial node embedding dimension, transforms the OHE
            input features through a dense layer.
        hidden_dim (int): Output dimension of the convolutional MPNN layers
        output_dim (int): Output dimension of the encoder model, projection dimension of
            a dense layer before the node aggregation function.
        n_conv_layers (int): Number of convolutional layers in the MPNN.
        num_edge_features (int): Edge feature dimension, number of bins of the distance
            encoding.
        dropout (float): Dropout rate of the MPNN layers.
        residual_connection (str): Residual connection type, either "res" or "dense".
            "res" adds the input to the output of the convolutional layer, "dense"
            concatenates the input to the output vector.
        pooling (str, optional): Aggregation operator for graph-level read-out, either
            "mean" or "add". Defaults to "add".
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_conv_layers: int,
        num_edge_features: int,
        dropout: float,
        residual_connection: str,
        pooling: str = "add",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_conv_layers = n_conv_layers
        self.num_edge_features = num_edge_features
        self.dropout = dropout
        self.residual_connection = residual_connection

        self.node_embedding = Linear(input_dim, hidden_dim)
        input_dim = hidden_dim

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
    """Implementation of the encoder model based on the NNConv layer.

    Args:
        input_dim (int): Input feature dimension of the node features. Equals the number
            of pharmacophore feature types.
        node_embedding_dim (int): Initial node embedding dimension, transforms the OHE
            input features through a dense layer.
        hidden_dim (int): Output dimension of the convolutional MPNN layers
        output_dim (int): Output dimension of the encoder model, projection dimension of
            a dense layer before the node aggregation function.
        n_conv_layers (int): Number of convolutional layers in the MPNN.
        num_edge_features (int): Edge feature dimension, number of bins of the distance
            encoding.
        dropout (float): Dropout rate of the MPNN layers.
        residual_connection (str): Residual connection type, either "res" or "dense".
            "res" adds the input to the output of the convolutional layer, "dense"
            concatenates the input to the output vector.
        pooling (str, optional): Aggregation operator for graph-level read-out, either
            "mean" or "add". Defaults to "add".
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_conv_layers: int,
        num_edge_features: int,
        dropout: float,
        residual_connection: str,
        pooling: str = "add",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_conv_layers = n_conv_layers
        self.num_edge_features = num_edge_features
        self.dropout = dropout
        self.residual_connection = residual_connection

        self.node_embedding = Linear(input_dim, hidden_dim)
        input_dim = hidden_dim

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


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(x) - self.shift


class CFConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_edge_features: int,
    ):
        super().__init__(aggr="add")
        self.lin1 = Linear(in_channels, out_channels, bias=False)
        self.lin2 = Linear(out_channels, out_channels)
        self.mlp = Sequential(
            Linear(num_edge_features, out_channels),
            ShiftedSoftplus(),
            Linear(out_channels, out_channels),
        )
        self.act = ShiftedSoftplus()
        self.lin = Linear(out_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[2].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        # C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        # Learnable weights from distances
        W = self.mlp(edge_attr)  # * C.view(-1, 1)
        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        x = self.act(x)
        x = self.lin(x)
        return x

    def message(self, x_j: Tensor, W: Tensor) -> Tensor:
        return x_j * W


class SchnetEncoder(torch.nn.Module):
    """Implementation of the encoder model based on the NNConv layer.

    Args:
        input_dim (int): Input feature dimension of the node features. Equals the number
            of pharmacophore feature types.
        node_embedding_dim (int): Initial node embedding dimension, transforms the OHE
            input features through a dense layer.
        hidden_dim (int): Output dimension of the convolutional MPNN layers
        output_dim (int): Output dimension of the encoder model, projection dimension of
            a dense layer before the node aggregation function.
        n_conv_layers (int): Number of convolutional layers in the MPNN.
        num_edge_features (int): Edge feature dimension, number of bins of the distance
            encoding.
        dropout (float): Dropout rate of the MPNN layers.
        residual_connection (str): Residual connection type, either "res" or "dense".
            "res" adds the input to the output of the convolutional layer, "dense"
            concatenates the input to the output vector.
        pooling (str, optional): Aggregation operator for graph-level read-out, either
            "mean" or "add". Defaults to "add".
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_conv_layers: int,
        num_edge_features: int,
        dropout: float,
        residual_connection: str,
        pooling: str = "add",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_conv_layers = n_conv_layers
        self.num_edge_features = num_edge_features
        self.dropout = dropout
        self.residual_connection = residual_connection

        self.node_embedding = Linear(input_dim, hidden_dim)
        input_dim = hidden_dim

        self.convolution = torch.nn.ModuleList()
        self.convolution_batch_norm = torch.nn.ModuleList()

        for _ in range(self.n_conv_layers):
            self.convolution.append(
                CFConv(
                    in_channels=input_dim,
                    out_channels=hidden_dim,
                    num_edge_features=num_edge_features,
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
