import torch
from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import (
    BatchNorm,
    MLP,
    NNConv,
    global_mean_pool,
    global_add_pool,
)


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
