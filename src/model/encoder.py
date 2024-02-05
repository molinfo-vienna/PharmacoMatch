import torch
from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool, BatchNorm


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
