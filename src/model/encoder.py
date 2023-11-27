import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool, BatchNorm


class Encoder(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        n_conv_layers,
        num_edge_features,
        dropout,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_conv_layers = n_conv_layers
        self.num_edge_features = num_edge_features
        self.num_heads = 10
        self.dropout = dropout

        embedding_dim = 10
        self.node_embedding = Linear(input_dim, embedding_dim)
        input_dim = embedding_dim

        # Convolutional layers
        # try out spectral layers, maybe 3-6 layers
        self.convolution = torch.nn.ModuleList()
        self.convolution_batch_norm = torch.nn.ModuleList()

        # input and outputdimension maybe bigger
        # input to say 32-64, output to maybe 25
        for i in range(self.n_conv_layers):
            self.convolution.append(
                GATConv(  # GAT is maybe not suitable for our problem
                    input_dim,
                    hidden_dim,
                    # nn=Linear(params["num_edge_features"], input_dimension*output_dim)
                    edge_dim=self.num_edge_features,
                    heads=self.num_heads,
                    concat=False,
                )
            )
            self.convolution_batch_norm.append(BatchNorm(hidden_dim))
            input_dim += hidden_dim

        # Expand dimensionality before node pooling
        self.linear1 = Linear(input_dim, output_dim)
        self.batch_norm = BatchNorm(output_dim)
        self.linear2 = Linear(output_dim, output_dim)
        self.pooling = global_mean_pool

        # input_dimension = output_dim
        # Graph read-out via node-pooling
        # self.gmt = GMT(input_dimension, input_dimension, heads=input_dimension)

    def forward(self, data):
        # Embedding of OHE features
        x = data.x
        x = self.node_embedding(x)

        # Graph convolution via message passing
        for i, conv in enumerate(self.convolution):
            x_conv = conv(x, data.edge_index, data.edge_attr)
            x_conv = self.convolution_batch_norm[i](x_conv)
            x_conv = torch.nn.functional.gelu(x_conv)
            # x_conv = F.dropout(x_conv, p=self.dropout, training=self.training)
            x = torch.cat((x, x_conv), dim=1)

        # Dimensionality expandion before read-out
        x = self.linear1(x)
        x = self.batch_norm(x)
        x = torch.nn.functional.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear2(x)

        # Graph-level read-out
        representation = self.pooling(x, data.batch)

        return representation
