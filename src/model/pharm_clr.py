import torch
from torch.nn import Linear
import torch.nn.functional as F

# layers that accept edge_attr
from torch_geometric.nn import (
    GATConv,
    GATv2Conv,
    TransformerConv,
    GINEConv,
    GMMConv,
    SplineConv,
    NNConv,
    CGConv,
    PNAConv,
    GENConv,
    PDNConv,
    GeneralConv,
)

# layers that accept edge_weights
from torch_geometric.nn import GCNConv
from torch_geometric.nn import (
    global_mean_pool,
    global_max_pool,
    global_add_pool,
    SAGPooling,
    BatchNorm,
    LayerNorm,
)
from torch_geometric.transforms import *
from torch_geometric.nn.aggr import GraphMultisetTransformer as GMT

from .lightning_module import CustomLightningModule


class PharmCLR(CustomLightningModule):
    def __init__(self, hyperparams, params):
        super(PharmCLR, self).__init__(hyperparams, params)

        # Further settings
        self.learning_rate = hyperparams["learning_rate"]
        self.dropout = hyperparams["dropout"]
        self.heads = 10
        self.temperature = hyperparams["temperature"]

        # Embedding layer
        input_dimension = params["num_node_features"]
        embedding_dim = 10
        self.node_embedding = Linear(input_dimension, embedding_dim)
        input_dimension = embedding_dim

        # Convolutional layers
        self.convolution = torch.nn.ModuleList()
        output_dim = hyperparams["output_dims_conv"]
        self.convolution_batch_norm = torch.nn.ModuleList()

        for _ in range(hyperparams["n_layers_conv"]):
            self.convolution.append(
                GATv2Conv(
                    input_dimension,
                    output_dim,
                    edge_dim=params["num_edge_features"],
                    heads=self.heads,
                    concat=False,
                )
            )
            self.convolution_batch_norm.append(BatchNorm(output_dim))
            input_dimension += output_dim

        # Node pooling layer
        self.gmt = GMT(input_dimension, input_dimension, heads=input_dimension)
        output_dim = hyperparams["output_dims_lin"]
        self.linear = Linear(input_dimension, output_dim)
        # --> This should yield the final representation

        # self.batchnorm1 = BatchNorm(output_dim)
        input_dimension = output_dim

        # Projection head MLP
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(input_dimension, input_dimension),
            torch.nn.GELU(),
            torch.nn.Linear(input_dimension, input_dimension // 2),
        )

    def forward(self, data):
        x = data.x
        x = self.node_embedding(x)
        for i, conv in enumerate(self.convolution):
            x_conv = conv(x, data.edge_index, data.edge_attr)
            x_conv = torch.nn.functional.gelu(x_conv)
            x_conv = self.convolution_batch_norm[i](x_conv)
            x_conv = F.dropout(x_conv, p=self.dropout, training=self.training)
            x = torch.cat((x, x_conv), dim=1)

        # Graph-level read-out
        x = self.gmt(x, data.batch)
        x = self.linear(x)
        #x = torch.nn.functional.gelu(x)
        #x = self.batchnorm(x)
        #x = F.dropout(x, p=self.dropout, training=self.training)

        # 3. Apply the projection_head
        x = self.projection_head(x)
        #x = x.softmax(dim=1)

        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adagrad(self.parameters(), lr=self.learning_rate)
        return optimizer

    def nt_xent_loss(self, x):
        assert len(x.size()) == 2

        # Cosine similarity
        xcs = F.cosine_similarity(x[None, :, :], x[:, None, :], dim=-1)
        xcs[torch.eye(x.size(0)).bool()] = float("-inf")

        # Ground truth labels
        target = torch.arange(x.size(0)).cuda()
        target[0::2] += 1
        target[1::2] -= 1

        # Standard cross-entropy loss
        return F.cross_entropy(xcs / self.temperature, target, reduction="mean")

    @classmethod
    def get_hyperparams(cls):
        hyperparams = dict(
            learning_rate=1e-2,
            dropout=0.1,
            n_layers_conv=3,
            output_dims_conv=12,
            output_dims_lin=64,
            temperature=0.5,
        )

        return hyperparams

    # @classmethod
    # def get_hyperparams(cls, trial):
    #     learning_rate = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    #     n_layers_conv = trial.suggest_int("n_conv_layers", 1, 3)
    #     output_dims_conv = trial.suggest_int(f"n_units_l_conv", 4, 64, log=True)
    #     output_dims_lin = trial.suggest_int(f"n_units_l_lin", 4, 128, log=True)
    #     dropout = trial.suggest_float("dropout", 0.05, 0.35)

    #     hyperparams = dict(
    #         learning_rate=learning_rate,
    #         dropout=dropout,
    #         n_layers_conv=n_layers_conv,
    #         output_dims_conv=output_dims_conv,
    #         output_dims_lin=output_dims_lin,
    #     )

    #     return hyperparams
