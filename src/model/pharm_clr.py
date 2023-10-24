import torch
from torch.nn import Linear
import torch.nn.functional as F
from flash.core.optimizers import LARS

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

# from torch_geometric.transforms import *
from dataset import *
from torch_geometric import transforms as T
from torch_geometric.nn.aggr import GraphMultisetTransformer as GMT

# from .lightning_module import CustomLightningModule
from lightning import LightningModule

from utils import *


# class DataAugmentation(torch.nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.transform = T.Compose(
#             [RandomMasking(), RandomGaussianNoise(), T.KNNGraph(k=50), T.ToUndirected(), T.Distance(norm=False), DistanceRDF()]
#         )

#     @torch.no_grad()
#     def forward(self, data):
#         data_out = self.transform(data)
#         return data_out


class PharmCLR(LightningModule):
    def __init__(self, hyperparams, params):
        super(PharmCLR, self).__init__()
        self.save_hyperparameters()

        # Settings
        self.learning_rate = hyperparams["learning_rate"]
        self.dropout = hyperparams["dropout"]
        self.heads = 10
        self.temperature = hyperparams["temperature"]

        # Data Augmentation
        # self.transform = DataAugmentation()
        self.transform = T.Compose(
            [
                RandomMasking(),
                RandomGaussianNoise(),
                T.KNNGraph(k=50),
                T.ToUndirected(),
                T.Distance(norm=False),
                DistanceRDF(),
            ]
        )

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
        # visualize_pharm(
        #     [data[0].clone(), self.transform(data[0].clone()), self.transform(data[0].clone())]
        # )
        data = self.transform(data.clone())
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
        # x = torch.nn.functional.gelu(x)
        # x = self.batchnorm(x)
        # x = F.dropout(x, p=self.dropout, training=self.training)

        # 3. Apply the projection_head
        x = self.projection_head(x)
        # x = x.softmax(dim=1)

        return x

    def configure_optimizers(self):
        optimizer = LARS(self.parameters(), lr=self.learning_rate)
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
            learning_rate=1e-4,
            dropout=0.1,
            n_layers_conv=3,
            output_dims_conv=12,
            output_dims_lin=64,
            temperature=0.5,
        )

        return hyperparams

    def training_step(self, batch, batch_idx):
        out1 = self(batch)
        out2 = self(batch)
        batch_size, _ = out1.shape
        out = torch.cat((out1, out2), dim=1).reshape(batch_size * 2, -1)
        loss = self.nt_xent_loss(out)
        self.log(
            "hp/train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )

        return loss

    def validation_step(self, batch, batch_idx):
        out1 = self(batch)
        out2 = self(batch)
        batch_size, _ = out1.shape
        out = torch.cat((out1, out2), dim=1).reshape(batch_size * 2, -1)
        val_loss = self.nt_xent_loss(out)
        self.log(
            "hp/val_loss",
            val_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )
        return val_loss

    def on_train_start(self):
        self.logger.log_hyperparams(
            self.hparams,
            {
                "hp/train_loss": 1,
                "hp/val_loss": 1,
            },
        )
