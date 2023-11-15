from typing import Any
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

class PharmCLR(LightningModule):
    def __init__(self, hyperparams, params):
        super(PharmCLR, self).__init__()
        self.save_hyperparameters()

        # Settings
        self.learning_rate = hyperparams["learning_rate"]
        self.dropout = hyperparams["dropout"]
        self.heads = 10
        self.temperature = hyperparams["temperature"]

        # Augmentation
        self.transform = AugmentationModule(train=True)
        self.val_transform = AugmentationModule(train=False)

        # Embedding layer
        input_dimension = params["num_node_features"]
        embedding_dim = 10
        self.node_embedding = Linear(input_dimension, embedding_dim)
        input_dimension = embedding_dim

        # Convolutional layers
        # try out spectral layers, maybe 3-6 layers
        self.convolution = torch.nn.ModuleList()
        output_dim = hyperparams["output_dims_conv"]
        self.convolution_batch_norm = torch.nn.ModuleList()

        # input and outputdimension maybe bigger
        # input to say 32-64, output to maybe 25
        for _ in range(hyperparams["n_layers_conv"]):
            self.convolution.append(
                GATv2Conv(# GAT is maybe not suitable for our problem
                    input_dimension,
                    output_dim,
                    #nn=Linear(params["num_edge_features"], input_dimension*output_dim)
                    edge_dim=params["num_edge_features"],
                    heads=self.heads,
                    concat=False,
                )
            )
            self.convolution_batch_norm.append(BatchNorm(output_dim))
            input_dimension += output_dim

        # Expand dimensionality before node pooling
        output_dim = hyperparams["output_dims_lin"]
        self.linear1 = Linear(input_dimension, output_dim)
        self.batch_norm = BatchNorm(output_dim)
        input_dimension = output_dim

        # Graph read-out via node-pooling
        self.gmt = GMT(input_dimension, input_dimension, heads=input_dimension)
        # --> This will yield the final representation

        # Projection head MLP
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(input_dimension, output_dim),
            torch.nn.GELU(),
            torch.nn.Linear(input_dimension, output_dim // 2),
        )

        # validation embeddings
        self.val_embeddings = []

    def forward(self, data):
        # Optional: Visualization of the input pharmacophore
        # visualize_pharm(
        #     [data[0].clone(), self.transform(data[0].clone()), self.transform(data[0].clone())]
        # )

        # Embedding of OHE features
        x = data.x
        x = self.node_embedding(x)

        # Graph convolution via message passing
        for i, conv in enumerate(self.convolution):
            x_conv = conv(x, data.edge_index, data.edge_attr)
            x_conv = torch.nn.functional.gelu(x_conv)
            x_conv = self.convolution_batch_norm[i](x_conv)
            x_conv = F.dropout(x_conv, p=self.dropout, training=self.training)
            x = torch.cat((x, x_conv), dim=1)

        # Dimensionality expandion before read-out
        x = self.linear1(x)
        x = torch.nn.functional.gelu(x)
        x = self.batch_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Graph-level read-out
        representation = self.gmt(x, data.batch)

        # Apply the projection_head
        embedding = self.projection_head(representation)

        return embedding

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
            learning_rate=1e-2,
            dropout=0.1,
            n_layers_conv=3,
            output_dims_conv=32,
            output_dims_lin=64,
            temperature=0.5,
        )

        return hyperparams
    
    def shared_step(self, batch):
        batch1, batch2 = batch
        out1 = self(batch1)
        out2 = self(batch2)
        batch_size, _ = out1.shape
        out = torch.cat((out1, out2), dim=1).reshape(batch_size * 2, -1)
        loss = self.nt_xent_loss(out)

        return loss
    
    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        if self.trainer.training:
            return self.transform(batch)
        if self.trainer.evaluating:
            return self.val_transform(batch)
        else:
            return batch

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log(
            "hp/train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=len(batch),
        )

        return loss
    
    def on_validation_epoch_start(self) -> None:
        self.val_embeddings = []

    def validation_step(self, batch, batch_idx):
        # val loss calculation
        batch1, batch2, batch3 = batch
        val_loss = self.shared_step((batch1, batch2))
        self.log(
            "hp/val_loss",
            val_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=len(batch),
        )

        # rankme criterion
        self.val_embeddings.append(self(batch3))
        
        return val_loss
    
    def on_validation_epoch_end(self) -> None:
        epsilon = 1e-7
        embeddings = torch.vstack(self.val_embeddings)
        _, singular_values, _ = torch.svd(embeddings)
        singular_values /= torch.sum(singular_values)
        singular_values += epsilon
        rank_me = torch.exp(-torch.sum(singular_values*torch.log(singular_values)))
        self.log(
            "hp/rank_me",
            rank_me,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    def on_train_start(self):
        self.logger.log_hyperparams(
            self.hparams,
            {
                "hp/rank_me": 0
            },
        )

    def predict_step(self, batch, batch_idx):
        return self(batch), batch.mol_id
    
