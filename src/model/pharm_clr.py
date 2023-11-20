from typing import Any
import torch
from torch.nn import Linear
import torch.nn.functional as F
from flash.core.optimizers import LARS, LinearWarmupCosineAnnealingLR
import math

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

class Projection(torch.nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Projection head MLP
        self.projection_head = torch.nn.Sequential(
            Linear(self.input_dim, self.hidden_dim, bias=True),
            torch.nn.BatchNorm1d(self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, x):
        x = self.projection_head(x)
        return F.normalize(x, p=2, dim=1)
    

class Encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_conv_layers, num_edge_features, dropout):
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
        for _ in range(self.n_conv_layers):
            self.convolution.append(
                GATConv(# GAT is maybe not suitable for our problem
                    input_dim,
                    hidden_dim,
                    #nn=Linear(params["num_edge_features"], input_dimension*output_dim)
                    edge_dim=self.num_edge_features,
                    heads=self.num_heads,
                    concat=False,
                )
            )
            self.convolution_batch_norm.append(BatchNorm(hidden_dim))
            input_dim += hidden_dim

        # Expand dimensionality before node pooling
        self.linear = Linear(input_dim, output_dim)
        self.batch_norm = BatchNorm(output_dim)
        self.pooling = global_mean_pool
        
        #input_dimension = output_dim
        # Graph read-out via node-pooling
        #self.gmt = GMT(input_dimension, input_dimension, heads=input_dimension)

    def forward(self, data):
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
        x = self.linear(x)
        x = torch.nn.functional.gelu(x)
        x = self.batch_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Graph-level read-out
        representation = self.pooling(x, data.batch)

        return representation

class PharmCLR(LightningModule):
    def __init__(self, **params):
        super(PharmCLR, self).__init__()
        self.save_hyperparameters()

        # SimCLR architecture
        self.transform = AugmentationModule(train=True)
        self.val_transform = AugmentationModule(train=False)
        self.encoder = Encoder(input_dim=self.hparams.num_node_features, 
                               hidden_dim=self.hparams.output_dims_conv,
                               output_dim=self.hparams.output_dims_lin, 
                               n_conv_layers=self.hparams.n_layers_conv, 
                               num_edge_features=self.hparams.num_edge_features,
                               dropout=self.hparams.dropout
                               )
        input_dimension = self.hparams.output_dims_lin
        self.projection_head = Projection(input_dimension, input_dimension, 64)

        # validation embeddings
        self.val_embeddings = []

    def forward(self, data):
        # Optional: Visualization of the input pharmacophore
        # visualize_pharm(
        #     [data[0].clone(), self.transform(data[0].clone()), self.transform(data[0].clone())]
        # )

        representation = self.encoder(data)
        embedding = self.projection_head(representation)

        return embedding
    
    def setup(self, stage):
        global_batch_size = self.trainer.world_size * self.hparams.batch_size
        self.train_iters_per_epoch = self.hparams.num_samples // global_batch_size
    
    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=['bias', 'bn']):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {'params': params, 'weight_decay': weight_decay},
            {'params': excluded_params, 'weight_decay': 0.}
        ]
    
    def configure_optimizers(self):
        # TRICK 1 (Use lars + filter weights)
        # exclude certain parameters
        parameters = self.exclude_from_wt_decay(
            self.named_parameters(),
            weight_decay=self.hparams.opt_weight_decay
        )

        optimizer = LARS(parameters, lr=self.hparams.learning_rate, momentum=self.hparams.opt_eta)

        # Trick 2 (after each step)
        self.hparams.warmup_epochs = self.hparams.warmup_epochs * self.train_iters_per_epoch
        max_epochs = self.trainer.max_epochs * self.train_iters_per_epoch

        linear_warmup_cosine_decay = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=max_epochs,
            warmup_start_lr=0,
            eta_min=0
        )

        scheduler = {
            'scheduler': linear_warmup_cosine_decay,
            'interval': 'step',
            'frequency': 1
        }

        return [optimizer], [scheduler]

    # def configure_optimizers(self):
    #     optimizer = LARS(self.parameters(), lr=self.hparams.learning_rate*math.sqrt(self.hparams.batch_size), weight_decay=1e-6, momentum=1e-3)
    #     return optimizer

    def nt_xent_loss(self, out_1, out_2):
        out = torch.cat([out_1, out_2], dim=0)
        n_samples = len(out)

        # Full similarity matrix
        cov = torch.mm(out, out.t().contiguous())
        sim = torch.exp(cov / self.hparams.temperature)

        mask = ~torch.eye(n_samples, device=sim.device).bool()
        neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)

        # Positive similarity
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.hparams.temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / neg).mean()
        return loss

    @classmethod
    def get_hyperparams(cls):
        hyperparams = dict(
            learning_rate=5e-2,
            dropout=0.1,
            n_layers_conv=3,
            output_dims_conv=32,
            output_dims_lin=1024,
            temperature=0.5,
        )

        return hyperparams
    
    def shared_step(self, batch, batch_idx):
        out1 = self(self.transform(batch))
        out2 = self(self.transform(batch))

        return self.nt_xent_loss(out1, out2)

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
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
        val_loss = self.shared_step(batch, batch_idx)
        self.log(
            "hp/val_loss",
            val_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )

        # rankme criterion
        self.val_embeddings.append(self(self.val_transform(batch)))
        
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
        return self(self.val_transform(batch)), batch.mol_id
    
