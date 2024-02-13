from typing import Generator, Union

import torch
from torch.optim import Optimizer
from flash.core.optimizers import LARS, LinearWarmupCosineAnnealingLR
from lightning import LightningModule
from torch import Tensor
from torch_geometric.data import Data
from dataset import AugmentationModule

from .encoder import GATEncoder, PointTransformerEncoder
from .projector import Projection


class PharmCLR(LightningModule):
    def __init__(self, **params) -> None:
        super(PharmCLR, self).__init__()
        self.save_hyperparameters()

        # SimCLR architecture
        self.transform = AugmentationModule(
            train=True,
            node_masking=self.hparams.node_masking,
            radius=self.hparams.radius,
        )
        self.val_transform = AugmentationModule(train=False)

        if self.hparams.encoder == "GAT":
            self.encoder = GATEncoder(
                input_dim=self.hparams.num_node_features,
                node_embedding_dim=self.hparams.node_embedding_dim,
                hidden_dim=self.hparams.hidden_dim_encoder,
                output_dim=self.hparams.input_dim_projector,
                n_conv_layers=self.hparams.n_layers_conv,
                num_edge_features=self.hparams.num_edge_features,
                dropout=self.hparams.dropout,
                residual_connection=self.hparams.residual_connection,
            )

        if self.hparams.encoder == "PointTransformerEncoder":
            self.encoder = PointTransformerEncoder(
                input_dim=self.hparams.num_node_features,
                hidden_dim=self.hparams.hidden_dim_encoder,
                output_dim=self.hparams.input_dim_projector,
                n_conv_layers=self.hparams.n_layers_conv,
                dropout=self.hparams.dropout,
                k=self.hparams.k,
            )

        self.projection_head = Projection(
            input_dim=self.hparams.input_dim_projector,
            hidden_dim=self.hparams.hidden_dim_projector,
            output_dim=self.hparams.output_dim_projector,
        )

        # validation embeddings
        self.val_embeddings = []

    def forward(self, data: Data) -> Tensor:
        # Optional: Visualization of the input pharmacophore
        # visualize_pharm(
        #     [data[0].clone(), self.transform(data[0].clone()), self.transform(data[0].clone())]
        # )

        representation = self.encoder(data)
        embedding = self.projection_head(representation)

        return embedding

    def setup(self, stage: str) -> None:
        global_batch_size = self.trainer.world_size * self.hparams.batch_size
        self.train_iters_per_epoch = self.hparams.num_samples // global_batch_size

    def exclude_from_wt_decay(
        self,
        named_params: Generator,
        weight_decay: float,
        skip_list: list[str] = ["bias", "bn"],
    ) -> list[dict]:
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
            {"params": params, "weight_decay": weight_decay},
            {"params": excluded_params, "weight_decay": 0.0},
        ]

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict]]:
        # TRICK 1 (Use lars + filter weights)
        # exclude certain parameters
        parameters = self.exclude_from_wt_decay(
            self.named_parameters(), weight_decay=self.hparams.weight_decay
        )

        optimizer = LARS(
            parameters, lr=self.hparams.learning_rate, momentum=self.hparams.momentum
        )

        # Trick 2 (after each step)
        warmup_epochs = self.hparams.warmup_epochs * self.train_iters_per_epoch
        max_epochs = self.trainer.max_epochs * self.train_iters_per_epoch

        linear_warmup_cosine_decay = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=max_epochs,
            warmup_start_lr=0,
            eta_min=0,
        )

        scheduler = {
            "scheduler": linear_warmup_cosine_decay,
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def nt_xent_loss(self, out_1: Tensor, out_2: Tensor) -> tuple[Tensor, Tensor]:
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

        # calc top1-accuracy
        mask = torch.eye(cov.shape[0], dtype=torch.bool, device=cov.device)
        cov.masked_fill_(mask, -1e4)
        mask = mask.roll(shifts=cov.shape[0] // 2, dims=0)

        comb_sim = torch.cat(
            [
                cov[mask][:, None],
                cov.masked_fill(mask, -1e4),
            ],  # First position positive example
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        accuracy = (sim_argsort == 0).float().mean()

        return loss, accuracy

    def shared_step(self, batch: Data, batch_idx: int) -> tuple[Tensor, Tensor]:
        out1 = self(self.transform(batch))
        out2 = self(self.transform(batch))

        return self.nt_xent_loss(out1, out2)

    def training_step(self, batch: Data, batch_idx: int) -> Tensor:
        loss, accuracy = self.shared_step(batch, batch_idx)
        self.log(
            "train/train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )
        self.log(
            "train/train_accuracy",
            accuracy,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )

        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_embeddings = []

    def validation_step(
        self, batch: Data, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        # val loss calculation
        if dataloader_idx == 0:
            val_loss, val_accuracy = self.shared_step(batch, batch_idx)
            self.log(
                "val/val_loss",
                val_loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=len(batch),
            )
            self.log(
                "val/val_accuracy",
                val_accuracy,
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
        rank_me = torch.exp(-torch.sum(singular_values * torch.log(singular_values)))
        self.log(
            "val/rank_me",
            rank_me,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    def on_train_start(self):
        self.logger.log_hyperparams(
            self.hparams,
            {
                "train/train_accuracy": 0,
                "val/val_accuracy/dataloader_idx_0": 0,
                "val/rank_me": 0,
            },
        )

    def predict_step(
        self, batch: Data, batch_idx: int
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        if "mol_id" in batch.keys:
            return self(self.val_transform(batch)), batch.mol_id
        else:
            return self(self.val_transform(batch))
