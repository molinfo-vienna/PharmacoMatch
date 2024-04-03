from typing import Generator, Union
import random

from flash.core.optimizers import LARS, LinearWarmupCosineAnnealingLR
from lightning import LightningModule
import torch
from torch.optim import Optimizer, AdamW, Adam
from torch import Tensor
from torch_geometric.data import Data
from torchmetrics import ROC, Accuracy

from dataset import AugmentationModule
from .encoder import GATEncoder, PointTransformerEncoder, GINEncoder
from .projector import ProjectionPhectorMatch


class PhectorMatch(LightningModule):
    def __init__(self, **params) -> None:
        super(PhectorMatch, self).__init__()
        self.save_hyperparameters()

        # SimCLR architecture
        self.transform = AugmentationModule(
            train=True,
            node_masking=self.hparams.node_masking,
            radius=self.hparams.radius,
        )
        self.negative_target_transform = AugmentationModule(
            train=True,
            node_masking=None,
            sphere_surface_sampling=True,
            radius=self.hparams.radius_negative,
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
                pooling=self.hparams.pooling,
            )

        if self.hparams.encoder == "GIN":
            self.encoder = GINEncoder(
                input_dim=self.hparams.num_node_features,
                node_embedding_dim=self.hparams.node_embedding_dim,
                hidden_dim=self.hparams.hidden_dim_encoder,
                output_dim=self.hparams.input_dim_projector,
                n_conv_layers=self.hparams.n_layers_conv,
                num_edge_features=self.hparams.num_edge_features,
                dropout=self.hparams.dropout,
                residual_connection=self.hparams.residual_connection,
                pooling=self.hparams.pooling,
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

        self.projection_head = ProjectionPhectorMatch(
            input_dim=self.hparams.input_dim_projector,
            hidden_dim=self.hparams.hidden_dim_projector,
            output_dim=self.hparams.output_dim_projector,
            normalize=False,
        )

    def forward(self, data: Data) -> Tensor:
        representation = self.encoder(data)
        embedding = self.projection_head(representation, data.num_ph4_features)

        return embedding

    def setup(self, stage: str) -> None:
        global_batch_size = self.trainer.world_size * self.hparams.batch_size
        self.train_iters_per_epoch = self.hparams.num_samples // global_batch_size

        for param in self.encoder.parameters():
            param.requires_grad = False

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict]]:
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
        )

        return [optimizer]

    def loss(
        self,
        queries: Tensor,
        targets: Tensor,
        negative_targets: Tensor,
    ) -> Tensor:
        # Extract number of features per ph4
        queries_num_features = queries.num_ph4_features
        targets_num_features = targets.num_ph4_features
        negative_targets_num_features = negative_targets.num_ph4_features
        num_features = torch.cat(
            (queries_num_features, targets_num_features, negative_targets_num_features)
        )

        # Extract the embeddings
        queries = self(queries)
        targets = self(targets)
        negative_targets = self(negative_targets)
        batch_size = len(queries)

        # Regularization loss term
        vector_norm = torch.norm(
            torch.cat((queries, targets, negative_targets)), p=1, dim=1
        )

        regularization_term = 0.01 * torch.sum((vector_norm - num_features) ** 2)

        # Positive loss term
        positives = torch.sum(
            torch.max(
                torch.zeros_like(targets),
                queries - targets,
            )
            ** 2,
            dim=1,
        )

        # Negative loss term
        negatives_1 = torch.sum(
            torch.max(
                torch.zeros_like(targets),
                queries - negative_targets,
            )
            ** 2,
            dim=1,
        )
        negatives_2 = torch.sum(
            torch.max(
                torch.zeros_like(targets),
                queries - torch.roll(targets, batch_size // 2, dims=0),
            )
            ** 2,
            dim=1,
        )
        negatives_3 = torch.sum(
            torch.max(
                torch.zeros_like(queries),
                queries - torch.roll(queries, batch_size // 2, dims=0),
            )
            ** 2,
            dim=1,
        )
        negatives_4 = torch.sum(
            torch.max(
                torch.zeros_like(targets),
                targets - torch.roll(targets, batch_size // 2, dims=0),
            )
            ** 2,
            dim=1,
        )

        # Calculate the accuracy of the positives and negatives w.r.t. the embedding space property
        accuracy_positives = torch.sum(positives <= 0) / batch_size
        accuracy_negatives_1 = 1 - (torch.sum(negatives_1 <= 0) / batch_size)
        accuracy_negatives_2 = 1 - (torch.sum(negatives_2 <= 0) / batch_size)
        accuracy_negatives_3 = 1 - (torch.sum(negatives_3 <= 0) / batch_size)
        accuracy_negatives_4 = 1 - (torch.sum(negatives_4 <= 0) / batch_size)

        negatives = torch.cat(
            [negatives_1, negatives_2, negatives_3, negatives_4], dim=0
        )
        # negatives = torch.cat([negatives_1, negatives_2], dim=0)

        # right here I can calculate the best threshold via auroc, and the MCC of the positives and negatives
        # threshold = 0.01
        # target = torch.cat(
        #     (
        #         torch.ones(batch_size, device=positives.device),
        #         torch.zeros(batch_size * 4, device=positives.device),
        #     )
        # )
        # pred = torch.cat((positives, negatives))  # <= threshold
        # # pred = torch.flip(pred, [0])
        # pred_normalized = -pred
        # pred_normalized = pred_normalized - torch.min(pred_normalized)
        # pred_normalized = pred_normalized / torch.max(pred_normalized)
        # roc = ROC(task="binary")
        # fpr, tpr, thresholds = roc(pred_normalized, target)
        # j = tpr - fpr
        # idx = torch.argmax(j)
        # best_threshold = pred[idx]
        # pred = pred <= best_threshold
        # self.mcc.update(pred.int(), target.int())

        negatives = torch.max(
            torch.tensor(0.0, device=queries.device),
            self.hparams.margin - negatives,
        )

        return (
            torch.sum(positives) + torch.sum(negatives) + regularization_term,
            accuracy_positives,
            accuracy_negatives_1,
            accuracy_negatives_2,
            accuracy_negatives_3,
            accuracy_negatives_4,
        )

    def shared_step(self, batch: Data, batch_idx: int) -> tuple[Tensor, Tensor]:
        queries = self.transform(batch)
        targets = self.val_transform(batch)
        negative_targets = self.negative_target_transform(batch)

        return self.loss(queries, targets, negative_targets)

    def training_step(self, batch: Data, batch_idx: int) -> Tensor:
        (
            loss,
            accuracy_positives,
            accuracy_negatives_1,
            accuracy_negatives_2,
            accuracy_negatives_3,
            accuracy_negatives_4,
        ) = self.shared_step(batch, batch_idx)

        self.log(
            "train/train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )
        self.log(
            "accuracy/train_accuracy_positives",
            accuracy_positives,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )
        self.log(
            "accuracy/train_accuracy_negatives_1",
            accuracy_negatives_1,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )
        self.log(
            "accuracy/train_accuracy_negatives_2",
            accuracy_negatives_2,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )
        self.log(
            "accuracy/train_accuracy_negatives_3",
            accuracy_negatives_3,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )
        self.log(
            "accuracy/train_accuracy_negatives_4",
            accuracy_negatives_4,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )

        return loss

    def validation_step(
        self, batch: Data, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        # val loss calculation
        if dataloader_idx == 0:
            (
                loss,
                accuracy_positives,
                accuracy_negatives_1,
                accuracy_negatives_2,
                accuracy_negatives_3,
                accuracy_negatives_4,
            ) = self.shared_step(batch, batch_idx)

            self.log(
                "val/inner_val_loss",
                loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=len(batch),
            )
            self.log(
                "accuracy/inner_val_accuracy_positives",
                accuracy_positives,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=len(batch),
            )
            self.log(
                "accuracy/inner_val_accuracy_negatives_1",
                accuracy_negatives_1,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=len(batch),
            )
            self.log(
                "accuracy/inner_val_accuracy_negatives_2",
                accuracy_negatives_2,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=len(batch),
            )
            self.log(
                "accuracy/inner_val_accuracy_negatives_3",
                accuracy_negatives_3,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=len(batch),
            )
            self.log(
                "accuracy/inner_val_accuracy_negatives_4",
                accuracy_negatives_4,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=len(batch),
            )

            return loss

        if dataloader_idx == 1:
            (
                loss,
                accuracy_positives,
                accuracy_negatives_1,
                accuracy_negatives_2,
                accuracy_negatives_3,
                accuracy_negatives_4,
            ) = self.shared_step(batch, batch_idx)

            self.log(
                "val/outer_val_loss",
                loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=len(batch),
            )
            self.log(
                "accuracy/outer_val_accuracy_positives",
                accuracy_positives,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=len(batch),
            )
            self.log(
                "accuracy/outer_val_accuracy_negatives_1",
                accuracy_negatives_1,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=len(batch),
            )
            self.log(
                "accuracy/outer_val_accuracy_negatives_2",
                accuracy_negatives_2,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=len(batch),
            )
            self.log(
                "accuracy/outer_val_accuracy_negatives_3",
                accuracy_negatives_3,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=len(batch),
            )
            self.log(
                "accuracy/outer_val_accuracy_negatives_4",
                accuracy_negatives_4,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=len(batch),
            )

            return loss

    def on_train_start(self):
        self.logger.log_hyperparams(
            self.hparams,
            {
                "train/train_loss": torch.inf,
                "val/inner_val_loss/dataloader_idx_0": torch.inf,
                "val/outer_val_loss/dataloader_idx_1": torch.inf,
            },
        )

    def predict_step(
        self, batch: Data, batch_idx: int
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        if "mol_id" in batch.keys:
            return self(self.val_transform(batch)), batch.mol_id
        else:
            return self(self.val_transform(batch))
