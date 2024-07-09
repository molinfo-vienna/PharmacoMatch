from typing import Union

from lightning import LightningModule
import torch
from torch.optim import Optimizer, Adam
from torch import Tensor
from torch_geometric.data import Data
from torchmetrics.classification import BinaryAUROC

from dataset import AugmentationModule, RandomNodeDeletion
from .encoder import GATEncoder, GINEncoder, NNConvEncoder
from .projector import ProjectionPhectorMatch


class PhectorMatch(LightningModule):
    def __init__(self, **params) -> None:
        super(PhectorMatch, self).__init__()
        self.save_hyperparameters()
        self.node_deletion = RandomNodeDeletion(3)
        self.query_transform = AugmentationModule(
            train=True,
            node_masking=None,
            radius=self.hparams.radius,
        )
        self.reference_transform = AugmentationModule(
            train=True,
            node_masking=None,
            radius=self.hparams.radius_negative,
        )
        self.negative_query_transform = AugmentationModule(
            train=True,
            node_masking=None,
            sphere_surface_sampling=True,
            radius=self.hparams.radius_negative,
        )
        self.target_transform = AugmentationModule(train=False)

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

        if self.hparams.encoder == "NNConv":
            self.encoder = NNConvEncoder(
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

        self.projection_head = ProjectionPhectorMatch(
            input_dim=self.hparams.input_dim_projector,
            hidden_dim=self.hparams.hidden_dim_projector,
            output_dim=self.hparams.output_dim_projector,
            num_layers=self.hparams.n_layers_projector,
            dropout=self.hparams.dropout_projector,
            norm=None,
        )

    def forward(self, data: Data) -> Tensor:
        representation = self.encoder(data)
        embedding = self.projection_head(representation, data.num_ph4_features)

        return embedding

    def setup(self, stage: str) -> None:
        global_batch_size = self.trainer.world_size * self.hparams.batch_size
        self.train_iters_per_epoch = self.hparams.num_samples // global_batch_size

        if self.hparams.freeze_encoder:
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
        reference_queries: Tensor,
        negative_queries: Tensor,
        targets: Tensor,
    ) -> Tensor:
        # Extract the embeddings
        queries = self(queries)
        reference_queries = self(reference_queries)
        negative_queries = self(negative_queries)
        targets = self(targets)
        batch_size = len(queries)

        # Regularization loss term
        # vector_norm = torch.norm(
        #     torch.cat((queries, negative_queries, targets)),
        #     p=self.hparams.regularization_p_norm,
        #     dim=1,
        # )

        # regularization_term = self.hparams.regularization_lambda * torch.sum(
        #     (vector_norm - num_features) ** 2
        # )

        # Positive loss term
        positives = torch.sum(
            torch.max(
                torch.zeros_like(targets),
                queries - targets,
            )
            ** 2,
            dim=1,
        )

        positives_reference = torch.sum(
            torch.max(
                torch.zeros_like(targets),
                reference_queries - targets,
            )
            ** 2,
            dim=1,
        )

        # Negative loss term - fine-grained negatives by displacement of tolerance sphere radius
        negatives_fine = torch.sum(
            torch.max(
                torch.zeros_like(targets),
                negative_queries - targets,
            )
            ** 2,
            dim=1,
        )

        # Coarse-grained negatives by mapping pairs that should not match
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
        negatives_coarse = torch.cat([negatives_2, negatives_3, negatives_4], dim=0)

        # Calculate the accuracy of the positives and negatives w.r.t. the embedding space property
        accuracy_positives = torch.sum(positives <= 0) / batch_size
        accuracy_reference = torch.sum(
            positives_reference <= self.hparams.margin
        ) / len(positives_reference)
        accuracy_negatives_1 = torch.sum(negatives_fine >= self.hparams.margin) / len(
            negatives_fine
        )
        accuracy_negatives_2 = torch.sum(negatives_coarse >= self.hparams.margin) / len(
            negatives_coarse
        )

        # Model performance is tracked via AUROC of the positive and negative pairs
        target = torch.cat(
            (
                torch.ones(batch_size, device=positives.device),
                torch.zeros(batch_size * 4, device=positives.device),
            )
        )
        pred = torch.cat((positives_reference, negatives_fine, negatives_coarse))
        pred_normalized = -pred
        pred_normalized = pred_normalized - torch.min(pred_normalized)
        pred_normalized = pred_normalized / torch.max(pred_normalized)
        auroc_metric = BinaryAUROC(task="binary")
        auroc = auroc_metric(pred_normalized, target)

        # Max-margin penalty for negative pairs
        negatives_fine = torch.max(
            torch.tensor(0.0, device=queries.device),
            self.hparams.margin - negatives_fine,
        )

        negatives_coarse = torch.max(
            torch.tensor(0.0, device=queries.device),
            self.hparams.margin - negatives_coarse,
        )

        return (
            self.hparams.positives_multiplier * torch.sum(positives)
            + torch.sum(negatives_fine)
            + torch.sum(negatives_coarse),
            auroc,
            accuracy_positives,
            accuracy_reference,
            accuracy_negatives_1,
            accuracy_negatives_2,
        )

    def shared_step(self, batch: Data, batch_idx: int) -> tuple[Tensor, Tensor]:
        targets = self.target_transform(batch)
        batch = self.node_deletion(batch)
        queries = self.query_transform(batch)
        reference_queries = self.reference_transform(batch)
        negative_queries = self.negative_query_transform(batch)

        return self.loss(queries, reference_queries, negative_queries, targets)

    def training_step(self, batch: Data, batch_idx: int) -> Tensor:
        (
            loss,
            auroc,
            accuracy_positives,
            accuracy_reference,
            accuracy_negatives_1,
            accuracy_negatives_2,
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
            "train/auroc",
            auroc,
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
            "accuracy/train_accuracy_reference",
            accuracy_reference,
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

        return loss

    def validation_step(
        self, batch: Data, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        if dataloader_idx == 0:
            (
                loss,
                auroc,
                accuracy_positives,
                accuracy_reference,
                accuracy_negatives_1,
                accuracy_negatives_2,
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
                "val/inner_val_auroc",
                auroc,
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
                "accuracy/inner_val_accuracy_reference",
                accuracy_reference,
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

            return loss

        if dataloader_idx == 1:
            (
                loss,
                auroc,
                accuracy_positives,
                accuracy_reference,
                accuracy_negatives_1,
                accuracy_negatives_2,
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
                "val/outer_val_auroc",
                auroc,
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
                "accuracy/outer_val_accuracy_reference",
                accuracy_reference,
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
            return self(self.target_transform(batch)), batch.mol_id
        else:
            return self(self.target_transform(batch))
