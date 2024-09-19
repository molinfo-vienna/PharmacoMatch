from typing import Union

from lightning import LightningModule
import torch
from torch.optim import Optimizer, Adam
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric import transforms as T
from torchmetrics.classification import BinaryAUROC

from .encoder import NNConvEncoder
from .projector import ProjectionPhectorMatch
from dataset import (
    RandomSphericalNoise,
    FurthestSphericalSurfaceDisplacement,
    PositionsToGraphTransform,
    TwiceRandomNodeDeletionWithoutOverlap,
)


class PharmacoMatch(LightningModule):
    """Implementation of the PharmacoMatch model

    This class implements the training logic for the PharmacoMatch model. The model
    consists of a GNN encoder and a projector, which are trained one an order embedding
    loss to encode query-target relationships. Queries and targets are created on the
    fly during training by applying random transformations to the input data.
    """

    def __init__(self, **params) -> None:
        super(PharmacoMatch, self).__init__()
        self.save_hyperparameters()

        # Data transforms for positive and negative pair creation
        self.twice_node_deletion = TwiceRandomNodeDeletionWithoutOverlap()
        self.target_transform = T.Compose([PositionsToGraphTransform()])
        self.query_transform = T.Compose(
            [RandomSphericalNoise(self.hparams.radius), PositionsToGraphTransform()]
        )
        self.reference_transform = T.Compose(
            [
                RandomSphericalNoise(self.hparams.radius_negative),
                PositionsToGraphTransform(),
            ]
        )
        self.negative_query_transform = T.Compose(
            [
                FurthestSphericalSurfaceDisplacement(self.hparams.radius_negative),
                PositionsToGraphTransform(),
            ]
        )

        # Encoder and projector
        if self.hparams.encoder == "GIN":
            self.encoder = GINEncoder(
                input_dim=self.hparams.num_node_features,
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

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict]]:
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
        )

        return [optimizer]

    def penalty(self, query: Tensor, target: Tensor) -> Tensor:
        """Order embedding penalty

        Args:
            query (Tensor): Query embedding vector(s). During trining, this will be a
                n x m tensor, where n is the batch size and m is the embedding
                dimension. During inference, this will be the 1 x m embedding of the
                query.
            target (Tensor): Target embedding vectors, a n x m tensor, where n is the
                batch size and m is the embedding dimension.

        Returns:
            Tensor: 1 x n tensor, holding the order embedding penalty for each of the
            targets.
        """
        diff = query - target
        diff.clamp_(min=0)
        diff.pow_(2)

        return diff.sum(dim=1)

    def loss(
        self,
        queries: Tensor,
        reference_queries: Tensor,
        negative_queries: Tensor,
        targets: Tensor,
        negative_targets: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Order embedding loss

        This function implements the order embedding loss, as described in:
        Ying, R.; Lou, Z.; You, J.; Wen, C.; Canedo, A.; and Leskovec, J. 2020.
        Neural Subgraph Matching. arXiv:2007.03092

        This function calculates the loss for several combinations of query-target
        pairs and further calculates accuracy and AUROC metrics for model monitoring.

        Args:
            queries (Tensor): Batch of query pharmacophore graphs.
            reference_queries (Tensor): Batch of reference query pharmacophore graphs.
                Reference queries are always generated with the same radius for node
                displacement. They are used to calculate metrics for model performance,
                but do not contribute to the loss calculation.
            negative_queries (Tensor): Batch of negative query pharmacophore graphs.
                Negative queries provide hard examples that shall not match the targets.
            targets (Tensor): Batch of target pharmacophore graphs. These graphs
                were generated without augmentation.
            negative_targets (Tensor): Batch of negative target pharmacophore graphs.
                Negative targets provide examples that shall not match the queries.

        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]: Order embedding loss
                and metrics for model performance.
        """
        # Extract the embeddings
        queries = self(queries)
        reference_queries = self(reference_queries)
        negative_queries = self(negative_queries)
        targets = self(targets)
        negative_targets = self(negative_targets)
        batch_size = len(queries)

        # Positive loss term
        positives = self.penalty(queries, targets)
        positives_reference = self.penalty(reference_queries, targets)

        # Negative loss term - fine-grained negatives by displacement of tolerance sphere radius
        negatives_fine = self.penalty(negative_queries, targets)

        # Semi-coarse-grained negatives by mapping queries to partially matching targets
        negatives_2 = self.penalty(reference_queries, negative_targets)

        # Coarse-grained negatives by mapping pairs that should not match
        negatives_3 = self.penalty(
            queries, torch.roll(targets, batch_size // 2, dims=0)
        )
        negatives_4 = self.penalty(
            targets, torch.roll(targets, batch_size // 2, dims=0)
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

    def shared_step(self, batch: Data, batch_idx: int) -> Tensor:
        targets = self.target_transform(batch.clone())
        reduced_batch1, reduced_batch2 = self.twice_node_deletion(batch.clone())
        queries = self.query_transform(reduced_batch1.clone())
        reference_queries = self.reference_transform(reduced_batch1.clone())
        negative_targets = self.target_transform(reduced_batch2.clone())
        negative_queries = self.negative_query_transform(reduced_batch1.clone())

        return self.loss(
            queries, reference_queries, negative_queries, targets, negative_targets
        )

    def training_step(self, batch: Data, batch_idx: int) -> Tensor:
        """Training step and logging"""
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
        """Validation step and logging. This function differentiates between an inner
        (idx 0) and an outer validation dataloader (idx 1)."""
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
        """Initialization of the logger"""
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
        """Prediction step for inference"""
        if "mol_id" in batch.keys:
            return self(self.target_transform(batch.clone())), batch.mol_id
        else:
            return self(self.target_transform(batch.clone()))
