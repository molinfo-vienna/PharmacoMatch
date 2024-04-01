from typing import Generator, Union
import random

from flash.core.optimizers import LARS, LinearWarmupCosineAnnealingLR
from lightning import LightningModule
import torch
from torch.optim import Optimizer, AdamW, Adam
from torch import Tensor
from torch_geometric.data import Data
from torchmetrics import ROC, MatthewsCorrCoef

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
            radius=3.0,
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

        # validation embeddings
        # self.val_embeddings = []
        self.mcc = MatthewsCorrCoef(num_classes=2)

    def forward(self, data: Data) -> Tensor:
        # Optional: Visualization of the input pharmacophore
        # visualize_pharm(
        #     [data[0].clone(), self.transform(data[0].clone()),
        #      self.transform(data[0].clone())]
        # )

        representation = self.encoder(data)
        embedding = self.projection_head(representation, data.num_ph4_features)

        return embedding

    def setup(self, stage: str) -> None:
        global_batch_size = self.trainer.world_size * self.hparams.batch_size
        self.train_iters_per_epoch = self.hparams.num_samples // global_batch_size

        for param in self.encoder.parameters():
            param.requires_grad = False

    # def exclude_from_wt_decay(
    #     self,
    #     named_params: Generator,
    #     weight_decay: float,
    #     skip_list: list[str] = ["bias", "bn"],
    # ) -> list[dict]:
    #     params = []
    #     excluded_params = []

    #     for name, param in named_params:
    #         if not param.requires_grad:
    #             continue
    #         elif any(layer_name in name for layer_name in skip_list):
    #             excluded_params.append(param)
    #         else:
    #             params.append(param)

    #     return [
    #         {"params": params, "weight_decay": weight_decay},
    #         {"params": excluded_params, "weight_decay": 0.0},
    #     ]

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict]]:
        # # TRICK 1 (Use lars + filter weights)
        # # exclude certain parameters
        # parameters = self.exclude_from_wt_decay(
        #     self.named_parameters(), weight_decay=self.hparams.weight_decay
        # )

        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            # momentum=self.hparams.momentum,
        )

        # # Trick 2 (after each step)
        # warmup_epochs = self.hparams.warmup_epochs * self.train_iters_per_epoch
        # max_epochs = self.trainer.max_epochs * self.train_iters_per_epoch

        # linear_warmup_cosine_decay = LinearWarmupCosineAnnealingLR(
        #     optimizer,
        #     warmup_epochs=warmup_epochs,
        #     max_epochs=max_epochs,
        #     warmup_start_lr=0,
        #     eta_min=0,
        # )

        # scheduler = {
        #     "scheduler": linear_warmup_cosine_decay,
        #     "interval": "step",
        #     "frequency": 1,
        # }

        return [optimizer]  # , [scheduler]

    # def nt_xent_loss(self, out_1: Tensor, out_2: Tensor) -> tuple[Tensor, Tensor]:
    #     out = torch.cat([out_1, out_2], dim=0)
    #     n_samples = len(out)

    #     # Full similarity matrix
    #     cov = torch.mm(out, out.t().contiguous())
    #     sim = torch.exp(cov / self.hparams.temperature)

    #     mask = ~torch.eye(n_samples, device=sim.device).bool()
    #     neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)

    #     # Positive similarity
    #     pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.hparams.temperature)
    #     pos = torch.cat([pos, pos], dim=0)

    #     loss = -torch.log(pos / neg).mean()

    #     # calc top1-accuracy
    #     mask = torch.eye(cov.shape[0], dtype=torch.bool, device=cov.device)
    #     cov.masked_fill_(mask, -1e4)
    #     mask = mask.roll(shifts=cov.shape[0] // 2, dims=0)

    #     comb_sim = torch.cat(
    #         [
    #             cov[mask][:, None],
    #             cov.masked_fill(mask, -1e4),
    #         ],  # First position positive example
    #         dim=-1,
    #     )
    #     sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
    #     accuracy = (sim_argsort == 0).float().mean()

    #     return loss, accuracy

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

        return 4 * torch.sum(positives) + torch.sum(negatives) + regularization_term

    def shared_step(self, batch: Data, batch_idx: int) -> tuple[Tensor, Tensor]:
        # queries_pos = self(self.transform(batch))
        # targets_pos = self(self.val_transform(batch))
        # targets_neg = self(self.transform(batch))

        # return self.loss(
        #     queries_pos, targets_pos, targets_neg
        # )

        queries = self.transform(batch)
        targets = self.val_transform(batch)
        negative_targets = self.negative_target_transform(batch)

        return self.loss(queries, targets, negative_targets)

    def training_step(self, batch: Data, batch_idx: int) -> Tensor:
        loss = self.shared_step(batch, batch_idx)
        self.log(
            "train/train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )
        # self.log(
        #     "train/train_accuracy",
        #     accuracy,
        #     prog_bar=True,
        #     on_step=False,
        #     on_epoch=True,
        #     batch_size=len(batch),
        # )

        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_embeddings = []

    def validation_step(
        self, batch: Data, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        # val loss calculation
        if dataloader_idx == 0:
            inner_val_loss = self.shared_step(batch, batch_idx)
            self.log(
                "val/inner_val_loss",
                inner_val_loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=len(batch),
            )

            return inner_val_loss

        if dataloader_idx == 1:
            outer_val_loss = self.shared_step(batch, batch_idx)
            self.log(
                "val/outer_val_loss",
                outer_val_loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=len(batch),
            )
            # self.log(
            #     "val/val_accuracy",
            #     val_accuracy,
            #     prog_bar=True,
            #     on_step=False,
            #     on_epoch=True,
            #     batch_size=len(batch),
            # )

            # rankme criterion
            # self.val_embeddings.append(self(self.val_transform(batch)))

            return outer_val_loss

    # def on_validation_epoch_end(self) -> None:
    #     epsilon = 1e-7
    #     embeddings = torch.vstack(self.val_embeddings)
    #     _, singular_values, _ = torch.svd(embeddings)
    #     singular_values /= torch.sum(singular_values)
    #     singular_values += epsilon
    #     rank_me = torch.exp(-torch.sum(singular_values * torch.log(singular_values)))
    #     self.log(
    #         "val/rank_me",
    #         rank_me,
    #         prog_bar=True,
    #         on_step=False,
    #         on_epoch=True,
    #     )

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
