from typing import Any

from lightning import LightningModule, Callback, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import global_max_pool
from torchmetrics import AUROC

from dataset import AugmentationModule


class CurriculumLearningScheduler(Callback):
    def __init__(
        self, graph_size_at_start: int, num_epochs_before_increase: int
    ) -> None:
        super().__init__()
        self.graph_size_at_start = graph_size_at_start
        self.num_epochs_before_increase = num_epochs_before_increase

    def on_train_epoch_start(self, trainer, model):
        epoch = trainer.current_epoch
        if epoch % self.num_epochs_before_increase == 0:
            trainer.datamodule.graph_size_upper_bound = (
                epoch // self.num_epochs_before_increase + self.graph_size_at_start
            )
            trainer.datamodule.setup("fit")


class ValidationDataTransformSetter(Callback):
    def __init__(self, node_masking: float, radius: float) -> None:
        super().__init__()
        self.node_masking = node_masking
        self.radius = radius

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        pl_module.val_transform = AugmentationModule(
            train=True,
            node_masking=self.node_masking,
            sphere_surface_sampling=True,
            radius=self.radius,
        )


class VirtualScreeningCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.query = []
        self.actives = []
        self.inactives = []
        self.auroc = AUROC()

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        with torch.no_grad():
            pl_module.to(batch.x.device)
            if dataloader_idx == 1:
                self.query.append(pl_module.predict_step(batch, batch_idx))
            if dataloader_idx == 2:
                self.actives.append(pl_module.predict_step(batch, batch_idx))
            if dataloader_idx == 3:
                self.inactives.append(pl_module.predict_step(batch, batch_idx))

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        # encode pharmacophore data
        query, _ = self.assemble(self.query)
        actives, active_mol_ids = self.assemble(self.actives)
        inactives, inactive_mol_ids = self.assemble(self.inactives)

        # calculate similarity & predict activity w.r.t. to query
        active_similarity = F.cosine_similarity(query, actives)
        inactive_similarity = F.cosine_similarity(query, inactives)
        active_similarity = global_max_pool(active_similarity, active_mol_ids)
        inactive_similarity = global_max_pool(inactive_similarity, inactive_mol_ids)

        y_pred = torch.cat((active_similarity, inactive_similarity))
        y_pred = (y_pred + 1) / 2
        y_true = torch.cat(
            (
                torch.ones(
                    len(active_similarity), dtype=torch.int, device=y_pred.device
                ),
                torch.zeros(
                    len(inactive_similarity), dtype=torch.int, device=y_pred.device
                ),
            )
        )

        # calculate and log metric
        auroc = self.auroc(preds=y_pred, target=y_true)
        mean_active_similarity = torch.mean(active_similarity)
        mean_inactive_similarity = torch.mean(inactive_similarity)
        self.log(
            "vs/auroc",
            auroc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "vs/mean_active_similarity",
            mean_active_similarity,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "vs/mean_inactive_similarity",
            mean_inactive_similarity,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        # free up
        self.query = []
        self.actives = []
        self.inactives = []

    def assemble(
        self, prediction_output: list[tuple[Tensor, Tensor]]
    ) -> tuple[Tensor, Tensor]:
        predictions = []
        mol_ids = []
        for output in prediction_output:
            prediction, mol_id = output
            predictions.append(prediction)
            mol_ids.append(mol_id)

        return torch.vstack(predictions), torch.hstack(mol_ids)
