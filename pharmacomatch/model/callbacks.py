from lightning import LightningModule, Callback, Trainer
import torch
from torch_geometric import transforms as T


from ..dataset import (
    RandomNodeDeletion,
    FurthestSphericalSurfaceDisplacement,
    PositionsToGraphTransform,
)


class CurriculumLearningScheduler(Callback):
    """A callback to increase the graph size during training.

    The callback increases the upper bound on the graph size of the training data by one
    node if the validation loss does not improve for a certain number of epochs.

    Args:
        graph_size_at_start (int): Graph size at the beginning of training.
        num_epochs_before_increase (int): Number of epochs without improvement before
            increasing the graph size.
    """

    def __init__(
        self, graph_size_at_start: int, num_epochs_before_increase: int
    ) -> None:
        super().__init__()
        self.graph_size_at_start = graph_size_at_start
        self.num_epochs_before_increase = num_epochs_before_increase
        self.loss_at_reference_point = torch.inf
        self.minimum_improvement_threshold = 0.1
        self.counter = 0

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        trainer.datamodule.graph_size_upper_bound = self.graph_size_at_start
        trainer.datamodule.setup("fit")

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if trainer.current_epoch == 0:
            return

        current_loss = trainer.logged_metrics["val/inner_val_loss/dataloader_idx_0"]
        if (
            current_loss + self.minimum_improvement_threshold
        ) < self.loss_at_reference_point:
            self.loss_at_reference_point = current_loss
            self.counter = 0
        elif self.counter < self.num_epochs_before_increase:
            self.counter += 1
        else:
            trainer.datamodule.graph_size_upper_bound += 1
            trainer.datamodule.setup("fit")
            self.loss_at_reference_point = torch.inf
            self.counter = 0
            print(
                f"Graph size increased to {trainer.datamodule.graph_size_upper_bound} at epoch {trainer.current_epoch}"
            )


class ValidationDataTransformSetter(Callback):
    """A callback to set/change parameters of the validation data transform.

    This callback is used in the self-similarity experiment for perception of 3D
    positional features to alter the parameters of the validation data transform.
    It further sets the displacement mode to sphere surface sampling.

    Args:
        node_masking (float): Percentage of nodes to be deleted.
        radius (float): Displacement radius for the node positions.
        node_to_keep_lower_bound (int, optional): Minimum number of nodes that shall
            remain after random node deletion. Defaults to None.
    """

    def __init__(
        self, node_masking: float, radius: float, node_to_keep_lower_bound: int = None
    ) -> None:
        super().__init__()
        self.node_masking = node_masking
        self.radius = radius
        self.node_to_keep_lower_bound = node_to_keep_lower_bound

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if self.node_masking == 0 or self.node_masking is None:
            pl_module.target_transform = T.Compose(
                [
                    FurthestSphericalSurfaceDisplacement(self.radius),
                    PositionsToGraphTransform(),
                ]
            )
        else:
            pl_module.target_transform = T.Compose(
                [
                    RandomNodeDeletion(self.node_to_keep_lower_bound),
                    FurthestSphericalSurfaceDisplacement(self.radius),
                    PositionsToGraphTransform(),
                ]
            )
