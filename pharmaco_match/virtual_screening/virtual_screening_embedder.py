import torch
from torch import Tensor
from lightning import LightningModule, LightningDataModule, Trainer


class VirtualScreeningEmbedder:
    """A class for embedding the virtual screening dataset.

    This class creates vector embeddings for the query, active, and inactive ligands
    of a virtual screening dataset using a pretrained PharmacoMatch model.

    Args:
        model (LightningModule): the trained PharmacoMatch model.
        vs_datamodule (LightningDataModule): the virtual screening dataset.
        trainer (Trainer): Lightning trainer for inference.
    """

    def __init__(
        self,
        model: LightningModule,
        vs_datamodule: LightningDataModule,
        trainer: Trainer,
    ) -> None:
        self.model = model
        self.vs_datamodule = vs_datamodule
        self.trainer = trainer

    def get_query_embeddings(self) -> Tensor:
        return self._assemble(
            self.trainer.predict(
                model=self.model, dataloaders=self.vs_datamodule.query_dataloader()
            )
        )

    def get_active_embeddings(self) -> Tensor:
        return self._assemble(
            self.trainer.predict(
                model=self.model, dataloaders=self.vs_datamodule.actives_dataloader()
            )
        )

    def get_inactive_embeddings(self) -> Tensor:
        return self._assemble(
            self.trainer.predict(
                model=self.model, dataloaders=self.vs_datamodule.inactives_dataloader()
            )
        )

    def _assemble(self, prediction_output: list[tuple[Tensor]]) -> Tensor:
        predictions = []
        mol_ids = []
        for output in prediction_output:
            prediction, mol_id = output
            predictions.append(prediction)
            mol_ids.append(mol_id)

        return torch.vstack(predictions), torch.hstack(mol_ids)

    def penalty(self, query: Tensor, target: Tensor) -> Tensor:
        return self.model.penalty(query, target)
