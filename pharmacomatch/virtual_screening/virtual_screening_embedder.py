import os, sys

import torch
from torch import Tensor
from lightning import LightningModule, LightningDataModule, Trainer
import numpy as np

import CDPL.Chem as Chem
import CDPL.Pharm as Pharm
import CDPL.Descr as Descr
import CDPL.Math as Math
import CDPL.Util as Util


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


class FingerprintEmbedder:
    def __init__(self, datamodule, fp_type="feature_count"):
        self.fp_type = fp_type
        self.vs_datamodule = datamodule
        self.query_file = os.path.join(
            datamodule.virtual_screening_data_dir, "raw", "query.pml"
        )
        self.actives_file = os.path.join(
            datamodule.virtual_screening_data_dir, "raw", "actives.psd"
        )
        self.inactives_file = os.path.join(
            datamodule.virtual_screening_data_dir, "raw", "inactives.psd"
        )
        if fp_type == "rdf":
            self.calculator = Descr.PharmacophoreRDFDescriptorCalculator()
            self.descriptor = Math.DVector(512)

    def get_query_embeddings(self) -> Tensor:
        reader = Pharm.PharmacophoreReader(self.query_file)
        ph4 = Pharm.BasicPharmacophore()
        while reader.read(ph4):
            try:
                fp = self.get_fingerprint(ph4)
            except Exception as e:
                sys.exit("Error: processing of pharmacophore failed: " + str(e))
        return fp, torch.tensor([0])

    def get_active_embeddings(self) -> Tensor:
        db_accessor = Pharm.PSDScreeningDBAccessor(self.actives_file)
        fps, mol_ids = self.calculate_fingerprints(db_accessor)
        return fps, mol_ids

    def get_inactive_embeddings(self) -> Tensor:
        db_accessor = Pharm.PSDScreeningDBAccessor(self.inactives_file)
        fps, mol_ids = self.calculate_fingerprints(db_accessor)
        return fps, mol_ids

    def penalty(self, query: Tensor, target: Tensor) -> Tensor:
        diff = query - target
        diff.clamp_(min=0)
        diff.pow_(2)

        return diff.sum(dim=1)

    def get_fingerprint(self, ph4):
        if self.fp_type == "feature_count":
            return self.get_feature_count(ph4)
        if self.fp_type == "rdf":
            return self.get_rdf(ph4)

    def get_rdf(self, ph4):
        self.calculator.calculate(ph4, self.descriptor)
        return torch.tensor(self.descriptor.toArray())

    def get_feature_count(self, ph4):
        x = torch.zeros(7)
        for i, feature in enumerate(ph4):
            x[Pharm.getType(feature) - 1] += 1
        return x

    def calculate_fingerprints(self, db_accessor):
        num_molecules = db_accessor.getNumMolecules()
        ph4 = Pharm.BasicPharmacophore()
        mol_ids = []
        fps = []
        for i in range(num_molecules):
            try:
                num_pharmacophores = db_accessor.getNumPharmacophores(i)
                for j in range(num_pharmacophores):
                    db_accessor.getPharmacophore(i, j, ph4)
                    fp = self.get_fingerprint(ph4)
                    mol_ids.append(i)
                    fps.append(fp)

            except Exception as e:
                sys.exit("Error: processing of pharmacophore failed: " + str(e))

        return torch.vstack(fps), torch.tensor(mol_ids)
