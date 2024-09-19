from dataclasses import dataclass

from lightning import LightningDataModule
import pandas as pd
from torch_geometric.loader import DataLoader

from .pharmacophore_dataset import PharmacophoreDataset, VirtualScreeningDataset


class PharmacophoreDataModule(LightningDataModule):
    """Implementation of a LightningDataModule for the pharmacophore dataset.

    The data module provides data loaders for model training and validation. The module
    offers two validation sets, an inner validation set and an outer validation set.
    Curriculum learning is applied by training only on the subset of graphs with a
    number of nodes that is bounded by the graph_size_upper_bound parameter. The inner
    validation data is a split of this subset, the outer validation data is a split of
    the complete dataset. Tracking model performance on the inner validation data
    indicates when to increase the graph_size_upper_bound parameter, tracking the outer
    data reflects the overall model performance.

    Args:
        training_data_dir (str): Path to the location of the unlabeled
            pharmacophore dataset for model training.
        batch_size (int, optional): Batch size of the data loaders. If None, the
            dataloader works in full batch mode. Defaults to None.
        small_set_size (int, optional): Upper bound on the number of graphs in the
            training data. If None, training is performed on the full dataset. Defaults
            to None.
        graph_size_upper_bound (int, optional): Upper bound on the number of nodes per
            pharmacophore graph to be included in the training data. If None, there
            number of nodes is unbounded. This parameter is used by the
            CurriculumLearningScheduler class. Defaults to None.
    """

    def __init__(
        self,
        training_data_dir: str,
        batch_size: int = None,
        small_set_size: int = -1,
        graph_size_upper_bound: int = None,
    ) -> None:
        super(PharmacophoreDataModule, self).__init__()
        self.training_data_dir = training_data_dir
        self.batch_size = batch_size
        self.small_set_size = small_set_size
        self.graph_size_upper_bound = graph_size_upper_bound
        self.num_workers = 8

    def setup(self, stage: str = "fit") -> None:
        if stage == "fit":
            full_data = PharmacophoreDataset(self.training_data_dir, transform=None)

            inner_data, self.outer_val_data = (
                full_data[: (int)(len(full_data) * 0.98)],
                full_data[(int)(len(full_data) * 0.98) :],
            )

            if self.graph_size_upper_bound:
                idx = inner_data.num_ph4_features <= self.graph_size_upper_bound
                inner_data = inner_data.copy(idx)

            if self.small_set_size < len(inner_data):
                inner_data = inner_data[: self.small_set_size]

            num_samples = len(inner_data)
            self.train_data, self.inner_val_data = (
                inner_data[: (int)(num_samples * 0.9)],
                inner_data[(int)(num_samples * 0.9) :],
            )

            print(f"Number of training graphs: {len(self.train_data)}")
            print(f"Number of inner validation graphs: {len(self.inner_val_data)}")
            print(f"Number of outer validation graphs: {len(self.outer_val_data)}")

    def train_dataloader(self) -> DataLoader:
        if self.batch_size is None:
            return DataLoader(
                self.train_data,
                batch_size=len(self.train_data),
                shuffle=True,
                drop_last=True,
                num_workers=self.num_workers,
                persistent_workers=True,
            )
        else:
            return DataLoader(
                self.train_data,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=self.num_workers,
                persistent_workers=True,
            )

    def val_dataloader(self) -> list[DataLoader]:
        return [
            self._create_val_dataloader(self.inner_val_data),
            self._create_val_dataloader(self.outer_val_data),
        ]

    def _create_val_dataloader(self, data) -> DataLoader:
        if self.batch_size is None:
            return DataLoader(
                data,
                batch_size=len(data),
                shuffle=False,
                drop_last=True,
                num_workers=self.num_workers,
                persistent_workers=True,
            )
        else:
            return DataLoader(
                data,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=self.num_workers,
                persistent_workers=True,
            )


@dataclass
class VirtualScreeningMetaData:
    active: pd.DataFrame
    inactive: pd.DataFrame
    query: pd.DataFrame


class VirtualScreeningDataModule(LightningDataModule):
    """Implementation of a LightningDataModule for the virtual screening dataset.

    A virtual screening benchmark dataset consists of three subsets: the query
    pharmacophore, active ligands, and inactive ligands. The query is stored in
    .pml-format, the actives and inactives are stored in .psd-format. The data module
    provides data loaders for each of these subsets.

    Args:
        virtual_screening_data_dir (str): Path to the location of the labeled virtual
            screening dataset for model evaluation.
        batch_size (int, optional): Batch size of the data loaders. If None, the
            dataloader works in full batch mode. Defaults to None.
    """

    def __init__(
        self,
        virtual_screening_data_dir: str,
        batch_size: int = None,
    ) -> None:
        super(VirtualScreeningDataModule, self).__init__()
        self.virtual_screening_data_dir = virtual_screening_data_dir
        self.batch_size = batch_size
        self.num_workers = 8

    def setup(self, stage: str = "screening") -> None:
        if stage == "screening":
            self.query = VirtualScreeningDataset(
                self.virtual_screening_data_dir, path_type="query", transform=None
            )
            self.actives = VirtualScreeningDataset(
                self.virtual_screening_data_dir, path_type="active", transform=None
            )
            self.inactives = VirtualScreeningDataset(
                self.virtual_screening_data_dir, path_type="inactive", transform=None
            )
            print(f"Number of query graphs: {len(self.query)}")
            print(f"Number of active graphs: {len(self.actives)}")
            print(f"Number of inactive graphs: {len(self.inactives)}")

            self.metadata = VirtualScreeningMetaData(
                self.actives.get_metadata(),
                self.inactives.get_metadata(),
                self.query.get_metadata(),
            )

    def vs_dataloader(self) -> list[DataLoader]:
        return [
            self.query_dataloader(),
            self.actives_dataloader(),
            self.inactives_dataloader(),
        ]

    def query_dataloader(self) -> DataLoader:
        if self.batch_size is None:
            return DataLoader(
                self.query,
                batch_size=len(self.query),
                num_workers=self.num_workers,
                persistent_workers=True,
            )
        else:
            return DataLoader(
                self.query,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                persistent_workers=True,
            )

    def actives_dataloader(self) -> DataLoader:
        if self.batch_size is None:
            return DataLoader(
                self.actives,
                batch_size=len(self.actives),
                num_workers=self.num_workers,
                persistent_workers=True,
            )
        else:
            return DataLoader(
                self.actives,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                persistent_workers=True,
            )

    def inactives_dataloader(self) -> DataLoader:
        if self.batch_size is None:
            return DataLoader(
                self.inactives,
                batch_size=len(self.inactives),
                num_workers=self.num_workers,
                persistent_workers=True,
            )
        else:
            return DataLoader(
                self.inactives,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                persistent_workers=True,
            )
