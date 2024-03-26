from lightning import LightningDataModule
from torch_geometric.loader import DataLoader
import torch

from .pharmacophore_dataset import PharmacophoreDataset, VirtualScreeningDataset


class PharmacophoreDataModule(LightningDataModule):
    def __init__(
        self,
        preprocessing_data_dir: str,
        virtual_screening_data_dir: str,
        batch_size: int = None,
        small_set_size: int = None,
        graph_size_upper_bound: int = None,
    ) -> None:
        super(PharmacophoreDataModule, self).__init__()
        self.preprocessing_data_dir = preprocessing_data_dir
        self.virtual_screening_data_dir = virtual_screening_data_dir
        self.batch_size = batch_size
        self.small_set_size = small_set_size
        self.graph_size_upper_bound = graph_size_upper_bound

    def setup(self, stage: str = "fit") -> None:
        if stage == "fit":
            preprocessing_data = PharmacophoreDataset(
                self.preprocessing_data_dir, transform=None
            )

            if self.graph_size_upper_bound:
                # idx = torch.tensor(
                #     [
                #         graph.num_ph4_features <= self.graph_size_upper_bound
                #         for graph in preprocessing_data
                #     ]
                # )
                idx = preprocessing_data.num_ph4_features <= self.graph_size_upper_bound
                preprocessing_data = preprocessing_data.copy(idx)

            if self.small_set_size < len(preprocessing_data):
                preprocessing_data = preprocessing_data[: self.small_set_size]

            print(f"Number of training graphs: {len(preprocessing_data)}")
            num_samples = len(preprocessing_data)
            self.train_data, self.val_data = (
                preprocessing_data[: (int)(num_samples * 0.9)],
                preprocessing_data[(int)(num_samples * 0.9) :],
            )

            self.query = VirtualScreeningDataset(
                self.virtual_screening_data_dir, path_type="query", transform=None
            )
            self.actives = VirtualScreeningDataset(
                self.virtual_screening_data_dir, path_type="active", transform=None
            )
            self.inactives = VirtualScreeningDataset(
                self.virtual_screening_data_dir, path_type="inactive", transform=None
            )
            print(f"Number of active graphs: {len(self.actives)}")
            print(f"Number of inactive graphs: {len(self.inactives)}")

    def train_dataloader(self) -> DataLoader:
        if self.batch_size is None:
            return DataLoader(
                self.train_data,
                batch_size=len(self.train_data),
                shuffle=True,
                drop_last=True,
            )
        else:
            return DataLoader(
                self.train_data,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
            )

    def val_dataloader(self) -> list[DataLoader]:
        return self.create_val_dataloader()
        # return [self.create_val_dataloader()] + self.vs_dataloader()

    def create_val_dataloader(self) -> DataLoader:
        if self.batch_size is None:
            return DataLoader(
                self.val_data,
                batch_size=len(self.val_data),
                shuffle=False,
                drop_last=True,
            )
        else:
            return DataLoader(
                self.val_data, batch_size=self.batch_size, shuffle=False, drop_last=True
            )

    def vs_dataloader(self) -> list[DataLoader]:
        return [
            self.query_dataloader(),
            self.actives_dataloader(),
            self.inactives_dataloader(),
        ]

    def query_dataloader(self) -> DataLoader:
        if self.batch_size is None:
            return DataLoader(self.query, batch_size=len(self.query))
        else:
            return DataLoader(self.query, batch_size=self.batch_size)

    def actives_dataloader(self) -> DataLoader:
        if self.batch_size is None:
            return DataLoader(self.actives, batch_size=len(self.actives))
        else:
            return DataLoader(self.actives, batch_size=self.batch_size)

    def inactives_dataloader(self) -> DataLoader:
        if self.batch_size is None:
            return DataLoader(self.inactives, batch_size=len(self.inactives))
        else:
            return DataLoader(self.inactives, batch_size=self.batch_size)
