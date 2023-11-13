from lightning import LightningDataModule
from torch_geometric import transforms as T
from torch_geometric.loader import DataLoader

from .dataset_transforms import DistanceOHE, DistanceRDF
from .pharmacophore_dataset import PharmacophoreDataset, VirtualScreeningDataset


class PharmacophoreDataModule(LightningDataModule):
    def __init__(self, preprocessing_data_dir, virtual_screening_data_dir, batch_size=None):
        super(PharmacophoreDataModule, self).__init__()
        self.preprocessing_data_dir = preprocessing_data_dir
        self.virtual_screening_data_dir = virtual_screening_data_dir
        self.batch_size = batch_size
        self.transform = None

    def setup(self, stage: str):
        if stage == "fit":
            preprocessing_data = PharmacophoreDataset(
                self.preprocessing_data_dir, path_number=0, transform=self.transform
            ).shuffle()
            print(f"Number of training graphs: {len(preprocessing_data)}")
            self.params = preprocessing_data.get_params()
            num_samples = len(preprocessing_data)
            self.train_data, self.val_data = (
                preprocessing_data[: (int)(num_samples * 0.9)],
                preprocessing_data[(int)(num_samples * 0.9) :],
            )

        if stage == 'virtual_screening':
            self.query = VirtualScreeningDataset(self.virtual_screening_data_dir, path_type='query', transform=self.transform)
            self.actives = VirtualScreeningDataset(self.virtual_screening_data_dir, path_type='active', transform=self.transform)
            self.inactives = VirtualScreeningDataset(self.virtual_screening_data_dir, path_type='inactive', transform=self.transform)
            print(f"Number of active graphs: {len(self.actives)}")
            print(f"Number of inactive graphs: {len(self.inactives)}")

    def train_dataloader(self):
        if self.batch_size == None:
            return DataLoader(self.train_data, batch_size=len(self.train_data))
        else:
            return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        if self.batch_size == None:
            return DataLoader(self.val_data, batch_size=len(self.val_data))
        else:
            return DataLoader(self.val_data, batch_size=self.batch_size)

    def query_dataloader(self):
        if self.batch_size == None:
            return DataLoader(self.query, batch_size=len(self.test_data))
        else:
            return DataLoader(self.query, batch_size=self.batch_size)
        
    def actives_dataloader(self):
        if self.batch_size == None:
            return DataLoader(self.actives, batch_size=len(self.test_data))
        else:
            return DataLoader(self.actives, batch_size=self.batch_size)
        
    def inactives_dataloader(self):
        if self.batch_size == None:
            return DataLoader(self.inactives, batch_size=len(self.test_data))
        else:
            return DataLoader(self.inactives, batch_size=self.batch_size)

    # This is needed as soon as I want to download the data from a repository.
    # Since I save it on disk, this hook is currently not needed.
    # def prepare_data(self) -> None:
    #    return super().prepare_data()
