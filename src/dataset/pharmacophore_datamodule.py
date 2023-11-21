from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from torch_geometric.loader import DataLoader

from .pharmacophore_dataset import PharmacophoreDataset, VirtualScreeningDataset
from .augmentation_module import AugmentationModule


class PharmacophoreDataModule(LightningDataModule):
    def __init__(self, preprocessing_data_dir, virtual_screening_data_dir, batch_size=None, small_set_size=None):
        super(PharmacophoreDataModule, self).__init__()
        self.preprocessing_data_dir = preprocessing_data_dir
        self.virtual_screening_data_dir = virtual_screening_data_dir
        self.batch_size = batch_size
        self.small_set_size = small_set_size
        #self.transform = AugmentationModule(train=True)
        #self.val_transform = AugmentationModule(train=False)

    def setup(self, stage: str):
        if stage == "fit":
            # I need to pass the val_transform to the val_set
            preprocessing_data = PharmacophoreDataset(
                self.preprocessing_data_dir, transform=None
            )
            if self.small_set_size:
                preprocessing_data = preprocessing_data[:self.small_set_size]
            print(f"Number of training graphs: {len(preprocessing_data)}")
            self.params = preprocessing_data.get_params()
            num_samples = len(preprocessing_data)
            self.train_data, self.val_data = (
                preprocessing_data[: (int)(num_samples * 0.9)],
                preprocessing_data[(int)(num_samples * 0.9) :],
            )
            #self.train_data.transform=self.transform
            #self.val_data.transform=self.val_transform

        #if stage == 'virtual_screening':
            self.query = VirtualScreeningDataset(self.virtual_screening_data_dir, path_type='query', transform=None)
            self.actives = VirtualScreeningDataset(self.virtual_screening_data_dir, path_type='active', transform=None)
            self.inactives = VirtualScreeningDataset(self.virtual_screening_data_dir, path_type='inactive', transform=None)
            print(f"Number of active graphs: {len(self.actives)}")
            print(f"Number of inactive graphs: {len(self.inactives)}")

    def train_dataloader(self):
        if self.batch_size == None:
            return DataLoader(self.train_data, batch_size=len(self.train_data), shuffle=True, drop_last=True)
        else:
            return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, drop_last=True)
        
    def val_dataloader(self):
        return [self.create_val_dataloader()] + self.vs_dataloader()
        
    def create_val_dataloader(self):
        if self.batch_size == None:
            return DataLoader(self.val_data, batch_size=len(self.val_data), shuffle=False, drop_last=True)
        else:
            return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, drop_last=True)

    def vs_dataloader(self):
        return [self.query_dataloader(), self.actives_dataloader(), self.inactives_dataloader()]

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
