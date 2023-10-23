from lightning import LightningDataModule
from torch_geometric import transforms as T
from torch_geometric.loader import DataLoader

from .dataset_transforms import DistanceOHE, DistanceRDF
from .pharmacophore_dataset import PharmacophoreDataset


class PharmacophoreDataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size=None):
        super(PharmacophoreDataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        # self.transform = None
        self.transform = T.Compose(
            [T.KNNGraph(k=100), T.ToUndirected(), T.Distance(norm=False), DistanceRDF()]
        )

    def setup(self, stage: str):
        if stage == "fit":
            data_full = PharmacophoreDataset(
                self.data_dir, train=True, transform=self.transform
            ).shuffle()
            print(f"Number of training graphs: {len(data_full)}")
            self.params = data_full.get_params()
            num_samples = len(data_full)
            self.train_data, self.val_data = (
                data_full[: (int)(num_samples * 0.9)],
                data_full[(int)(num_samples * 0.9) :],
            )

        # There is no explicit test data, yet
        # if stage == "test":
        #     self.test_data = PharmacophoreDataset(
        #         self.data_dir, train=False, pre_transform=self.transform
        #     )
        #     print(f"Number of test graphs: {len(self.test_data)}")

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

    def test_dataloader(self):
        if self.batch_size == None:
            return DataLoader(self.test_data, batch_size=len(self.test_data))
        else:
            return DataLoader(self.test_data, batch_size=self.batch_size)

    # This is needed as soon as I download the data.
    # Since I save it on disk, this hook is currently not needed
    # def prepare_data(self) -> None:
    #    return super().prepare_data()
