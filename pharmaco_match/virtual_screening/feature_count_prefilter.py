import torch
from torch import Tensor
from torch_geometric.nn import global_add_pool
from torch_geometric.loader import DataLoader

from dataset import VirtualScreeningDataModule


class FeatureCountPrefilter:
    def __init__(self, vs_datamodule: VirtualScreeningDataModule) -> None:
        self.vs_datamodule = vs_datamodule

    def get_actives_mask(self, query_idx: int = 0) -> Tensor:
        query_count = self.get_feature_count(self.vs_datamodule.query_dataloader())
        self.actives_count = self.get_feature_count(
            self.vs_datamodule.actives_dataloader()
        )
        actives_count = self.actives_count - query_count[query_idx].reshape(1, -1)
        mask = torch.min(actives_count, dim=1).values
        mask = mask >= 0
        # mask = torch.norm(actives_count, p=1, dim=1) == 0
        return mask

    def get_inactives_mask(self, query_idx: int = 0) -> Tensor:
        query_count = self.get_feature_count(self.vs_datamodule.query_dataloader())
        self.inactives_count = self.get_feature_count(
            self.vs_datamodule.inactives_dataloader()
        )
        inactives_count = self.inactives_count - query_count[query_idx].reshape(1, -1)
        mask = torch.min(inactives_count, dim=1).values
        mask = mask >= 0
        # mask = torch.norm(inactives_count, p=1, dim=1) == 0
        return mask

    def get_feature_count(self, dataloader: DataLoader) -> Tensor:
        return torch.cat(
            [global_add_pool(batch.x, batch.batch) for batch in iter(dataloader)]
        )
