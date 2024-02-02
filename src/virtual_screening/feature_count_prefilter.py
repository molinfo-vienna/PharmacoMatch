import torch
from torch_geometric.nn import global_add_pool


class FeatureCountPrefilter:
    def __init__(self, vs_datamodule) -> None:
        self.vs_datamodule = vs_datamodule

    def get_actives_mask(self, query_idx=0):
        query_count = self.get_feature_count(self.vs_datamodule.query_dataloader())
        self.actives_count = self.get_feature_count(
            self.vs_datamodule.actives_dataloader()
        )
        actives_count = self.actives_count - query_count[query_idx].reshape(1, -1)
        # mask = torch.min(actives_count, dim=1).values
        # mask = mask >= 0
        mask = torch.norm(actives_count, p=1, dim=1) == 0
        return mask

    def get_inactives_mask(self, query_idx=0):
        query_count = self.get_feature_count(self.vs_datamodule.query_dataloader())
        self.inactives_count = self.get_feature_count(
            self.vs_datamodule.inactives_dataloader()
        )
        inactives_count = self.inactives_count - query_count[query_idx].reshape(1, -1)
        # mask = torch.min(inactives_count, dim=1).values
        # mask = mask >= 0
        mask = torch.norm(inactives_count, p=1, dim=1) == 0
        return mask

    def get_feature_count(self, dataloader):
        return torch.cat(
            [global_add_pool(batch.x, batch.batch) for batch in iter(dataloader)]
        )
