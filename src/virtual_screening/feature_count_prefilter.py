import torch
from torch_geometric.nn import global_add_pool


class FeatureCountPrefilter:
    def __init__(self, vs_datamodule) -> None:
        self.vs_datamodule = vs_datamodule

    def get_actives_mask(self):
        query_count = self.get_feature_count(self.vs_datamodule.query_dataloader())
        actives_count = self.get_feature_count(self.vs_datamodule.actives_dataloader())
        actives_count -= query_count
        mask = torch.min(actives_count, dim=1).values
        mask = mask >= 0
        return mask
    
    def get_inactives_mask(self):
        query_count = self.get_feature_count(self.vs_datamodule.query_dataloader())
        inactives_count = self.get_feature_count(self.vs_datamodule.inactives_dataloader())
        inactives_count -= query_count
        mask = torch.min(inactives_count, dim=1).values
        mask = mask >= 0
        return mask

    def get_feature_count(self, dataloader):
        return torch.cat([global_add_pool(batch.x, batch.batch) for batch in iter(dataloader)])


