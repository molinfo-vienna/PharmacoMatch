import torch
from torch import Tensor
from torch_geometric.nn import global_max_pool

from .feature_count_prefilter import FeatureCountPrefilter
from .virtual_screening_embedder import VirtualScreeningEmbedder


class VirtualScreener:
    def __init__(self, embedder: VirtualScreeningEmbedder, query_idx: int = 0) -> None:
        self.embedder = embedder
        # self.val_embeddings = self.embedder.get_val_embeddings()
        self.query_embedding, _ = self.embedder.get_query_embeddings()
        (
            self.active_embeddings,
            self.active_mol_ids,
        ) = self.embedder.get_active_embeddings()
        (
            self.inactive_embeddings,
            self.inactive_mol_ids,
        ) = self.embedder.get_inactive_embeddings()

        # create mask for feature count prefilter
        self.prefilter = FeatureCountPrefilter(embedder.vs_datamodule)
        self.actives_prefilter_mask = self.prefilter.get_actives_mask(query_idx)
        self.inactives_prefilter_mask = self.prefilter.get_inactives_mask(query_idx)

        # calculate decision function of (in)actives w.r.t. to query

        self.active_query_match = torch.sum(
            torch.max(
                torch.zeros_like(self.active_embeddings),
                self.query_embedding.expand(self.active_embeddings.shape)
                - self.active_embeddings,
            )
            ** 2,
            dim=1,
        )
        self.inactive_query_match = torch.sum(
            torch.max(
                torch.zeros_like(self.inactive_embeddings),
                self.query_embedding.expand(self.inactive_embeddings.shape)
                - self.inactive_embeddings,
            )
            ** 2,
            dim=1,
        )
        # self.active_query_match = torch.max(
        #     self.query_embedding.expand(self.active_embeddings.shape)
        #     - self.active_embeddings,
        #     dim=1,
        # ).values
        # self.inactive_query_match = torch.max(
        #     self.query_embedding.expand(self.inactive_embeddings.shape)
        #     - self.inactive_embeddings,
        #     dim=1,
        # ).values

    #     self.active_match = global_max_pool(self.active_query_match, self.active_mol_ids)
    #     self.inactive_match = global_max_pool(self.inactive_query_match, self.inactive_mol_ids)

    #     # pick most similar conformation per compound
    #     self.active_mask = self._create_mask(
    #         self.active_query_match, self.active_mol_ids
    #     )
    #     self.inactive_mask = self._create_mask(
    #         self.inactive_query_match, self.inactive_mol_ids
    #     )
    #     self.top_active_similarity = self.active_query_match[self.active_mask]
    #     self.top_inactive_similarity = self.inactive_query_match[
    #         self.inactive_mask
    #     ]
    #     self.top_active_embeddings = self.active_embeddings[self.active_mask]
    #     self.top_inactive_embeddings = self.inactive_embeddings[self.inactive_mask]

    #     # Map similarity [-1, 1] --> [0, 1] and print AUC statistics
    #     self.y_pred = torch.cat(
    #         (self.top_active_similarity, self.top_inactive_similarity)
    #     )
    #     self.y_pred = (self.y_pred + 1) / 2
    #     self.y_true = torch.cat(
    #         (
    #             torch.ones(len(self.top_active_similarity), dtype=torch.int),
    #             torch.zeros(len(self.top_inactive_similarity), dtype=torch.int),
    #         )
    #     )

    # def _create_mask(self, similarities: Tensor, mol_ids: Tensor) -> Tensor:
    #     splits = [similarities[mol_ids == i] for i in range(max(mol_ids + 1))]
    #     split_argmax = [torch.argmax(split) for split in splits]
    #     mask = [torch.zeros(split.shape) for split in splits]
    #     for idx, mask_split in zip(split_argmax, mask):
    #         mask_split[idx] = 1
    #     mask = torch.cat(mask) != 0

    #     return mask
