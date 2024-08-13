import time

import torch
from torch_geometric.nn import global_max_pool

from .feature_count_prefilter import FeatureCountPrefilter
from .virtual_screening_embedder import VirtualScreeningEmbedder


class VirtualScreener:
    def __init__(self, embedder: VirtualScreeningEmbedder, query_idx: int = 0) -> None:
        self.embedder = embedder
        start = time.time()
        self.query_embedding, _ = self.embedder.get_query_embeddings()
        (
            self.active_embeddings,
            self.active_mol_ids,
        ) = self.embedder.get_active_embeddings()
        (
            self.inactive_embeddings,
            self.inactive_mol_ids,
        ) = self.embedder.get_inactive_embeddings()
        ende = time.time()
        self.embedding_time = ende - start

        # create mask for feature count prefilter
        self.prefilter = FeatureCountPrefilter(embedder.vs_datamodule)
        self.actives_prefilter_mask = self.prefilter.get_actives_mask(query_idx)
        self.inactives_prefilter_mask = self.prefilter.get_inactives_mask(query_idx)
        self.prefilter_mask = torch.cat(
            (self.actives_prefilter_mask, self.inactives_prefilter_mask)
        )

        start = time.time()
        # calculate decision function of actives w.r.t. to query
        self.active_conformation_score = self.embedder.penalty(
            self.query_embedding, self.active_embeddings
        )
        # calculate decision function of inactives w.r.t. to query
        self.inactive_conformation_score = self.embedder.penalty(
            self.query_embedding, self.inactive_embeddings
        )
        end = time.time()
        self.matching_time = end - start

        self.active_ligand_score = -global_max_pool(
            -self.active_conformation_score, self.active_mol_ids
        )
        self.inactive_ligand_score = -global_max_pool(
            -self.inactive_conformation_score, self.inactive_mol_ids
        )

        self.mol_ids = torch.cat(
            (self.active_mol_ids, self.inactive_mol_ids + self.active_mol_ids[-1] + 1)
        )

        self.conformation_score = torch.cat(
            (self.active_conformation_score, self.inactive_conformation_score)
        )

        self.ligand_score = torch.cat(
            (self.active_ligand_score, self.inactive_ligand_score)
        )

        self.ligand_label = torch.cat(
            (
                torch.ones(self.active_mol_ids[-1] + 1),
                torch.zeros(self.inactive_mol_ids[-1] + 1),
            )
        )

        self.conformation_label = torch.cat(
            (
                torch.ones(len(self.active_mol_ids)),
                torch.zeros(len(self.inactive_mol_ids)),
            )
        )
