import torch

from .feature_count_prefilter import FeatureCountPrefilter


class VirtualScreener:
    def __init__(self, embedder, query_idx=0) -> None:
        self.embedder = embedder
        self.val_embeddings = self.embedder.get_val_embeddings()
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

        # calculate cosine similarity of (in)actives w.r.t. to query
        self.active_query_similarity = torch.mm(
            self.query_embedding[query_idx].reshape(1, -1), self.active_embeddings.T
        ).flatten()
        self.inactive_query_similarity = torch.mm(
            self.query_embedding[query_idx].reshape(1, -1), self.inactive_embeddings.T
        ).flatten()

        # pick most similar conformation per compound
        self.active_mask = self._create_mask(
            self.active_query_similarity, self.active_mol_ids
        )
        self.inactive_mask = self._create_mask(
            self.inactive_query_similarity, self.inactive_mol_ids
        )
        self.top_active_similarity = self.active_query_similarity[self.active_mask]
        self.top_inactive_similarity = self.inactive_query_similarity[
            self.inactive_mask
        ]
        self.top_active_embeddings = self.active_embeddings[self.active_mask]
        self.top_inactive_embeddings = self.inactive_embeddings[self.inactive_mask]

        # Map similarity [-1, 1] --> [0, 1] and print AUC statistics
        self.y_pred = torch.cat(
            (self.top_active_similarity, self.top_inactive_similarity)
        )
        self.y_pred = (self.y_pred + 1) / 2
        self.y_true = torch.cat(
            (
                torch.ones(len(self.top_active_similarity), dtype=torch.int),
                torch.zeros(len(self.top_inactive_similarity), dtype=torch.int),
            )
        )

    def _create_mask(self, similarities, mol_ids):
        splits = [similarities[mol_ids == i] for i in range(max(mol_ids + 1))]
        split_argmax = [torch.argmax(split) for split in splits]
        mask = [torch.zeros(split.shape) for split in splits]
        for idx, mask_split in zip(split_argmax, mask):
            mask_split[idx] = 1
        mask = torch.cat(mask) != 0

        return mask
