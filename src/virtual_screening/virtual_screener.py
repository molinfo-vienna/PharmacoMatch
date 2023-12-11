import torch


class VirtualScreener:
    def __init__(self, embedder) -> None:
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

        # calculate cosine similarity of (in)actives w.r.t. to query
        self.active_query_similarity = torch.mm(
            self.query_embedding, self.active_embeddings.T
        ).flatten()
        self.inactive_query_similarity = torch.mm(
            self.query_embedding, self.inactive_embeddings.T
        ).flatten()

        # pick most similar conformation per compound
        active_mask = self._create_mask(
            self.active_query_similarity, self.active_mol_ids
        )
        inactive_mask = self._create_mask(
            self.inactive_query_similarity, self.inactive_mol_ids
        )
        self.top_active_similarity = self.active_query_similarity[active_mask]
        self.top_inactive_similarity = self.inactive_query_similarity[inactive_mask]
        self.top_active_embeddings = self.active_embeddings[active_mask]
        self.top_inactive_embeddings = self.inactive_embeddings[inactive_mask]

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
