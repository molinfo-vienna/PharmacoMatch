import torch
from torch_geometric.nn import global_max_pool
from torch.nn.functional import cosine_similarity
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
import umap

from utils import *
from dataset import *
from model import *


class VirtualScreening:
    def __init__(self, model, trainer) -> None:
        self.model = model
        self.trainer = trainer

    def __call__(self, datamodule) -> None:
        y_true, y_pred = self.perform_screening(datamodule)
        self.plot_metrics(y_true, y_pred)

    def perform_screening(self, datamodule):
        # encode pharmacophore data
        query, _ = self.assemble(self.trainer.predict(
            model=self.model, dataloaders=datamodule.query_dataloader()))
        actives, active_mol_ids = self.assemble(self.trainer.predict(
            model=self.model, dataloaders=datamodule.actives_dataloader()))
        inactives, inactive_mol_ids = self.assemble(self.trainer.predict(
            model=self.model, dataloaders=datamodule.inactives_dataloader()))

        # calculate similarity & predict activity w.r.t. to query
        active_similarity = cosine_similarity(query, actives)
        inactive_similarity = cosine_similarity(query, inactives)
        # self.plot_tSNE(query, actives, inactives,
        #                active_similarity, inactive_similarity)

        def create_mask(similarities, mol_ids):
            splits = [similarities[mol_ids == i]
                      for i in range(max(mol_ids+1))]
            split_argmax = [torch.argmax(split) for split in splits]
            mask = [torch.zeros(split.shape) for split in splits]
            for idx, mask_split in zip(split_argmax, mask):
                mask_split[idx] = 1
            mask = torch.cat(mask) != 0

            return mask

        self.plot_UMAP(query, actives, inactives,
                       active_similarity, inactive_similarity)

        active_mask = create_mask(active_similarity, active_mol_ids)
        inactive_mask = create_mask(inactive_similarity, inactive_mol_ids)

        # Pick most similar conformation per compounds
        active_similarity = active_similarity[active_mask]
        inactive_similarity = inactive_similarity[inactive_mask]
        actives = actives[active_mask]
        inactives = inactives[inactive_mask]

        # active_similarity = global_max_pool(active_similarity, active_mol_ids)
        # inactive_similarity = global_max_pool(
        #     inactive_similarity, inactive_mol_ids)
        y_pred = torch.cat((active_similarity, inactive_similarity))
        y_pred = (y_pred + 1) / 2
        y_true = torch.cat((torch.ones(len(active_similarity), dtype=torch.int), torch.zeros(
            len(inactive_similarity), dtype=torch.int)))
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()

        return y_true, y_pred

    def assemble(self, prediction_output):
        predictions = []
        mol_ids = []
        for output in prediction_output:
            prediction, mol_id = output
            predictions.append(prediction)
            mol_ids.append(mol_id)

        return torch.vstack(predictions), torch.hstack(mol_ids)

    def plot_metrics(self, y_true, y_pred):
        # print metrics & plot figures
        print(roc_auc_score(y_true, y_pred))
        fpr, tpr, thr = roc_curve(y_true, y_pred)
        fig2 = plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC LigandScout Tutorial')
        plt.savefig('plots/auroc.png')
        fig3 = plt.figure()
        precision, recall, thr = precision_recall_curve(y_true, y_pred)
        plt.plot(precision, recall)
        plt.savefig('plots/prcurve.png')

    def plot_UMAP(self, query, actives, inactives, active_similarity, inactive_similarity):
        labels = torch.cat((torch.zeros(len(inactives)), torch.ones(
            len(actives)), torch.ones(len(query))*2)).numpy()
        features = torch.cat((inactives, actives, query)).numpy()
        similarity = torch.cat((inactive_similarity, active_similarity, torch.tensor(
            [1], device=active_similarity.device)))
        reducer = umap.UMAP()
        X_embedded = reducer.fit_transform(features)
        fig = plt.figure()
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1],
                    c=labels, marker='o', s=5)
        plt.savefig('umap.png')
