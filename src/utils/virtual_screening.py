import torch
from torch_geometric.nn import global_max_pool
from    torch.nn.functional import cosine_similarity
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

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
        query, _ = self.assemble(self.trainer.predict(model=self.model, dataloaders=datamodule.query_dataloader()))
        actives, active_mol_ids = self.assemble(self.trainer.predict(model=self.model, dataloaders=datamodule.actives_dataloader()))
        inactives, inactive_mol_ids = self.assemble(self.trainer.predict(model=self.model, dataloaders=datamodule.inactives_dataloader()))
        
        # calculate similarity & predict activity w.r.t. to query
        active_similarity = cosine_similarity(query, actives)
        inactive_similarity = cosine_similarity(query, inactives)
        active_similarity = global_max_pool(active_similarity, active_mol_ids)
        inactive_similarity = global_max_pool(inactive_similarity, inactive_mol_ids)
        y_pred = torch.cat((active_similarity, inactive_similarity))
        y_pred = (y_pred + 1) / 2
        y_true = torch.cat((torch.ones(len(active_similarity), dtype=torch.int), torch.zeros(len(inactive_similarity), dtype=torch.int)))
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
        plt.plot(fpr, tpr)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC LigandScout Tutorial')
        plt.savefig('plots/auroc.png')
        precision, recall, thr = precision_recall_curve(y_true, y_pred)
        plt.plot(precision, recall)
        plt.savefig('plots/prcurve.png')
        