import sys
import os

import torch
import torch_geometric

from lightning import Trainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from utils import *
from dataset import *
from model import *

from torch_geometric.nn import global_max_pool, global_mean_pool
from    torch.nn.functional import cosine_similarity
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt


def run(device):
    ROOT = "/data/shared/projects/PhectorDB/pretraining_data/small"
    EPOCHS = 1000
    TRAINING = True
    BATCH_SIZE = 512
    MODEL = PharmCLR
    torch.set_float32_matmul_precision("medium")
    torch_geometric.seed_everything(42)
    datamodule = PharmacophoreDataModule(ROOT, batch_size=BATCH_SIZE)

    def training():
        datamodule.setup("fit")
        params = dict(num_node_features=9, num_edge_features=5)
        hyperparams = MODEL.get_hyperparams()
        model = MODEL(hyperparams, params)

        tb_logger = TensorBoardLogger(
            "logs/", name=f"PharmCLR", default_hp_metric=False
        )

        callbacks = []

        trainer = Trainer(
            num_nodes=1,
            devices=[device],
            max_epochs=EPOCHS,
            accelerator="gpu",
            logger=tb_logger,
            log_every_n_steps=1,
            callbacks=callbacks,
        )

        trainer.fit(model=model, datamodule=datamodule)

    def testing():
        # load the trained model
        path = f'logs/PharmCLR/version_4/checkpoints/'
        for file in os.listdir(path):
            if file.endswith(".ckpt"):
                path = os.path.join(path, file)
        model = MODEL.load_from_checkpoint(path)
        datamodule.setup('test')

        trainer = Trainer(num_nodes=1,
                    devices=[device],
                    max_epochs=EPOCHS,
                    accelerator='auto',
                    logger=False,
                    log_every_n_steps=1)
        
        def assemble(prediction_output):
            predictions = []
            mol_ids = []
            for output in prediction_output:
                prediction, mol_id = output
                predictions.append(prediction)
                mol_ids.append(mol_id)
            
            return torch.vstack(predictions), torch.hstack(mol_ids)
        
        # encode pharmacophore data
        query, _ = assemble(trainer.predict(model=model, dataloaders=datamodule.query_dataloader()))
        actives, active_mol_ids = assemble(trainer.predict(model=model, dataloaders=datamodule.actives_dataloader()))
        inactives, inactive_mol_ids = assemble(trainer.predict(model=model, dataloaders=datamodule.inactives_dataloader()))
        
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

    if TRAINING:
        training()
    else:
        testing()


if __name__ == "__main__":
    device = int(sys.argv[1])
    run(device)
