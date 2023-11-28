import sys
import yaml

from lightning import Trainer, seed_everything
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import torch
import torch_geometric
import umap

from dataset import PharmacophoreDataModule
from model import PharmCLR
from utils import load_model_from_path


class VirtualScreening:
    def __init__(self, model, trainer) -> None:
        self.model = model
        self.trainer = trainer

    def __call__(self, datamodule) -> None:
        # create embeddings

        val_embeddings = torch.cat(
            self.trainer.predict(
                model=self.model, dataloaders=datamodule.create_val_dataloader()
            )
        )
        query, _ = self.assemble(
            self.trainer.predict(
                model=self.model, dataloaders=datamodule.query_dataloader()
            )
        )
        actives, active_mol_ids = self.assemble(
            self.trainer.predict(
                model=self.model, dataloaders=datamodule.actives_dataloader()
            )
        )
        inactives, inactive_mol_ids = self.assemble(
            self.trainer.predict(
                model=self.model, dataloaders=datamodule.inactives_dataloader()
            )
        )

        # calculate cosine similarity of (in)actives w.r.t. to query
        active_similarity = torch.mm(query, actives.T).flatten()
        inactive_similarity = torch.mm(query, inactives.T).flatten()

        # pick most similar conformation per compound
        active_mask = self.create_mask(active_similarity, active_mol_ids)
        inactive_mask = self.create_mask(inactive_similarity, inactive_mol_ids)
        active_similarity = active_similarity[active_mask]
        inactive_similarity = inactive_similarity[inactive_mask]
        top_actives = actives[active_mask]
        top_inactives = inactives[inactive_mask]

        # plot UMAP of highest scoring active and inactive embeddings
        self.plot_UMAP(
            query, top_actives, top_inactives, actives, inactives, val_embeddings
        )

        # Map similarity [-1, 1] --> [0, 1] and print AUC statistics
        y_pred = torch.cat((active_similarity, inactive_similarity))
        y_pred = (y_pred + 1) / 2
        y_true = torch.cat(
            (
                torch.ones(len(active_similarity), dtype=torch.int),
                torch.zeros(len(inactive_similarity), dtype=torch.int),
            )
        )
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()

        self.plot_auc(y_true, y_pred)

    def assemble(self, prediction_output):
        predictions = []
        mol_ids = []
        for output in prediction_output:
            prediction, mol_id = output
            predictions.append(prediction)
            mol_ids.append(mol_id)

        return torch.vstack(predictions), torch.hstack(mol_ids)

    def create_mask(self, similarities, mol_ids):
        splits = [similarities[mol_ids == i] for i in range(max(mol_ids + 1))]
        split_argmax = [torch.argmax(split) for split in splits]
        mask = [torch.zeros(split.shape) for split in splits]
        for idx, mask_split in zip(split_argmax, mask):
            mask_split[idx] = 1
        mask = torch.cat(mask) != 0

        return mask

    def plot_auc(self, y_true, y_pred):
        # print metrics & plot figures
        print(roc_auc_score(y_true, y_pred))
        fpr, tpr, thr = roc_curve(y_true, y_pred)
        fig2 = plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC LigandScout Tutorial")
        plt.savefig("plots/auroc.png")
        fig3 = plt.figure()
        precision, recall, thr = precision_recall_curve(y_true, y_pred)
        plt.plot(precision, recall)
        plt.savefig("plots/prcurve.png")

    def plot_UMAP(
        self, query, actives, inactives, all_actives, all_inactives, val_embeddings
    ):
        reducer = umap.UMAP()
        reducer.fit(val_embeddings)
        all_inactives_embedded = reducer.transform(all_inactives)
        all_actives_embedded = reducer.transform(all_actives)
        inactives_embedded = reducer.transform(inactives)
        actives_embedded = reducer.transform(actives)
        query_embedded = reducer.transform(query)

        fig = plt.figure(figsize=(15, 15))
        plt.scatter(
            all_inactives_embedded[:, 0],
            all_inactives_embedded[:, 1],
            c="cornflowerblue",
            marker="o",
            s=10,
        )
        plt.scatter(
            all_actives_embedded[:, 0],
            all_actives_embedded[:, 1],
            c="lightcoral",
            marker="o",
            s=10,
        )
        plt.scatter(
            inactives_embedded[:, 0],
            inactives_embedded[:, 1],
            c="blue",
            marker="o",
            edgecolor="darkblue",
            s=20,
        )
        plt.scatter(
            actives_embedded[:, 0],
            actives_embedded[:, 1],
            c="red",
            marker="o",
            edgecolor="darkred",
            s=20,
        )
        plt.scatter(
            query_embedded[:, 0],
            query_embedded[:, 1],
            c="yellow",
            marker="*",
            s=300,
            edgecolor="black",
        )
        plt.legend(
            [
                "Inactive Conformation (CDK2)",
                "Active Conformation (CDK2)",
                "Inactive Conformation, compound-wise highest query similarity",
                "Active Conformation, compound-wise highest query similarity",
                "Query (Shared-feature pharamcophore of 1ke6/7/8)",
            ]
        )
        plt.title("UMAP of PharmCLR Embedding Space")
        plt.savefig("umap.png", dpi=250)


def evaluation(device):
    PRETRAINING_ROOT = "/data/shared/projects/PhectorDB/chembl_data"
    VS_ROOT = "/data/shared/projects/PhectorDB/virtual_screening_cdk2"
    CONFIG_FILE_PATH = "/home/drose/git/PhectorDB/src/scripts/config.yaml"
    MODEL = PharmCLR
    VS_MODEL_NUMBER = 36
    MODEL_PATH = f"logs/PharmCLR/version_{VS_MODEL_NUMBER}/checkpoints/"

    params = yaml.load(open(CONFIG_FILE_PATH, "r"), Loader=yaml.FullLoader)

    torch.set_float32_matmul_precision("medium")
    torch_geometric.seed_everything(params["seed"])
    seed_everything(params["seed"])
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    datamodule = PharmacophoreDataModule(
        PRETRAINING_ROOT,
        VS_ROOT,
        batch_size=params["batch_size"],
        small_set_size=params["num_samples"],
    )
    datamodule.setup()

    model = load_model_from_path(MODEL_PATH, MODEL)

    trainer = Trainer(
        num_nodes=1,
        devices=device,
        max_epochs=params["epochs"],
        accelerator="auto",
        logger=False,
        log_every_n_steps=1,
    )

    vs = VirtualScreening(model, trainer)
    vs(datamodule)


if __name__ == "__main__":
    device = [int(i) for i in list(sys.argv[1])]
    evaluation(device)
