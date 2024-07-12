import os
import yaml

from lightning import Trainer, seed_everything
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
)
import torch
import torch_geometric
from torch_geometric.nn import global_max_pool

from dataset import VirtualScreeningDataModule
from model import PhectorMatch
from utils import (
    load_model_from_path,
    PharmacophoreMetaData,
    enrichment_factor,
    UmapEmbeddingPlotter,
    PcaEmbeddingPlotter,
)
from virtual_screening import (
    VirtualScreeningEmbedder,
    VirtualScreener,
    PharmacophoreAlignment,
)

results = []
PROJECT_ROOT = "/data/shared/projects/PhectorDB"
DATASET_ROOT = f"{PROJECT_ROOT}/DUDE-Z"

fig1, axes1 = plt.subplots(2, 5, figsize=(25, 10), sharex=True, sharey=True)
fig2, axes2 = plt.subplots(2, 5, figsize=(25, 10), sharex=True, sharey=True)
i = 0

for TARGET in sorted(os.listdir(DATASET_ROOT)):
    try:
        print(TARGET)
        # Define global variables
        VS_ROOT = f"{DATASET_ROOT}/{TARGET}"
        MODEL = PhectorMatch
        VS_MODEL_NUMBER = 250
        MODEL_PATH = f"{PROJECT_ROOT}/logs/{MODEL.__name__}/version_{VS_MODEL_NUMBER}/"
        HPARAMS_FILE = "hparams.yaml"

        params = yaml.load(
            open(os.path.join(PROJECT_ROOT, MODEL_PATH, HPARAMS_FILE), "r"),
            Loader=yaml.FullLoader,
        )

        # Deterministic flags (should not be necessary for inference, but just in case)
        torch.set_float32_matmul_precision("medium")
        torch_geometric.seed_everything(params["seed"])
        seed_everything(params["seed"])
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

        # Setup datamodule
        datamodule = VirtualScreeningDataModule(
            VS_ROOT,
            batch_size=params["batch_size"],
        )
        datamodule.setup()

        # Load the model
        model = load_model_from_path(os.path.join(PROJECT_ROOT, MODEL_PATH), MODEL)
        trainer = Trainer(
            num_nodes=1,
            devices=[0],
            max_epochs=params["epochs"],
            accelerator="auto",
            logger=False,
            log_every_n_steps=2,
        )

        # Create embeddings of the VS dataset
        embedder = VirtualScreeningEmbedder(model, datamodule, trainer)
        screener = VirtualScreener(embedder)
        metadata = PharmacophoreMetaData(VS_ROOT)

        # inactive_path = os.path.join(VS_ROOT, "raw", "inactives.psd")
        # active_path = os.path.join(VS_ROOT, "raw", "actives.psd")
        # query_path = os.path.join(VS_ROOT, "raw", "query.pml")

        # inactive_metadata = create_metadata(inactive_path)
        # active_metadata = create_metadata(active_path)
        # query_metadata = create_metadata(query_path)

        experiment_data = dict()
        experiment_data["target"] = TARGET
        experiment_data["model"] = VS_MODEL_NUMBER
        experiment_data["embedding_time"] = screener.embedding_time
        experiment_data["matching_time"] = screener.matching_time
        num_features_query = metadata.query_metadata["num_features"].sum()
        experiment_data["query_num_features"] = num_features_query
        experiment_data["active_ligands"] = (
            torch.max(screener.active_mol_ids).item() + 1
        )
        experiment_data["inactive_ligands"] = (
            torch.max(screener.inactive_mol_ids).item() + 1
        )
        experiment_data["active_conformations"] = len(screener.active_mol_ids)
        experiment_data["inactive_conformations"] = len(screener.inactive_mol_ids)

        # Calculate alignment solution with CDPKit
        # You only need to run this once, the results are saved in the VS_ROOT
        # alignment = PharmacophoreAlignment(vs_root=VS_ROOT)
        # alignment.align_preprocessed_ligands_to_query()
        # experiment_data["alignment_time"] = alignment.alignment_time

        # Retrieve alignment score
        active_alignment_score = torch.load(
            os.path.join(VS_ROOT, "vs/all_actives_aligned.pt")
        )
        inactive_alignment_score = torch.load(
            os.path.join(VS_ROOT, "vs/all_inactives_aligned.pt")
        )

        def order_embedding_alignment_evaluation(
            order_embedding_score,
            cdp_kit_aligned_features,
            num_features_query,
            mol_ids,
            prefilter=None,
            path=None,
        ):
            # The alignment was successful if the number of aligned features matches the number of query features
            y_true = cdp_kit_aligned_features >= num_features_query
            y_pred = -order_embedding_score
            # Determine optimal threshold via ROC curve
            fpr, tpr, threshold = roc_curve(y_true, y_pred)
            auroc = roc_auc_score(y_true, y_pred)
            j = tpr - fpr
            idx = np.argmax(j)
            best_threshold = -threshold[idx]
            y_pred = order_embedding_score <= best_threshold

            if prefilter is not None:
                y_pred = y_pred * prefilter

            else:
                axes1[i // 5][i % 5].plot(fpr, tpr)
                axes1[i // 5][i % 5].set_title(f"{TARGET} (AUROC = {auroc:.2f})")
                axes1[i // 5][i % 5].set_xlabel("False positive rate")
                axes1[i // 5][i % 5].set_ylabel("True positive rate")

            print(
                f"According to the ROC curve, the best threshold is {best_threshold}."
            )
            print("----------------------------")
            print(f"This yields an MCC of {matthews_corrcoef(y_true, y_pred)}.")
            print(f"AUROC = {auroc}.")
            print("Confusion matrix:")
            print(confusion_matrix(y_true, y_pred))

            print("----------------------------")
            print(
                "Aggregation of conformational ensembles into one result per molecule:"
            )
            y_true = global_max_pool(y_true, mol_ids)
            y_pred = global_max_pool(y_pred, mol_ids)
            print(f"This yields an MCC of {matthews_corrcoef(y_true, y_pred)}.")
            print("Confusion matrix:")
            print(confusion_matrix(y_true, y_pred))

            return auroc

        print("\n--- Results for the combination of both ---")
        auroc = order_embedding_alignment_evaluation(
            torch.cat((screener.active_query_match, screener.inactive_query_match)),
            torch.cat((active_alignment_score[:, 0], inactive_alignment_score[:, 0])),
            num_features_query,
            torch.cat(
                (
                    screener.active_mol_ids,
                    screener.inactive_mol_ids + screener.active_mol_ids[-1] + 1,
                )
            ),
            path=f"visualization/comparison_{TARGET}.png",
        )
        experiment_data["auroc_comparison"] = auroc

        # Hitlist with the order embedding algorithm
        order_embedding_score = -torch.cat(
            (screener.active_query_match, screener.inactive_query_match)
        )
        mol_ids = torch.cat(
            (
                screener.active_mol_ids,
                screener.inactive_mol_ids + screener.active_mol_ids[-1] + 1,
            )
        )
        y_pred = global_max_pool(order_embedding_score, mol_ids)
        y_true = torch.cat(
            (
                torch.ones(screener.active_mol_ids[-1] + 1),
                torch.zeros(screener.inactive_mol_ids[-1] + 1),
            )
        )
        path = "matching.png"
        fpr, tpr, threshold = roc_curve(y_true, y_pred)
        auroc = roc_auc_score(y_true, y_pred)
        experiment_data["order_embedding_auroc"] = auroc
        j = tpr - fpr
        idx = np.argmax(j)

        axes2[i // 5][i % 5].plot(fpr, tpr)
        axes2[i // 5][i % 5].set_xlabel("False positive rate")
        axes2[i // 5][i % 5].set_ylabel("True positive rate")
        alphas = [0.005, 0.01, 0.05]
        for alpha in alphas:
            enrichment = enrichment_factor(y_true, y_pred, alpha).item()
            print(f"The enrichment factor at {alpha} is {enrichment}.")
            experiment_data[f"order_embedding_ef{alpha}"] = enrichment
        print("----------------------------")
        best_threshold = -threshold[idx]
        y_pred = -y_pred <= best_threshold
        print(f"According to the ROC curve, the best threshold is {best_threshold}.")
        print(f"This yields an MCC of {matthews_corrcoef(y_true, y_pred)}.")
        print(f"AUROC = {auroc}.")
        print("Confusion matrix:")
        print(confusion_matrix(y_true, y_pred))

        # Hitlist with the CDPKit alignment
        alignment_score = torch.cat(
            (
                active_alignment_score[:, 0] + active_alignment_score[:, 1],
                inactive_alignment_score[:, 0] + inactive_alignment_score[:, 1],
            )
        )
        mol_ids = torch.cat(
            (
                screener.active_mol_ids,
                screener.inactive_mol_ids + screener.active_mol_ids[-1] + 1,
            )
        )
        y_pred = global_max_pool(alignment_score, mol_ids)
        y_true = torch.cat(
            (
                torch.ones(screener.active_mol_ids[-1] + 1),
                torch.zeros(screener.inactive_mol_ids[-1] + 1),
            )
        )
        path = "matching.png"
        fpr, tpr, threshold = roc_curve(y_true, y_pred)
        auroc_cdp = roc_auc_score(y_true, y_pred)
        experiment_data["cdpkit_auroc"] = auroc_cdp

        j = tpr - fpr
        idx = np.argmax(j)

        axes2[i // 5][i % 5].plot(fpr, tpr)
        axes2[i // 5][i % 5].set_title(f"{TARGET}")
        axes2[i // 5][i % 5].legend(
            [
                f"Order Embedding (AUROC = {auroc:.2f})",
                f"CDPKit (AUROC = {auroc_cdp:.2f})",
            ],
            loc="lower right",
        )
        axes2[i // 5][i % 5].set_xlabel("False positive rate")
        axes2[i // 5][i % 5].set_ylabel("True positive rate")
        alphas = [0.005, 0.01, 0.05]

        for alpha in alphas:
            enrichment = enrichment_factor(y_true, y_pred, alpha).item()
            print(f"The enrichment factor at {alpha} is {enrichment}.")
            experiment_data[f"cdpkit_ef{alpha}"] = enrichment

        print("----------------------------")
        best_threshold = threshold[idx]
        y_pred = y_pred >= best_threshold
        print(f"According to the ROC curve, the best threshold is {best_threshold}.")
        print(f"This yields an MCC of {matthews_corrcoef(y_true, y_pred)}.")
        print(f"AUROC = {auroc_cdp}.")
        print("Confusion matrix:")
        print(confusion_matrix(y_true, y_pred))

        i += 1

        # PCA should conserve the order embedding space property
        pca_plotter = PcaEmbeddingPlotter(screener, metadata)
        fig3 = pca_plotter.create_pca_plot()
        fig3.savefig(f"visualization/PCA_{TARGET}.png", dpi=300)

        umap_plotter = UmapEmbeddingPlotter(screener, metadata)
        fig4 = umap_plotter.create_umap_plot()
        fig4.savefig(
            f"visualization/embeddings_{TARGET}.png",
            dpi=300,
            bbox_inches="tight",
        )

        results.append(experiment_data)

    except Exception as e:
        print(e)
        continue

fig1.savefig("comparison.png", dpi=300, bbox_inches="tight")
fig2.savefig("hitlist.png", dpi=300, bbox_inches="tight")
final_results = pd.DataFrame(results)
final_results.to_csv("final_results.csv")
