import os
import re
import sys
import yaml

from lightning import Trainer, seed_everything
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_curve,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
)
import torch
import torch_geometric
from torch_geometric.nn import global_mean_pool, global_max_pool
import umap

import CDPL.Pharm as Pharm
from dataset import PharmacophoreDataModule
from model import PhectorMatch
from utils import load_model_from_path
from virtual_screening import (
    VirtualScreeningEmbedder,
    VirtualScreener,
    PharmacophoreAlignment,
)

results = []
PROJECT_ROOT = "/data/shared/projects/PhectorDB"
PRETRAINING_ROOT = f"{PROJECT_ROOT}/training_data"
DATASET_ROOT = f"{PROJECT_ROOT}/DUDE-Z"

for TARGET in os.listdir(DATASET_ROOT):
    try:
        # Define global variables
        VS_ROOT = f"{DATASET_ROOT}/{TARGET}"
        MODEL = PhectorMatch
        VS_MODEL_NUMBER = 239
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
        datamodule = PharmacophoreDataModule(
            PRETRAINING_ROOT,
            VS_ROOT,
            batch_size=params["batch_size"],
            small_set_size=params["num_samples"],
        )
        datamodule.setup()

        # Load the model
        model = load_model_from_path(os.path.join(PROJECT_ROOT, MODEL_PATH), MODEL)
        device = [model.device.index]
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

        # Retrieve meta data from screening data base
        def getReaderByFileExt(filename: str) -> Pharm.PharmacophoreReader:
            name_and_ext = os.path.splitext(filename)

            if name_and_ext[1] == "":
                sys.exit(
                    "Error: could not determine pharmacophore input file format (file extension missing)"
                )

            # get input handler for the format specified by the input file's extension
            ipt_handler = Pharm.PharmacophoreIOManager.getInputHandlerByFileExtension(
                name_and_ext[1][1:].lower()
            )

            if not ipt_handler:
                sys.exit(
                    "Error: unsupported pharmacophore input file format '%s'"
                    % name_and_ext[1]
                )

            # create and return file reader instance
            return ipt_handler.createReader(filename)

        def create_metadata(path):
            reader = getReaderByFileExt(path)
            ph4 = Pharm.BasicPharmacophore()
            names = []
            features = []
            index = []
            conf_index = []
            num_features = []
            conf = 0
            i = 0
            name = ""

            while reader.read(ph4):
                if ph4.getNumFeatures() == 0:
                    continue
                feature_types = Pharm.generateFeatureTypeHistogramString(ph4)
                if name == Pharm.getName(ph4):
                    conf += 1
                else:
                    conf = 0
                    name = Pharm.getName(ph4)
                conf_index.append(conf)
                features.append(feature_types)
                names.append(name)
                index.append(i)
                num_features.append(ph4.getNumFeatures())
                i += 1

            metadata = pd.DataFrame(
                {
                    "index": index,
                    "name": names,
                    "conf_idx": conf_index,
                    "features": features,
                    "num_features": num_features,
                }
            )

            return metadata

        inactive_path = os.path.join(VS_ROOT, "raw", "inactives.psd")
        active_path = os.path.join(VS_ROOT, "raw", "actives.psd")
        query_path = os.path.join(VS_ROOT, "raw", "query.pml")

        inactive_metadata = create_metadata(inactive_path)
        active_metadata = create_metadata(active_path)
        query_metadata = create_metadata(query_path)

        experiment_data = dict()
        experiment_data["target"] = TARGET
        experiment_data["model"] = VS_MODEL_NUMBER
        experiment_data["embedding_time"] = screener.embedding_time
        experiment_data["matching_time"] = screener.matching_time
        num_features_query = query_metadata["num_features"].sum()
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
        alignment = PharmacophoreAlignment(vs_root=VS_ROOT)
        alignment.align_preprocessed_ligands_to_query()
        experiment_data["alignment_time"] = alignment.alignment_time

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
                fig1 = plt.figure()
                plt.plot(fpr, tpr)
                plt.title(
                    f"Order Embedding space alignment vs. CDPKit alignment (AUROC = {auroc:.2f})"
                )
                plt.xlabel("False positive rate")
                plt.ylabel("True positive rate")
                if path:
                    plt.savefig(path, dpi=300)
                else:
                    plt.show()
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

        def enrichment_factor(y_true, y_pred, alpha):
            sorted, indices = torch.sort(y_pred, descending=True)
            high_ranked_actives = sum(y_true[indices][: int(len(y_true) * alpha)])
            actives = sum(y_true)
            return high_ranked_actives / (actives * alpha)

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

        fig2 = plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
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

        plt.plot(fpr, tpr)
        plt.title(f"Hitlist")
        plt.legend(
            [
                f"Order Embedding (AUROC = {auroc:.2f})",
                f"CDPKit (AUROC = {auroc_cdp:.2f})",
            ]
        )
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
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

        plt.savefig(f"visualization/hitlist_{TARGET}.png", dpi=300)

        # PCA should conserve the order embedding space property
        mean_actives = global_mean_pool(
            screener.active_embeddings, screener.active_mol_ids
        )
        mean_inactives = global_mean_pool(
            screener.inactive_embeddings, screener.inactive_mol_ids
        )
        mean_vectors = torch.cat((mean_actives, mean_inactives))

        inactive_counts = inactive_metadata["num_features"].values
        pca = PCA(n_components=4)
        pca.fit(mean_vectors)
        inactive_transformed = pca.transform(screener.inactive_embeddings)
        variance = pca.explained_variance_ratio_
        fig3 = plt.figure()
        plt.scatter(
            inactive_transformed[:, 0],
            inactive_transformed[:, 1],
            c=inactive_counts,
            cmap="viridis",
            s=0.1,
        )
        plt.xlabel(f"PC1 ({variance[0]*100:.2f}%)")
        plt.ylabel(f"PC2 ({variance[1]*100:.2f}%)")
        cbar = plt.colorbar(aspect=40)
        cbar.set_label("Number of features")
        plt.xlim(-10, 15)
        plt.ylim(-10, 20)
        # plt.show()

        # PCA should conserve the order embedding space property
        active_counts = active_metadata["num_features"].values
        active_transformed = pca.transform(screener.active_embeddings)
        plt.xlabel(f"PC1 ({variance[0]*100:.2f}%)")
        plt.ylabel(f"PC2 ({variance[1]*100:.2f}%)")
        plt.scatter(
            active_transformed[:, 0],
            active_transformed[:, 1],
            c=active_counts,
            cmap="viridis",
            s=1,
        )
        plt.savefig(f"visualization/PCA_{TARGET}.png", dpi=300)
        # plt.show()

        query_transformed = pca.transform(screener.query_embedding)

        def get_feature_count(feature, hover_data):
            feature_count = []
            count = 0
            for feature_string in hover_data["features"]:
                if f"{feature}(" in feature_string:
                    for str in feature_string.split(","):
                        if f"{feature}(" in str:
                            count = int(re.findall(r"\d+", str)[0])
                else:
                    count = 0
                feature_count.append(count)

            return feature_count

        reducer = umap.UMAP(metric="manhattan")
        reducer.fit(mean_vectors)
        reduced_inactive_embeddings = reducer.transform(screener.inactive_embeddings)
        reduced_active_embeddings = reducer.transform(screener.active_embeddings)
        reduced_query_embedding = reducer.transform(screener.query_embedding)

        # The same plots, not interactive, but combined in one figure
        max_num_inactives = -1
        hover_data = pd.concat(
            (inactive_metadata[:max_num_inactives], active_metadata, query_metadata),
            ignore_index=True,
        )
        points = np.concatenate(
            (
                reduced_inactive_embeddings[:max_num_inactives],
                reduced_active_embeddings,
                reduced_query_embedding,
            )
        )

        fig4, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)

        # Plot actives and inactives
        ax = axes[0][0]
        sc = ax.scatter(
            reduced_inactive_embeddings[:, 0],
            reduced_inactive_embeddings[:, 1],
            c="darkblue",
            s=1,
            alpha=1,
            marker=".",
        )
        sc = ax.scatter(
            reduced_active_embeddings[:, 0],
            reduced_active_embeddings[:, 1],
            c="red",
            s=1,
            alpha=1,
            marker=".",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_xlim(-10, 15)
        # ax.set_ylim(-10, 20)
        ax.set_title("Active / Inactive")
        ax.legend(["Inactive", "Active"])
        ax.legend_.legend_handles[0]._sizes = [5]
        ax.legend_.legend_handles[1]._sizes = [5]

        # Plot feature counts
        features = {
            "H": "Hydrophobic",
            "AR": "Aromatic",
            "HBD": "Hydrogen Bond Donor",
            "HBA": "Hydrogen Bond Acceptor",
            "PI": "Positive Ionizable",
            "NI": "Negative Ionizable",
            "XBD": "Halogen Bond Donor",
        }
        cmaps = ["Oranges", "Purples", "Greens", "Reds", "Blues", "RdPu", "Greys"]

        for ax, feature, cmap_str in zip(
            axes.flatten()[1:8], list(features.keys()), cmaps
        ):
            feature_count = get_feature_count(feature, hover_data)
            new_cmap = cm.get_cmap(cmap_str, 256)
            cmap = ListedColormap(new_cmap(np.linspace(0.15, 1.0, max(feature_count))))
            sc = ax.scatter(
                points[:, 0], points[:, 1], c=feature_count, cmap=cmap, s=1, marker="."
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{features[feature]}")
            pos = ax.get_position()
            cax = plt.axes([pos.x0 + 0.01, pos.y0 + 0.025, 0.05, 0.005])
            cbar = plt.colorbar(
                sc, cax=cax, ticks=[0, max(feature_count)], location="bottom"
            )
        plt.savefig(
            f"visualization/embeddings_{TARGET}.png", dpi=300, bbox_inches="tight"
        )

        results.append(experiment_data)

    except Exception as e:
        print(e)
        continue

final_results = pd.DataFrame(results)
final_results.to_csv("final_results.csv")
