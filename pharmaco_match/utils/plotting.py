import re

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from torch_geometric.nn import global_mean_pool
import torch
import umap


class UmapEmbeddingPlotter:
    def __init__(
        self, screener, datamodule
    ):
        self.screener = screener
        self.metadata = pd.concat(
            (
                datamodule.inactive_metadata,
                datamodule.active_metadata,
                datamodule.query_metadata,
            ),
            ignore_index=True,
        )

        mean_actives = global_mean_pool(
            screener.active_embeddings, screener.active_mol_ids
        )
        mean_inactives = global_mean_pool(
            screener.inactive_embeddings, screener.inactive_mol_ids
        )
        mean_vectors = torch.cat((mean_actives, mean_inactives))

        reducer = umap.UMAP(metric="manhattan")
        reducer.fit(mean_vectors)
        self.reduced_inactive_embeddings = reducer.transform(
            screener.inactive_embeddings
        )
        self.reduced_active_embeddings = reducer.transform(screener.active_embeddings)
        self.reduced_query_embedding = reducer.transform(screener.query_embedding)
        self.embeddings = np.concatenate(
            (
                self.reduced_inactive_embeddings,
                self.reduced_active_embeddings,
                self.reduced_query_embedding,
            )
        )

    def get_feature_count(self, feature, hover_data):
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

    def create_umap_plot(self):
        fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)

        # Plot actives and inactives
        ax = axes[0][0]
        sc = ax.scatter(
            self.reduced_inactive_embeddings[:, 0],
            self.reduced_inactive_embeddings[:, 1],
            c="darkblue",
            s=1,
            alpha=1,
            marker=".",
        )
        sc = ax.scatter(
            self.reduced_active_embeddings[:, 0],
            self.reduced_active_embeddings[:, 1],
            c="red",
            s=1,
            alpha=1,
            marker=".",
        )
        ax.set_xticks([])
        ax.set_yticks([])
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
        cmaps = [
            "Oranges",
            "Purples",
            "Greens",
            "Reds",
            "Blues",
            "RdPu",
            "Greys",
        ]

        for ax, feature, cmap_str in zip(
            axes.flatten()[1:8], list(features.keys()), cmaps
        ):
            feature_count = self.get_feature_count(feature, self.metadata)
            new_cmap = cm.get_cmap(cmap_str, 256)
            cmap = ListedColormap(new_cmap(np.linspace(0.15, 1.0, max(feature_count))))
            sc = ax.scatter(
                self.embeddings[:, 0],
                self.embeddings[:, 1],
                c=feature_count,
                cmap=cmap,
                s=1,
                marker=".",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{features[feature]}")
            pos = ax.get_position()
            cax = plt.axes([pos.x0 + 0.01, pos.y0 + 0.025, 0.05, 0.005])
            cbar = plt.colorbar(
                sc, cax=cax, ticks=[0, max(feature_count)], location="bottom"
            )

        return fig


class PcaEmbeddingPlotter:
    def __init__(self, screener, datamodule):
        mean_actives = global_mean_pool(
            screener.active_embeddings, screener.active_mol_ids
        )
        mean_inactives = global_mean_pool(
            screener.inactive_embeddings, screener.inactive_mol_ids
        )
        mean_vectors = torch.cat((mean_actives, mean_inactives))

        self.active_counts = datamodule.active_metadata["num_features"].values
        self.inactive_counts = datamodule.inactive_metadata["num_features"].values

        pca = PCA(n_components=4)
        pca.fit(mean_vectors)
        self.variance = pca.explained_variance_ratio_
        self.active_transformed = pca.transform(screener.active_embeddings)
        self.inactive_transformed = pca.transform(screener.inactive_embeddings)
        self.query_transformed = pca.transform(screener.query_embedding)

    def create_pca_plot(self):
        fig = plt.figure()
        plt.scatter(
            self.inactive_transformed[:, 0],
            self.inactive_transformed[:, 1],
            c=self.inactive_counts,
            cmap="viridis",
            s=0.1,
        )
        plt.xlabel(f"PC1 ({self.variance[0]*100:.2f}%)")
        plt.ylabel(f"PC2 ({self.variance[1]*100:.2f}%)")
        cbar = plt.colorbar(aspect=40)
        cbar.set_label("Number of features")
        # plt.xlim(-10, 15)
        # plt.ylim(-10, 20)
        # plt.show()

        # PCA should conserve the order embedding space property
        plt.xlabel(f"PC1 ({self.variance[0]*100:.2f}%)")
        plt.ylabel(f"PC2 ({self.variance[1]*100:.2f}%)")
        plt.scatter(
            self.active_transformed[:, 0],
            self.active_transformed[:, 1],
            c=self.active_counts,
            cmap="viridis",
            s=1,
        )

        return fig
