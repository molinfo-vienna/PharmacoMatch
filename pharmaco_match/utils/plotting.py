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
import umap.plot as uplot

from dataset import VirtualScreeningMetaData
from virtual_screening import VirtualScreener


class UmapEmbeddingPlotter:
    """UMAP plotter for visual inspection of the learned embedding space.

    Args:
        screener (VirtualScreener): VirtualScreener object.
        metadata (VirtualScreeningMetaData): VirtualScreeningMetaData object.
    """

    def __init__(self, screener: VirtualScreener, metadata: VirtualScreeningMetaData):
        self.screener = screener
        self.metadata = metadata

        # The UMAP embedder is trained on the mean embeddings of actives and inactives.
        # The reason is that some ligands have more conformations than others.
        # This way, each ligand is weighed equally.
        mean_actives = global_mean_pool(
            screener.active_embeddings, screener.active_mol_ids
        )
        mean_inactives = global_mean_pool(
            screener.inactive_embeddings, screener.inactive_mol_ids
        )
        mean_vectors = torch.cat((mean_actives, mean_inactives))

        # Train the UMAP embedder and create the reduced embeddings
        self.reducer = umap.UMAP(metric="manhattan")
        self.reducer.fit(mean_vectors)
        self.reduced_active_embeddings = self.reducer.transform(
            screener.active_embeddings
        )
        self.reduced_inactive_embeddings = self.reducer.transform(
            screener.inactive_embeddings
        )
        self.reduced_query_embedding = self.reducer.transform(screener.query_embedding)
        self.embeddings = np.concatenate(
            (
                self.reduced_active_embeddings,
                self.reduced_inactive_embeddings,
                self.reduced_query_embedding,
            )
        )

    def get_feature_count(self, feature, metadata):
        """Get count of pharmacophoric features from the metadata object."""
        feature_count = []
        count = 0
        for feature_string in metadata["features"]:
            if f"{feature}(" in feature_string:
                for str in feature_string.split(","):
                    if f"{feature}(" in str:
                        count = int(re.findall(r"\d+", str)[0])
            else:
                count = 0
            feature_count.append(count)

        return feature_count

    def create_umap_plot(self):
        """Configuration of the UMAP plot."""
        fig, axes = plt.subplots(
            2, 4, figsize=(24, 12), sharex=True, sharey=True, frameon=False
        )
        fontsize = 24
        metadata = pd.concat(
            (
                self.metadata.active,
                self.metadata.inactive,
                self.metadata.query,
            ),
            ignore_index=True,
        )

        # Plot actives and inactives
        ax = axes[0][0]
        plt.setp(ax.spines.values(), lw=1.5)
        sc = ax.scatter(
            self.reduced_inactive_embeddings[:, 0],
            self.reduced_inactive_embeddings[:, 1],
            c="darkblue",
            s=1,
            marker=".",
        )
        sc = ax.scatter(
            self.reduced_active_embeddings[:, 0],
            self.reduced_active_embeddings[:, 1],
            c="orange",
            s=1,
            marker=".",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Actives / Decoys", fontsize=fontsize, pad=15)
        ax.legend(["Decoys", "Actives"], fontsize=fontsize)
        ax.legend_.legend_handles[0]._sizes = [200]
        ax.legend_.legend_handles[1]._sizes = [200]

        # Plot by feature count
        features = {
            "H": "Hydrophobic",
            "AR": "Aromatic",
            "HBD": "Hydrogen bond donor",
            "HBA": "Hydrogen bond acceptor",
            "PI": "Positive ionizable",
            "NI": "Negative ionizable",
            "XBD": "Halogen bond donor",
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
            plt.setp(ax.spines.values(), lw=1.5)
            feature_count = self.get_feature_count(feature, metadata)
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
            ax.set_title(f"{features[feature]}", fontsize=fontsize, pad=15)
            pos = ax.get_position()
            cax = plt.axes([pos.x0 + 0.015, pos.y0 + 0.04, 0.05, 0.01])
            cbar = plt.colorbar(
                sc,
                cax=cax,
                ticks=[0, max(feature_count)],
                location="bottom",
            )
            cbar.ax.tick_params(labelsize=fontsize)

        return fig

    def _create_reduced_points_set(self, max_num_inactives=30000):
        """The interactive UMAP plot can't handle too much datapoints, so here we can
        reduce their number by some threshold"""

        hover_data = pd.concat(
            (
                self.metadata.inactive[:max_num_inactives],
                self.metadata.active,
                self.metadata.query,
            ),
            ignore_index=True,
        )
        points = np.concatenate(
            (
                self.reduced_inactive_embeddings[:max_num_inactives],
                self.reduced_active_embeddings,
                self.reduced_query_embedding,
            )
        )
        labels = np.concatenate(
            (
                np.zeros(len(self.reduced_inactive_embeddings[:max_num_inactives])),
                np.ones(len(self.reduced_active_embeddings)),
                np.ones(len(self.reduced_query_embedding)) * 2,
            )
        )

        return points, labels, hover_data

    def create_interactive_umap_by_activity(self):
        """Creates an interactive UMAP plot with activity labels"""
        points, labels, hover_data = self._create_reduced_points_set()
        self.reducer.embedding_ = points

        p = uplot.interactive(
            self.reducer,
            labels=labels,
            theme="inferno",
            hover_data=hover_data,
            point_size=2,
        )

        return p

    def create_interactive_umap_by_feature_type(self, feature="AR"):
        """Creates an interactive UMAP plot labeled by the number of a given
        pharmacophoric type"""
        points, _, hover_data = self._create_reduced_points_set()
        self.reducer.embedding_ = points
        feature_count = self.get_feature_count(feature, hover_data)

        p = uplot.interactive(
            self.reducer,
            values=feature_count,
            theme="darkblue",
            hover_data=hover_data,
            point_size=2,
        )

        return p


class PcaEmbeddingPlotter:
    """PCA plotter for visual inspection of the learned embedding space.

    Args:
        screener (VirtualScreener): VirtualScreener object.
        metadata (VirtualScreeningMetaData): VirtualScreeningMetaData object.
    """

    def __init__(self, screener: VirtualScreener, metadata: VirtualScreeningMetaData):
        # The PCA embedder is trained on the mean embeddings of actives and inactives.
        # The reason is that some ligands have more conformations than others.
        # This way, each ligand is weighed equally.
        mean_actives = global_mean_pool(
            screener.active_embeddings, screener.active_mol_ids
        )
        mean_inactives = global_mean_pool(
            screener.inactive_embeddings, screener.inactive_mol_ids
        )
        mean_vectors = torch.cat((mean_actives, mean_inactives))
        self.active_counts = metadata.active["num_features"].values
        self.inactive_counts = metadata.inactive["num_features"].values

        # Train the PCA embedder and create the reduced embeddings
        pca = PCA(n_components=4)
        pca.fit(mean_vectors)
        self.variance = pca.explained_variance_ratio_
        self.active_transformed = pca.transform(screener.active_embeddings)
        self.inactive_transformed = pca.transform(screener.inactive_embeddings)
        self.query_transformed = pca.transform(screener.query_embedding)

    def create_pca_plot(self):
        """Create PCA plot from the transformed embeddings."""
        transformed = np.concatenate(
            (self.inactive_transformed, self.active_transformed)
        )
        counts = np.concatenate((self.inactive_counts, self.active_counts))
        fontsize = 12
        fig = plt.figure(figsize=(4.5, 6))
        plt.scatter(
            transformed[:, 0],
            transformed[:, 1],
            c=counts,
            cmap="inferno",
            s=0.1,
        )

        cbar = plt.colorbar(
            aspect=40,
            ticks=[min(counts), max(counts)],
            orientation="horizontal",
            pad=0.075,
            spacing="uniform",
        )
        cbar.set_label("Number of points", fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)
        plt.xlabel(f"PC1 ({self.variance[0]*100:.2f}%)", fontsize=fontsize)
        plt.ylabel(f"PC2 ({self.variance[1]*100:.2f}%)", fontsize=fontsize)
        plt.xticks([])
        plt.yticks([])

        return fig
