import matplotlib.pyplot as plt
import numpy as np


def visualize_pharm(data_list):
    # plot configurations
    fig = plt.figure()

    ax = fig.add_subplot(111, projection="3d")
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])

    colors = [
        "blue",
        "orange",
        "green",
        "red",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "black",
    ]

    def plot_data(data, edge_color):
        pos = data.pos.cpu().numpy()
        x = data.x.cpu().numpy()

        if data.edge_index is not None:
            edge_index = data.edge_index.cpu()
            for src, dst in edge_index.t().tolist():
                src = pos[src].tolist()
                dst = pos[dst].tolist()
                ax.plot(
                    [src[0], dst[0]],
                    [src[1], dst[1]],
                    [src[2], dst[2]],
                    linewidth=0.3,
                    color=edge_color,
                )

        # List of feature types
        features = [(np.argmax(x, axis=1) == 0) * (np.sum(x, axis=1) != 0)]
        for i in range(1, 9):
            features.append(np.argmax(x, axis=1) == i)
        features.append(np.sum(x, axis=1) == 0)  # --> masked features

        for feature, c in zip(features, colors):
            ax.scatter(
                pos[feature, 0],
                pos[feature, 1],
                pos[feature, 2],
                s=50,
                color=c,
            )

    for i, data in enumerate(data_list):
        plot_data(data, colors[i % 10])

    plt.show()
