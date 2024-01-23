import numpy as np
import matplotlib.pyplot as plt


def plot_con(Cij):
    fig = plt.figure(figsize=(7, 7))
    
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(4, 1),
        height_ratios=(1, 4),
        left=0.1,
        right=0.9,
        bottom=0.1,
        top=0.9,
        wspace=0.15,
        hspace=0.15,
    )
    
    # Create the Axes.
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_xy = fig.add_subplot(gs[0, 1])

    # fig, ax = plt.subplots(figsize=(7,7))
    im = ax.imshow(Cij, cmap="jet", aspect=1)
    ax.set_xlabel("Presynaptic")
    ax.set_ylabel("Postsynaptic")
    
    # cbar = plt.colorbar(im, ax=ax)
    # cbar.set_label("$J_{ij}$")
    # cbar.set_ticks([0, 5, 10, 15])

    Kj = np.sum(Cij, axis=0)  # sum over pres
    ax_histx.plot(Kj)
    ax_histx.set_xticklabels([])
    ax_histx.set_ylabel("$K_j$")
    
    Ki = np.sum(Cij, axis=1)  # sum over pres
    ax_histy.plot(Ki, np.arange(0, Ki.shape[0], 1))
    ax_histy.set_yticklabels([])
    ax_histy.set_xlabel("$K_i$")

    con_profile(Cij, ax=ax_xy)

def con_profile(Cij, ax=None):
    diags = []
    for i in range(int(Cij.shape[0] / 2)):
        diags.append(np.trace(Cij, offset=i) / Cij.shape[0])

    diags = np.array(diags)
    if ax is None:
        plt.plot(diags)
    else:
        ax.plot(diags)
        ax.set_xticklabels([])
        # ax.set_yticklabels([])

    plt.xlabel("Neuron #")
    plt.ylabel("$P_{ij}$")
