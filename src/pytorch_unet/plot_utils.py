import itertools
from functools import reduce
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from matplotlib.ticker import MaxNLocator


def plot_side_by_side(grayscale_img: np.ndarray, colored_mask_img: np.ndarray):
    """Side by side plot for grayscale and colored image."""
    _, (ax1, ax2) = plt.subplots(
        ncols=2, sharex=True, sharey=True, figsize=(8, 4)
    )
    ax1.imshow(grayscale_img)
    ax2.imshow(colored_mask_img)


def plot_losses(losses: Dict[str, Dict[str, List[float]]]):
    clear_output(wait=True)
    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(8, 4))
    for i, mode in enumerate(losses.keys()):
        for k in losses[mode].keys():
            axes[i].plot(losses[mode][k], label=k)
        axes[i].set_title(mode)
        axes[i].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[i].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    fig.text(s="Epoch", x=".5", y=".01", ha="center")
    fig.text(
        s="Loss",
        x=".05",
        y=".5",
        ha="center",
        va="center",
        rotation="vertical",
    )
    plt.show()
