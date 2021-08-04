import io
import itertools
import re
from textwrap import wrap
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def mat_vis(
    mat: np.array,
    x_ticks: List[str],
    y_ticks: List[str],
    save_path: str,
    x_label="x",
    y_label="y",
    title='matrix vis',
    tile_scale=0.4,
    normalize=False
):
    """

    :param true: These are your true classification categories.
    :param pred: These are you predicted classification categories
    :param classes: This is a lit of labels which will be used to display the
    axis labels
    :param title: Title for your matrix
    :param tensor_name: Name for the output summay tensor
    :param normalize:

    :return summary: TensorFlow summary

    Other itema to note:
    - Depending on the number of category and the data , you may have to modify
    the fig size, font sizes etc.
    - Currently, some of the ticks dont line up due to rotations.
    """
    plt.switch_backend('agg')

    if normalize:
        mat = mat.astype('float') * 10 / mat.sum(axis=1)[:, np.newaxis]
        mat = mat.nan_to_num(mat, copy=True)
        mat = mat.astype('int')

    np.set_printoptions(precision=2)
    axis_offset = 4
    fig = plt.figure(
        figsize=(
            int(len(x_ticks) * tile_scale) + axis_offset,
            int(len(y_ticks) * tile_scale) + axis_offset
        ),
        dpi=160,
        facecolor='w',
        edgecolor='k'
    )
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(mat, cmap='Oranges')

    x_ticks = [
        re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x)
        for x in x_ticks
    ]

    y_ticks = [
        re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x)
        for x in y_ticks
    ]

    x_ticks = ['\n'.join(wrap(l, 40)) for l in x_ticks]
    y_ticks = ['\n'.join(wrap(l, 40)) for l in y_ticks]

    ax.set_xlabel(x_label, fontsize=7)
    ax.set_xticks(np.arange(len(x_ticks)))
    c = ax.set_xticklabels(x_ticks, fontsize=4, rotation=-90, ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel(y_label, fontsize=7)
    ax.set_yticks(np.arange(len(y_ticks)))
    ax.set_yticklabels(y_ticks, fontsize=4, va='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(mat.shape[0]), range(mat.shape[1])):
        ax.text(
            j,
            i,
            f"{mat[i, j]:.2f}" if mat[i, j] != 0 else '.',
            horizontalalignment="center",
            fontsize=6,
            verticalalignment='center',
            color="black"
        )
    # fig.set_tight_layout(True)

    plt.savefig(save_path, format='png')
