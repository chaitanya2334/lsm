import io
import itertools
import re
from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from PIL import Image
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    true,
    pred,
    classes,
    title='Confusion matrix',
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

    if len(true) == len(pred) == 0:
        cm = np.zeros((len(classes), len(classes)))
    else:
        cm = confusion_matrix(true, pred, labels=classes)

    if normalize:
        cm = cm.astype('float') * 10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')

    np.set_printoptions(precision=2)
    axis_offset = 1
    fig = plt.figure(
        figsize=(
            int(len(classes) * tile_scale) + axis_offset,
            int(len(classes) * tile_scale) + axis_offset
        ),
        dpi=160,
        facecolor='w',
        edgecolor='k'
    )
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(cm, cmap='Oranges')

    classes = [
        re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x)
        for x in classes
    ]

    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=4, rotation=-90, ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=4, va='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(
            j,
            i,
            format(cm[i, j], 'd') if cm[i, j] != 0 else '.',
            horizontalalignment="center",
            fontsize=6,
            verticalalignment='center',
            color="black"
        )
    # fig.set_tight_layout(True)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = np.array(Image.open(buf))[:, :, :3]

    return image
