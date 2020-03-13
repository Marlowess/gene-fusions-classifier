# ======================================================================================= #
#                                    plot_functions.py                                         #
# ======================================================================================= #

from __future__ import print_function

# ======================================================================================= #
#                                    SciKit - Learn                                       #
# ======================================================================================= #
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

from sklearn.model_selection import train_test_split

from sklearn.utils.multiclass import unique_labels

# ======================================================================================= #
#                                    Other Imports                                        #
# ======================================================================================= #
from matplotlib import pyplot as plt
# plt.style.use('dark_background')

import numpy as np
import pandas as  pd

import io
import os
import pprint
import random
import sys

from matplotlib.ticker import FuncFormatter

dark_ppt = False # False

# =============================================================================================== #
#                              PLOTS: VALIDATION LOSS & ACCURACY                                  #
# =============================================================================================== #
def plot_loss(history, fig_dir: str, title: str, fig_name: str = None, fig_format: str = 'png', savefig_flag: bool = False, showfig_flag: bool = True) -> None:
    plt.clf()

    loss = history.history['loss']
    epochs = range(1, len(loss) + 1)
    
    plt.plot(epochs, loss, 'g', label='Training loss')

    if 'val_loss' in history.history.keys():
        val_loss = history.history['val_loss']
        plt.plot(epochs, val_loss, 'y', label='Validation loss')

    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    if savefig_flag is True:
        full_fig_name = os.path.join(fig_dir, f"{fig_name}.{fig_format}")
        plt.savefig(full_fig_name)
    
    if showfig_flag is True:
        plt.show()
    pass

def plot_accuracy(history, fig_dir: str, title: str, fig_name: str = None, fig_format: str = 'png', savefig_flag: bool = False, showfig_flag: bool = True) -> None:
    plt.clf()

    acc = history.history['accuracy']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'g', label='Training acc')

    if 'val_accuracy' in history.history.keys():
        val_loss = history.history['val_accuracy']
        plt.plot(epochs, val_loss, 'y', label='Validation accuracy')

    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    if savefig_flag is True:
        full_fig_name = os.path.join(fig_dir, f"{fig_name}.{fig_format}")
        plt.savefig(full_fig_name)
    
    if showfig_flag is True:
        plt.show()
    pass

# =========================================================================================== #
#                                   PLOT ROC CURVE                                            #
# =========================================================================================== #    
def plot_roc_curve(y_test, y_pred, fig_dir: str, title: str, fig_name: str = None, fig_format: str = 'png', savefig_flag: bool = False, showfig_flag: bool = True) -> None:
    global dark_ppt

    plt.clf()
    
    fpr_model, tpr_model, _ = roc_curve(y_test, y_pred) # thresholds_keras
    auc_model = auc(fpr_model, tpr_model)

    plt.figure(1)
    if dark_ppt is True:
        plt.plot([0, 1], [0, 1], 'w--')
    else:
        plt.plot([0, 1], [0, 1], 'k--')
    
    plt.plot(fpr_model, tpr_model, label='Net Model (area = {:.3f})'.format(auc_model))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(title)
    plt.legend(loc='best')
    
    if savefig_flag is True:
        full_fig_name = os.path.join(fig_dir, f"{fig_name}.{fig_format}")
        plt.savefig(full_fig_name)
    
    if showfig_flag is True:
        plt.show()
    
    return auc_model


# =========================================================================================== #
#                                   PLOT PRECISION-RECALL CURVE                               #
# =========================================================================================== #    
def plot_precision_recall_curve(y_test, y_pred, fig_dir: str, title: str, fig_name: str = None, fig_format: str = 'png', savefig_flag: bool = False, showfig_flag: bool = True) -> None:
    global dark_ppt

    plt.clf()
    
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    avg_precision_score = average_precision_score(y_test, y_pred)

    plt.figure(1)
    if dark_ppt is True:
        plt.plot([0, 1], [0.5, 0.5], 'w--')
    else:
        plt.plot([0, 1], [0.5, 0.5], '--')
    
    plt.plot(recall, precision, label='Precision-Recall curve (AP = {:.3f})'.format(avg_precision_score))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='best')
    
    if savefig_flag is True:
        full_fig_name = os.path.join(fig_dir, f"{fig_name}.{fig_format}")
        plt.savefig(full_fig_name)
    
    if showfig_flag is True:
        plt.show()
    
    return avg_precision_score

# =========================================================================================== #
#                                   PLOT CONFUSION MATRIX MODEL                               #
# =========================================================================================== #

def plot_confusion_matrix(y_test, y_pred, class_names, fig_dir: str, title: str, fig_name: str = None, fig_format: str = 'png', savefig_flag: bool = False, showfig_flag: bool = True):
    cm_title = f"{title}"
    ax, cm = _plot_confusion_matrix(
        y_test, y_pred,
        class_names,
        title=cm_title,
        fig_dir=fig_dir,
        fig_name=fig_name,
        savefig_flag=savefig_flag,
        fig_format=fig_format)

    cm_title = f"Normalized {title}"
    ax, cm_norm = _plot_confusion_matrix(
        y_test, y_pred,
        class_names,
        title=cm_title,
        normalize=True,
        fig_dir=fig_dir,
        fig_name=fig_name,
        savefig_flag=savefig_flag,
        fig_format=fig_format)

    if showfig_flag is True:
        plt.show()
    return cm, cm_norm

def _plot_confusion_matrix(y_true, y_pred, classes, fig_dir: str, title: str, fig_name: str = None, fig_format: str = 'png', normalize: bool = False, cmap=plt.cm.Blues, savefig_flag: bool = False) -> object:
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if savefig_flag is True:
        full_fig_name = os.path.join(fig_dir, f"{fig_name}.{fig_format}")
        plt.savefig(full_fig_name)
    return ax, cm

def plot_confidence_graph(predict, fig_dir: str, title: str, fig_name: str = None, fig_format: str = 'png', savefig_flag: bool = False, showfig_flag: bool = True):

    C_dfs = predict[predict['Label'] == 1]
    N_dfs = predict[predict['Label'] == 0]

    bins = np.linspace(0, 1, num=11)

    fig, ax = plt.subplots()
    ax.set_xticks(bins)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: "%.1f" % x))

    plt.hist([N_dfs['Prob'], C_dfs['Prob']], bins, histtype='bar', stacked=True,
         fill=True, label=['NotOnco','Onco'], edgecolor='black', linewidth=1.3, width=0.05, rwidth=0.5, align='mid')


    plt.legend(prop={'size': 9})
    plt.xlabel('Prediction scores')
    plt.ylabel('Number of samples')
    plt.title(f'{title}')

    full_name: str = os.path.join(fig_dir, fig_name)
    plt.savefig(f'{full_name}.{fig_format}')

    if showfig_flag is True:
        plt.show()

    pass