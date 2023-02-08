import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from helpers import load_data


def corr_plot():
    X_train, X_test, y_train, y_test = load_data(
        top_n=10, balanced=True, label_type='HIV_Binary')
    X_features = X_train.columns

    # X_train = X_train.iloc[:,:10]
    corr = X_train.corr()
    sns.heatmap(corr,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values,
                cmap="YlGnBu")


def individual_roc_curves_for_top_n(X_features, X_train, X_test, y_train, y_test):
    n = len(X_train[0])
    aucs = []
    plt.figure()
    for i in range(n):
        y_hat = X_test[:, i]
        binary_metrics(
            y_test, y_hat, title=f"Overlay of individual ROC curves using the top {n}", binary=False)
        fpr, tpr, thresholds = roc_curve(y_test, y_hat)
        aucs.append(auc(fpr, tpr))
    plt.show()

    # plot only highest performing auc:
    plt.figure()
    max_index = aucs.index(max(aucs))
    y_hat = X_test[:, max_index]
    feature = X_features[max_index]

    binary_metrics(y_test, y_hat, title=f"{feature}", binary=False)

    return aucs
