import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, RocCurveDisplay, f1_score, log_loss, mean_squared_error, zero_one_loss
import seaborn as sns
import math

def load_data(top_n=50, type=None, balanced=False, label_type='HIV_Binary'):
    if type:
        f_train = f"data/processed/{type}_train.csv"
        f_test = f"data/processed/{type}_test.csv"
        X_train = pd.read_csv(f_train, index_col=[0]).iloc[:, np.arange(top_n)]
        X_test = pd.read_csv(f_test, index_col=[0]).iloc[:, np.arange(top_n)]
    elif balanced:
        rna_train = pd.read_csv("data/processed/rna_train.csv", index_col=[0]).iloc[:, np.arange(top_n)]
        motif_train = pd.read_csv("data/processed/motif_train.csv", index_col=[0]).iloc[:, np.arange(top_n)]
        atac_train = pd.read_csv("data/processed/atac_train.csv", index_col=[0]).iloc[:, np.arange(top_n)]
        rna_test = pd.read_csv("data/processed/rna_test.csv", index_col=[0]).iloc[:, np.arange(top_n)]
        motif_test = pd.read_csv("data/processed/motif_test.csv", index_col=[0]).iloc[:, np.arange(top_n)]
        atac_test = pd.read_csv("data/processed/atac_test.csv", index_col=[0]).iloc[:, np.arange(top_n)]
        X_train = pd.concat([rna_train, motif_train, atac_train], axis=1)
        X_test = pd.concat([rna_test, motif_test, atac_test], axis=1)
    else:
        X_train = pd.read_csv("data/processed/overall_train.csv", index_col=[0]).iloc[:, np.arange(top_n)]
        X_test = pd.read_csv("data/processed/overall_test.csv", index_col=[0]).iloc[:, np.arange(top_n)]
    
    y_train = pd.read_csv(f"data/processed/{label_type}_train.csv", index_col = [0])
    y_test = pd.read_csv(f"data/processed/{label_type}_test.csv", index_col = [0])
    
    return X_train, X_test, y_train, y_test

def y_log_positive_transform(y_train, y_test):
        y_train['HIV_Float'] = y_train['HIV_Float'].apply(lambda x: math.log(x) if x > 0 else 0)
        y_test['HIV_Float'] = y_test['HIV_Float'].apply(lambda x: math.log(x) if x > 0 else 0)
        return y_train, y_test

def threshold(rf, quantile, X_train, X_test, Y_train, Y_test, y_hat_train, y_hat_test):
    value = y_hat_train['HIV_Binary'].quantile(quantile)
    train_index = y_hat_train['HIV_Binary'] >= value
    test_index = y_hat_test['HIV_Binary'] >= value
    X_train, X_test, y_train, y_test = X_train[train_index], X_test[test_index], y_train[train_index], y_test[test_index]

def overlay(y_train_true, y_train_pred, y_test_true, y_test_pred, title=None):
    sns.set_style("whitegrid")
    sns.set_context("talk")

    train_fpr, train_tpr, train_thresholds = roc_curve(y_train_true, y_train_pred)
    train_aucs = auc(train_fpr, train_tpr)

    test_fpr, test_tpr, test_thresholds = roc_curve(y_test_true, y_test_pred)
    test_aucs = auc(test_fpr, test_tpr)

    plt.figure()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    if title is None:
        plt.title(f"ROC Curve")
    else:
        plt.title(title)
    plt.plot(train_fpr, train_tpr,linestyle="--", marker=".", markersize=15, label = f'Train, AUC = {round(train_aucs, 2)}')
    plt.plot(test_fpr, test_tpr,linestyle="--", marker=".", markersize=15, label = f'Test, AUC = {round(test_aucs, 2)}')
    plt.plot([0, 1], [0, 1], linestyle="--", c="k")
    plt.legend()