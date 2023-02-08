import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, RocCurveDisplay, f1_score, log_loss, mean_squared_error, zero_one_loss
import seaborn as sns
import math
import json

# def load_data(top_n=30, type=None, balanced=False, label_type='HIV_Binary'):
#     if type:
#         f_train = f"data/processed/{type}_train.csv"
#         f_test = f"data/processed/{type}_test.csv"
#         X_train = pd.read_csv(f_train, index_col=[0])
#         X_test = pd.read_csv(f_test, index_col=[0])
#     elif balanced:
#         rna_train = pd.read_csv("data/processed/rna_train.csv", index_col=[0])
#         motif_train = pd.read_csv("data/processed/motif_train.csv", index_col=[0])
#         atac_train = pd.read_csv("data/processed/atac_train.csv", index_col=[0])
#         rna_test = pd.read_csv("data/processed/rna_test.csv", index_col=[0])
#         motif_test = pd.read_csv("data/processed/motif_test.csv", index_col=[0])
#         atac_test = pd.read_csv("data/processed/atac_test.csv", index_col=[0])
#         X_train = pd.concat([rna_train, motif_train, atac_train], axis=1)
#         X_test = pd.concat([rna_test, motif_test, atac_test], axis=1)
#     else:
#         X_train = pd.read_csv("data/processed/overall_train.csv", index_col=[0])
#         X_test = pd.read_csv("data/processed/overall_test.csv", index_col=[0])
#         X_new = pd.read_csv("data/rna.csv", index_col=[0])

#     # Select top n and filter out all Mitochondrial features
#     filter = [x for x in X_train.columns if "MT-" not in x][:top_n]
#     X_train, X_test = X_train[filter], X_test[filter]
#     X_new = X_new[filter]

#     y_train = pd.read_csv(f"data/processed/HIV_train.csv", index_col = [0])[label_type]
#     y_test = pd.read_csv(f"data/processed/HIV_test.csv", index_col = [0])[label_type]
#     y_new_test = pd.read_csv(f"data/HIV.csv", index_col = [0])[label_type]

#     return X_train, X_test, y_train, y_test, X_new, y_new_test


def load_data(n=30, version='combined', label_type='10'):
    X_train = pd.read_csv(
        f"data/processed/{version}/overall_train.csv", index_col=[0])
    X_train = X_train[X_train.columns[:n]]
    X_test = pd.read_csv(
        f"data/processed/{version}/overall_test.csv", index_col=[0])
    X_test = X_test[X_test.columns[:n]]

    y_train = pd.read_csv(
        f"data/processed/{version}/HIV_train.csv", index_col=[0])[label_type]
    y_test = pd.read_csv(
        f"data/processed/{version}/HIV_test.csv", index_col=[0])[label_type]
    return X_train, X_test, y_train, y_test


def overlay(y_train_true, y_train_pred, y_test_true, y_test_pred, title=None):
    sns.set_style("whitegrid")
    sns.set_context("talk")

    train_fpr, train_tpr, train_thresholds = roc_curve(
        y_train_true, y_train_pred)
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
    plt.plot(train_fpr, train_tpr, linestyle="--", marker=".",
             markersize=15, label=f'Train, AUC = {round(train_aucs, 2)}')
    plt.plot(test_fpr, test_tpr, linestyle="--", marker=".",
             markersize=15, label=f'Test, AUC = {round(test_aucs, 2)}')
    plt.plot([0, 1], [0, 1], linestyle="--", c="k")
    plt.legend()


def fetch_observation_type(X_train, X_test, y_train, y_test, type):
    observation_dict = {'1': 'DMSO',
                        '2': 'iBET151',
                        '3': 'Prostratin',
                        '4': 'SAHA'}
    link_observations = [observation_dict[key[-1]]
                         for key in json.load(open('data/processed/hiv.json', 'r'))['HIV_Float'].keys()]


def add_label(label_name, f, version='combined'):
    tr, te = f'data/processed/{version}/HIV_train.csv',  f'data/processed/{version}/HIV_test.csv'

    y_train = pd.read_csv(tr, index_col=[0])
    y_test = pd.read_csv(te, index_col=[0])

    train, test = f(y_train, y_test)
    y_train[label_name], y_test[label_name] = train, test

    y_train.to_csv(tr)
    y_test.to_csv(te)


def quantize(y_train, y_test):
    q1, q2, q3 = y_train['HIV_Float'].quantile([0.25, 0.5, 0.75])

    def to_quantiles(x):
        if x <= q1:
            return '1'
        elif x <= q2:
            return '2'
        elif x <= q3:
            return '3'
        else:
            return '4'

    train = y_train.apply(lambda x: to_quantiles(x.float), axis=1).values
    test = y_test.apply(lambda x: to_quantiles(x.float), axis=1).values
    return train, test


def top_n_percent(y_train, y_test):
    split_value = y_train['float'].quantile(0.9)
    print(f"Split if HIV > {split_value}")

    def to_quantiles(x):
        if x <= split_value:
            return '0'
        else:
            return '1'

    train = y_train.apply(lambda x: to_quantiles(x.float), axis=1).values
    test = y_test.apply(lambda x: to_quantiles(x.float), axis=1).values
    return train, test


def y_log_positive_transform(y_train, y_test):
    y_train['HIV_Float'] = y_train['HIV_Float'].apply(lambda x: math.log(x+1))
    y_test['HIV_Float'] = y_test['HIV_Float'].apply(lambda x: math.log(x+1))
    return y_train, y_test
