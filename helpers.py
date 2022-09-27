from winreg import REG_NOTIFY_CHANGE_LAST_SET
import pandas as pd
import os
import json
import numpy as np

def load_data(top_n=60, type=None, balanced=False, label_type='HIV_Binary'):
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

    return X_train, X_test
        
def convert_label_dict_values_to_list(labels):
    return list(labels.values())

def load_labels(label_type='HIV_Float', list_val = True):
    labels = json.load(open('processed_data/hiv.json', 'r'))[label_type]
    if list_val:
        labels = list(labels.values())
    return labels

def load_data(features=None, **kwargs):
    df = pd.read_csv('processed_data/multiomics.csv')
    if features != None:
        df = df[features]

    X = df.to_numpy()
    y = load_labels(**kwargs)
    return X, y

def top_n_X(X_train, X_test, n):
    new_X_train, new_X_test = X_train[:, :n], X_test[:, :n]
    return new_X_train, new_X_test