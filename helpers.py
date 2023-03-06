import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import math
import json
from collections import namedtuple


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


def fetch_person_index(y_train_index, y_test_index):
    # Observations with index <= 61708 come from subject 1, while observations with
    # index > 61708 come from subject 2.
    def filter_p1(y):
        return [True if y_i < 61708 else False for y_i in y]

    def filter_p2(y):
        return [True if y_i >= 61708 else False for y_i in y]
    train_index_p1, test_index_p1 = filter_p1(
        y_train_index), filter_p1(y_test_index)
    train_index_p2, test_index_p2 = filter_p2(
        y_train_index), filter_p2(y_test_index)
    return train_index_p1, test_index_p1, train_index_p2, test_index_p2


def split_data_by_person(X_train, X_test, y_train, y_test):
    train_index_p1, test_index_p1, train_index_p2, test_index_p2 = fetch_person_index(
        y_train.index, y_test.index)
    personal_data = namedtuple(
        "personal_data", ["X_train", "X_test", "y_train", "y_test"])
    p1 = personal_data(X_train[train_index_p1], X_test[test_index_p1],
                       y_train[train_index_p1], y_test[test_index_p1])
    p2 = personal_data(X_train[train_index_p2], X_test[test_index_p2],
                       y_train[train_index_p2], y_test[test_index_p2])
    return p1, p2


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


def top_10_percent(y_train, y_test):
    y = pd.concat([y_train, y_test])
    split_value = y.float.quantile(0.9)
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
