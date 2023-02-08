import os
import pandas as pd
import numpy as np
import json
import time
import pathlib
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, precision_score, confusion_matrix
import gosdt.libgosdt as gosdt
from gosdt.model.gosdt import GOSDT
from helpers import load_data
from tqdm import tqdm
from threshold_guess import *


def threshold_sweep(minimum_depth, maximum_depth, minimum_n_est, maximum_n_est):
    X_train, X_test, y_train, y_test = load_data(n=30, label_type='10')
    y_train, y_test = y_train.values, y_test.values
    sample_weight = fetch_sample_weights(y_train)
    d = dict()
    colnames = ['n_est']
    for max_depth in range(minimum_depth, maximum_depth + 1):
        for weight in ['weights', 'no_weights']:
            colnames.append(f'{max_depth}_{weight}')
    f = open('thresholds.txt', 'w')
    f.write(', '.join(colnames) + '\n')

    for n_est in tqdm(range(minimum_n_est, maximum_n_est + 1)):
        row = [str(n_est)]
        for max_depth in range(minimum_depth, maximum_depth + 1):
            for weight in ['weights', 'no_weights']:
                if weight == 'no_weights':
                    X_train_threshold, thresholds, header, threshold_guess_time = compute_thresholds(
                        X_train.copy(), y_train.copy(), n_est, max_depth)
                else:
                    X_train_threshold, thresholds, header, threshold_guess_time = compute_thresholds(
                        X_train.copy(), y_train.copy(), n_est, max_depth, weight=sample_weight)
                row.append(str(X_train_threshold.shape[1]))
        f.write(', '.join(row) + '\n')
    f.close()


def fetch_sample_weights(y, desired_prop):
    # Given a desired proportion of weight of the positive class returns an array
    # with desired weights by sample.
    #
    # e.g. I want examples with label 1 to represent 20%, therefore I set desired_prop=2
    n_pos = sum(y)
    n_neg = y.shape[0] - n_pos
    prop_pos, prop_neg = n_pos / (n_pos + n_neg), n_neg / (n_pos + n_neg)

    desired_ratio = desired_prop / (1 - desired_prop)
    positive_weight = desired_ratio * prop_neg / prop_pos

    sample_weight = [positive_weight if label == 1 else 1 for label in y]
    return sample_weight
#
# LEVEL OF THRESHOLD_GUESS
# n_est = [25, 50, 100]
# max_depth = [1, 2]
# weight = [unweighted, 0.2, 0.3, 0.4, 0.5]

# LEVEL OF GOSDT
# Depth = [2, 3, 4]
# regularization = [0.0003, 0.0010, 0,0]
# weight = [.0977, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]


if __name__ == "__main__":
    # Read the dataset
    X_train, X_test, y_train, y_test = load_data(n=30, label_type='10')
    y_train, y_test = y_train.values, y_test.values

    # GBDT parameters for threshold and lower bound guesses
    # Tune to get out ~10 - 30
    n_est = 19
    max_depth = 2
    sample_weight = fetch_sample_weights(y_train, 0.4)
    max_thresholds = 20

    # guess thresholds
    X_train_threshold, thresholds, header, threshold_guess_time = compute_thresholds(
        X_train.copy(), y_train.copy(), n_est, max_depth, max_thresholds, weight=sample_weight)
    print("X:", X_train_threshold.shape)
    print("y:", y_train.shape)

    X_test_threshold = cut(X_test.copy(), thresholds)
    X_test_threshold = X_test_threshold[header]

    depth, regularization = 3, 0.003
    config = {
        "regularization": regularization,
        "depth_budget": depth,
        "balance": True,
        "time_limit": 400,
        "model": "/Users/caleb/Desktop/latency/model.json",
        "tree": "/Users/caleb/Desktop/latency/tree.json",
    }
    print(config)
    model = GOSDT(config)

    model.fit(X_train_threshold, y_train)

    y_hat_train, y_hat_test = model.predict(
        X_train_threshold), model.predict(X_test_threshold)
    conf_train, conf_test = model.tree.confidence(
        X_train_threshold), model.tree.confidence(X_test_threshold)
    y_conf_train = [confidence if label == 1 else 1 -
                    confidence for label, confidence, in zip(y_hat_train, conf_train)]
    y_conf_test = [confidence if label == 1 else 1 -
                   confidence for label, confidence, in zip(y_hat_test, conf_test)]
    overlay_pr_gosdt(y_train, y_test, y_conf_train, y_conf_test)
    plt.show()
    # print("evaluate the model, extracting tree and scores", flush=True)

    # get the results
    n_leaves = model.leaves()
    n_nodes = model.nodes()
    time = model.utime

    print("train acc:{}, test acc:{}".format(model.score(
        X_train_threshold, y_train), model.score(X_test_threshold, y_test)))
    print("train bacc:{}, test bacc:{}".format(balanced_accuracy_score(
        y_train, y_hat_train), balanced_accuracy_score(y_test, y_hat_test)))
    print("train precision:{}, test precision:{}".format(precision_score(
        y_train, y_hat_train), precision_score(y_test, y_hat_test)))

    tn, fp, fn, tp = confusion_matrix(y_train, y_hat_train).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print("train fp {}, fn {}".format(fp, fn))

    print("Model training time: {}".format(time))
    print("# of leaves: {}".format(n_leaves))
    print(model.tree)
    # print(model.tree.source)
