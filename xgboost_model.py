import os
from helpers import fetch_person_index, load_data, split_data_by_person
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve, precision_recall_curve, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import shuffle
from tqdm import tqdm
import numpy as np
import random
import time
import json
import pandas as pd

def fetch_aupr(clf, X_train, X_test, y_train, y_test):
    y_hat_train = clf.predict_proba(X_train)[:, 1]
    y_hat_test = clf.predict_proba(X_test)[:, 1]
    train_pr, train_re, train_thresholds = precision_recall_curve(
        y_train, y_hat_train)
    test_pr, test_re, test_thresholds = precision_recall_curve(
        y_test, y_hat_test)
    train_aupr = auc(train_re, train_pr)
    test_aupr = auc(test_re, test_pr)
    return train_aupr, test_aupr

def fetch_sample_weights_ratio(y, desired_prop):
    # Given a desired proportion of weight of the positive class returns an array
    # with desired weights by sample.
    #
    # e.g. I want examples with label 1 to represent 20%, therefore I set desired_prop=0.2
    #
    # -1 indicates no weight change.
    if desired_prop == -1:
        return 1

    n_pos = sum(y)
    n_neg = y.shape[0] - n_pos
    prop_pos, prop_neg = n_pos / (n_pos + n_neg), n_neg / (n_pos + n_neg)

    desired_ratio = desired_prop / (1 - desired_prop)
    positive_weight = desired_ratio * prop_neg / prop_pos

    return positive_weight

def xgboost_sweep(params, X_train, y_train, path=None, n_iter=10):
    start = time.time()
    clf = xgb.XGBClassifier()
    
    sweep_params = params.copy()
    # Convert from proportions in config json to actual multiple for sample weight.
    sweep_params["scale_pos_weight"] = [fetch_sample_weights_ratio(y_train, prop) for prop in params["scale_pos_weight"]]

    grid_search = RandomizedSearchCV(clf, sweep_params, n_iter=n_iter, cv=5, scoring='average_precision', refit=True, n_jobs=4, random_state=0, verbose=1)
    grid_search.fit(X_train, y_train)
    end = time.time()
    print(f"Grid search took {round(end - start, 1)} seconds")

    if path:
        grid_search.best_estimator_.save_model(path)
    return grid_search.best_estimator_


def main():
    out = 'xgb_results/v2'
    kParamPath = os.path.join(out, 'xgboost_parameters.json')

    X_train, X_test, y_train, y_test = load_data(n=1000, label_type='10')
    params = json.load(open(kParamPath))
    
    # Overall
    clf_overall = xgboost_sweep(params, X_train, y_train, path=os.path.join(out, 'xgboost_model_overall.json'), n_iter=100)

    # Trained on person 1
    p1, p2 = split_data_by_person(X_train, X_test, y_train, y_test)
    clf_p1 = xgboost_sweep(params, p1.X_train, p1.y_train, path=os.path.join(out, 'xgboost_model_p1.json'), n_iter=100)

    # Trained on person 2
    clf_p2 = xgboost_sweep(params, p2.X_train, p2.y_train, path=os.path.join(out, 'xgboost_model_p2.json'), n_iter=10)


if __name__ == "__main__":
    main()
