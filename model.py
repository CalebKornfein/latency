from helpers import load_data, overlay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from graphing_helpers import binary_metrics
from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm
import numpy as np
import pandas as pd

def fit_xgb(X_train, X_test, y_train, y_test):
    xg_clf = xgb.XGBClassifier(objective ='binary:logistic', verbosity=1)
    xg_clf.fit(X_train, y_train)

    y_hat_train = xg_clf.predict_proba(X_train)[:,1]
    y_hat_test = xg_clf.predict_proba(X_test)[:,1]
    plt.figure()
    binary_metrics(y_train, y_hat_train, title=f"XGBoost using top {len(X_train[0])} features, train", binary=False)
    plt.show()

    plt.figure()
    binary_metrics(y_test, y_hat_test, title=f"XGBoost using top {len(X_train[0])} features, test", binary=False)
    plt.show()

def fit_rf(X_features, X_train, X_test, y_train, y_test):
    n_features = X_train.shape[1]
    clf = RandomForestClassifier(bootstrap=False, max_depth=2, random_state=0, verbose=False)
    clf.fit(X_train, y_train)

    # feature importance by Gini importance
    sorted_idx = clf.feature_importances_.argsort()

    # top ten features by importance
    features, importances = np.array(X_features)[sorted_idx][n_features - 10: ], clf.feature_importances_[sorted_idx][n_features - 10: ]
    plt.figure()
    plt.barh(features, importances)
    plt.title("Top 10 features by Gini importance:")

    # get predicted probabilities for train and test and plot auc curves
    y_hat_train = clf.predict_proba(X_train)[:,1]
    y_hat_test = clf.predict_proba(X_test)[:,1]
    overlay(y_train, y_hat_train, y_test, y_hat_test)

    return clf


def individual_roc_curves_for_top_n(X_features, X_train, X_test, y_train, y_test):
    n = len(X_train[0])
    aucs = []
    plt.figure()
    for i in range(n):
        y_hat = X_test[:,i]
        binary_metrics(y_test, y_hat, title=f"Overlay of individual ROC curves using the top {n}", binary=False)
        fpr, tpr, thresholds = roc_curve(y_test, y_hat)
        aucs.append(auc(fpr, tpr))
    plt.show()

    # plot only highest performing auc:
    plt.figure()
    max_index = aucs.index(max(aucs))
    y_hat = X_test[:,max_index]
    feature = X_features[max_index]

    binary_metrics(y_test, y_hat, title=f"{feature}", binary=False)

    return aucs

#def rf_regression(X_features, X_train, X_test, y_train, y_test):
    


def main():
    # Load dataset for top 50 features in 80% train 20% test splits
    X_train, X_test, y_train, y_test = load_data()
    X_features = X_train.columns

    # try rf using top 60 features
    fit_rf(X_features, X_train, X_test, y_train, y_test)
    plt.show()

if __name__ == "__main__":
    main()



