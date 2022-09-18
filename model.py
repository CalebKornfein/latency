from helpers import load_data, create_train_test_split, top_n_X
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from metrics import binary_metrics
from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm
import numpy as np

def fit_xgb(X_train, X_test, y_train, y_test):

    xg_clf = xgb.XGBClassifier(objective ='binary:logistic', verbosity=1)
    xg_clf.fit(X_train, y_train)
    y_hat = xg_clf.predict_proba(X_test)[:,1]

    binary_metrics(y_test, y_hat, title=f"XGBoost using top {len(X_train[0])} features", binary=False)
    plt.show()

def calc_avg_auc_looped_xgb(X_train, X_test, y_train, y_test):
    n = len(X_train[0])
    aucs = 0
    store = []

    for i in tqdm(range(n)):
        xg_clf = xgb.XGBClassifier(objective ='binary:logistic', verbosity=0)
        xg_clf.fit(np.vstack(X_train[:,i]), np.vstack(y_train))
        y_hat = xg_clf.predict_proba(X_test[:,i])[:,1]
        fpr, tpr, thresholds = roc_curve(y_test, y_hat)
        aucs += auc(fpr, tpr)
        store.append(auc(fpr, tpr))
    avg_auc = aucs / n
    print(f"AVERAGE AUC FOR SINGLE FEATURE XGB MODELS ACROSS TOP {n} FEATURES: {avg_auc}")
    return store

def fit_rf(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(bootstrap=False, random_state=0, verbose=False)
    clf.fit(X_train, y_train)
    y_hat = clf.predict_proba(X_test)[:,1]
    
    binary_metrics(y_test, y_hat, title="RF", binary=False)
    plt.show()

def individual_roc_curves_for_top_n(X_train, X_test, y_train, y_test):
    n = len(X_train[0])
    aucs = []
    for i in range(n):
        y_hat = X_test[:,i]
        binary_metrics(y_test, y_hat, title=f"Overlay of individual ROC curves using the top {n}", binary=False)
        fpr, tpr, thresholds = roc_curve(y_test, y_hat)
        aucs.append(auc(fpr, tpr))
    plt.show()
    return aucs

def main():

    # load data and create 80% train 20% test splits
    X, y = load_data(type='HIV_Binary')
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)
    
    # only use top 60 features
    X_train, X_test = top_n_X(X_train, X_test, 60)

    # try XGBoost
    #fit_xgb(X_train, X_test, y_train, y_test)

    # calculate average auc using single feature xgboost
    #calc_avg_auc_looped_xgb(X_train, X_test, y_train, y_test)

    # try rf using top 60 features
    fit_rf(X_train, X_test, y_train, y_test)
    plt.show()

if __name__ == "__main__":
    main()



