from helpers import load_data, y_log_positive_transform
from graphing_helpers import overlay_roc_temp, overlay_pr_temp
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import xgboost as xgb
from graphing_helpers import binary_metrics
from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm
import numpy as np
import pandas as pd
import math

def fit_xgb(X_train, X_test, y_train, y_test, X_new_train, y_new_test, params=None):
    if params:
        clf = xgb.XGBClassifier(**params)
    else:
        clf = xgb.XGBClassifier(objective = 'binary:logistic', max_depth=2, colsample_bytree=0.5, scale_pos_weight=10)
    clf.fit(X_train, y_train)

    overlay_roc_temp(clf, X_train, X_test, y_train, y_test, X_new_train, y_new_test)
    overlay_pr_temp(clf, X_train, X_test, y_train, y_test, X_new_train, y_new_test)
    plt.show()

def fit_rf_classifier(X_features, X_train, X_test, y_train, y_test, X_new_train, y_new_test):
    n_features = X_train.shape[1]
    clf = RandomForestClassifier(class_weight='balanced', max_depth=2, random_state=0, verbose=False)
    clf.fit(X_train, y_train)

    # feature importance by Gini importance
    sorted_idx = clf.feature_importances_.argsort()
    features = X_train.columns.values
    # top ten features by importance
    features, importances = np.array(features)[sorted_idx][n_features - 10: ], clf.feature_importances_[sorted_idx][n_features - 10: ]
    plt.figure()
    plt.barh(features, importances)
    plt.title("Top 10 features by Gini importance:")
    overlay_pr_temp(clf, X_train, X_test, y_train, y_test, X_new_train, y_new_test)
    overlay_roc_temp(clf, X_train, X_test, y_train, y_test, X_new_train, y_new_test)

    return clf

def threshold(quantile, y_hat_train, y_hat_test):
    value = y_hat_train['HIV_Binary'].quantile(quantile)
    train_predicted_zero_index = (y_hat_train['HIV_Binary'] < value)
    test_predicted_zero_index = (y_hat_test['HIV_Binary'] < value)
    return train_predicted_zero_index, test_predicted_zero_index

def rf_regression(X_features, X_train, X_test, y_train, y_test):
    n_features = X_train.shape[1]
    clf = RandomForestRegressor(bootstrap=False, max_depth=2, random_state=0, verbose=False)
    clf.fit(X_train, y_train)

    # feature importance by Gini importance
    sorted_idx = clf.feature_importances_.argsort()

    # top ten features by importance
    features, importances = np.array(X_features)[sorted_idx][n_features - 10: ], clf.feature_importances_[sorted_idx][n_features - 10: ]
    plt.figure()
    plt.barh(features, importances)
    plt.title("Top 10 features by Gini importance:")

def add_type_column(X_train, X_test):
    observation_dict = {'1' : 'DMSO',
                    '2' : 'iBET151',
                    '3' : 'Prostratin',
                    '4' : 'SAHA'}
    X_train['type'] = [observation_dict[x[-1]] for x in X_train.index]
    X_test['type'] = [observation_dict[x[-1]] for x in X_test.index]
    return X_train, X_test

def main():
    X_train, X_test, y_train, y_test, X_new_train, y_new_test = load_data(label_type='HIV_Top_25')
    params = {'objective' : 'binary:logistic', 
             'max_depth' : 2,
             'colsample_bytree' : 0.5,
             'scale_pos_weight' : 2}
    fit_xgb(X_train, X_test, y_train, y_test, X_new_train, y_new_test, params=params)

    params = {'objective' : 'binary:logistic', 
             'max_depth' : 2,
             'colsample_bytree' : 0.5,
             'scale_pos_weight' : 5}
    fit_xgb(X_train, X_test, y_train, y_test, X_new_train, y_new_test, params=params)

    X_train, X_test, y_train, y_test, X_new_train, y_new_test = load_data(label_type='HIV_Top_10')
    params = {'objective' : 'binary:logistic', 
            'max_depth' : 2,
            'colsample_bytree' : 0.5,
            'scale_pos_weight' : 5,
            }
    fit_xgb(X_train, X_test, y_train, y_test, X_new_train, y_new_test, params=params)

    X_train, X_test, y_train, y_test, X_new_train, y_new_test = load_data(label_type='HIV_Top_5')
    params = {'objective' : 'binary:logistic', 
        'max_depth' : 2,
        'colsample_bytree' : 0.5,
        'scale_pos_weight' : 2}
    fit_xgb(X_train, X_test, y_train, y_test, X_new_train, y_new_test, params=params)

    # Load dataset for top 50 features in 80% train 20% test splits
    X_train, X_test, y_train, y_test = load_data(label_type='HIV_Top_25')
    X_features = X_train.columns

    # try rf using top 50 features
    clf = fit_rf_classifier(X_features, X_train, X_test, y_train, y_test)

    y_hat_train = pd.DataFrame(clf.predict_proba(X_train)[:,1], columns = ['HIV_Binary'])
    y_hat_test = pd.DataFrame(clf.predict_proba(X_test)[:,1], columns = ['HIV_Binary'])
    
    quantile = y_train.value_counts()[0] / y_train.shape[0]
    train_predicted_zero_index, test_predicted_zero_index = threshold(quantile, y_hat_train, y_hat_test)
    train_predicted_zero_index, test_predicted_zero_index = threshold(.01, y_hat_train, y_hat_test)



    plt.show()

if __name__ == "__main__":
    main()



