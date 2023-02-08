import os
import pandas as pd
import numpy as np
import json
import time
import pathlib
import joblib
import gosdt.libgosdt as gosdt
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, precision_score, confusion_matrix
from gosdt.model.gosdt import GOSDT
from helpers import load_data
from tqdm import tqdm
from threshold_guess import *
from itertools import product


# -------------- HELPER FUNCTIONS --------------------

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

    sample_weight = [positive_weight if label==1 else 1 for label in y]
    return sample_weight

def oversample(X, y, prop_minority):
    ratio = prop_minority / (1 - prop_minority) 
    ros = RandomOverSampler(sampling_strategy=ratio, random_state=0)
    X_ros, y_ros = ros.fit_resample(X, y)
    return X_ros, y_ros

def config2name(path, n_features, n_est, max_depth, sample_weight, gosdt_depth, regularization, weight):
    name = '_'.join(['TG', str(n_features), str(n_est), str(max_depth), str(sample_weight), 'GOSDT', str(gosdt_depth), str(regularization), str(weight)]) +'.json'
    return os.path.join(path, name)

def name2config(name):
    tokens = name.split('_')
    d = {
        'n_features' : int(tokens[1][0]),
        'n_est' : [int(x) for x in tokens[2]],
        'max_depth' : [int(x) for x in tokens[3]],
        'sample_weight' : [float(x) for x in tokens[4]],
        'gosdt_depth' : [int(x) for x in tokens[6]],
        'regularization' : [float(x) for x in tokens[7]],
        'weight' : [float(x) for x in tokens[8]]
    }
    return d

def sweep(params, out_folder, out_csv):
    csv_path = os.path.join(out_folder, out_csv)
    with open(csv_path, 'w') as f:
        f.write('n_features,n_est,max_depth,sample_weight,gosdt_depth,regularization,weight,train_precision,train_recall,test_precision,test_recall,model_features' +'\n')

    # Load data
    n_features = params['n_features']
    X_train, X_test, y_train, y_test = load_data(n=n_features, label_type='10')
    y_train, y_test = y_train.values, y_test.values

    threshold_param_combinations = list(product(params['n_est'],
                                           params['max_depth'],
                                           params['sample_weight']))

    gosdt_param_combinations = list(product(params['gosdt_depth'],
                                       params['regularization'],
                                       params['weight']))

    for n_est, max_depth, sample_weight in tqdm(threshold_param_combinations):
        
        # Retrieve sample weights.
        if sample_weight == -1:
            # Default weights.
            weights = [1] * y_train.shape[0]
        else:
            # Case when minority class should get higher weights.
            weights = fetch_sample_weights(y_train, sample_weight)
        
        # guess thresholds
        X_train_threshold, thresholds, header, threshold_guess_time = compute_thresholds(X_train.copy(), y_train.copy(), n_est, max_depth, max_thresholds=30, weight=weights)
        X_test_threshold = cut(X_test.copy(), thresholds)
        X_test_threshold = X_test_threshold[header]
        print('--------------------')
        print(f'N_est: {n_est}, max_depth: {max_depth}, sample_weight: {sample_weight}')
        print("X:", X_train_threshold.shape)
        print("y:", y_train.shape)
        print('--------------------')

        for gosdt_depth, regularization, weight in tqdm(gosdt_param_combinations):
            if weight == -1:
                X_train_threshold_ros = X_train_threshold.copy()
                y_train_threshold_ros = y_train.copy()
            else:
                X_train_threshold_ros, y_train_threshold_ros = oversample(X_train_threshold.copy(), y_train.copy(), weight)
            
            print('--------------------')
            print(f'GOSDT_depth: {gosdt_depth}, regularization: {regularization}, weight: {weight}')
            print("X:", X_train_threshold_ros.shape)
            print("y:", y_train_threshold_ros.shape)
            print('--------------------')

            out_path = config2name(out_folder, n_features, n_est, max_depth, sample_weight, gosdt_depth, regularization, weight)
            config = {
                        "regularization": regularization,
                        "depth_budget": gosdt_depth,
                        "balance" : False,
                        "time_limit" : 400,
                        "model": out_path,
            }
            
            model = GOSDT(config)
            try:
                model.fit(X_train_threshold_ros, y_train_threshold_ros)
            except:
                print('COULD NOT FINISH GOSDT PROCESS')
                continue
            if model.time == -1:
                continue
            
            def strip(feature):
                indices = [feature.find('<'), feature.find('='), feature.find('>')]
                stop = min([x for x in indices if x > 0])
                return feature[:stop]
            model_features = '|'.join([strip(feature) for feature in model.tree.features()])
        
            def pr(y, y_hat):
                tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()
                precision = tp / (tp + fp + 0.0000001)
                recall = tp / (tp + fn + 0.0000001)
                return precision, recall

            y_hat_train, y_hat_test= model.predict(X_train_threshold), model.predict(X_test_threshold)
            train_precision, train_recall = pr(y_train, y_hat_train)
            test_precision, test_recall = pr(y_test, y_hat_test)

            data_row = [n_features, n_est, max_depth, sample_weight, gosdt_depth, regularization, weight, train_precision, train_recall, test_precision, test_recall, model_features]
            
            with open(csv_path, 'a') as f:
                f.write(','.join([str(x) for x in data_row]) + '\n')
            
    print('----------------------')
    print('FINISHED GOSDT SWEEP')
    print('----------------------')




if __name__ == "__main__":
    kParamPath = '/Users/caleb/Desktop/latency/gosdt_sweep_results/v0/gosdt_sweep_params.json'
    kOutFolder = '/Users/caleb/Desktop/latency/gosdt_sweep_results/v0/'
    kCsvName = 'sweep.csv'

    params = json.load(open(kParamPath, 'r'))
    sweep(params, kOutFolder, kCsvName)