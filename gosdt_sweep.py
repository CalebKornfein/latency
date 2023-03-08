import os
import pandas as pd
import numpy as np
import json
import time
import pathlib
import multiprocessing as mp
import gosdt.libgosdt as gosdt
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, precision_score, confusion_matrix
from gosdt.model.gosdt import GOSDT
from helpers import load_data, fetch_person_index
from threshold_guess import *
from itertools import product


# -------------- HELPER FUNCTIONS --------------------

def fetch_sample_weights(y, desired_prop):
    # Given a desired proportion of weight of the positive class returns an array
    # with desired weights by sample.
    #
    # e.g. I want examples with label 1 to represent 20%, therefore I set desired_prop=0.2
    n_pos = sum(y)
    n_neg = y.shape[0] - n_pos
    prop_pos, prop_neg = n_pos / (n_pos + n_neg), n_neg / (n_pos + n_neg)

    desired_ratio = desired_prop / (1 - desired_prop)
    positive_weight = desired_ratio * prop_neg / prop_pos

    sample_weight = [positive_weight if label == 1 else 1 for label in y]
    return sample_weight


def oversample(X, y, prop_minority):
    ratio = prop_minority / (1 - prop_minority)
    ros = RandomOverSampler(sampling_strategy=ratio, random_state=0)
    X_ros, y_ros = ros.fit_resample(X, y)
    return X_ros, y_ros


def config2name(path, n_features, max_thresholds, n_est, max_depth, sample_weight, gosdt_depth, regularization, weight):
    name = '_'.join(['TG', str(n_features), str(max_thresholds), str(n_est), str(max_depth), str(
        sample_weight), 'GOSDT', str(gosdt_depth), str(regularization), str(weight)])
    return os.path.join(path, name)


def threshold_guess_(n_features, max_thresholds, n_est, max_depth, sample_weight, out):
    # Load the data.
    X_train, X_test, y_train, y_test = load_data(n=n_features, label_type='10')
    y_train, y_test = y_train.values, y_test.values

    # Retrieve sample weights.
    if sample_weight == -1:
        # Default weights.
        weights = [1] * y_train.shape[0]
    else:
        # Weighted scenario.
        weights = fetch_sample_weights(y_train, sample_weight)

    # Guess thresholds.
    X_train_threshold, thresholds, header, threshold_guess_time = compute_thresholds(
        X_train.copy(), y_train.copy(), n_est, max_depth, max_thresholds=max_thresholds, weight=weights)

    # Save the thresholds and header.
    json_ = {
        'thresholds': thresholds,
        'header': header.tolist(),
        'threshold_guess_time': threshold_guess_time,
        'max_thresholds': max_thresholds,
        'n_est': n_est,
        'n_features': n_features,
        'max_depth': max_depth,
        'sample_weight': sample_weight,
    }
    with open(out, "w") as outfile:
        json.dump(json_, outfile, indent=4)


def gosdt_(X_train, y_train, X_test, y_test, y_train_index, y_test_index, n_features, max_thresholds, n_est, max_depth, sample_weight, gosdt_depth, regularization, weight, out):
    if weight == -1:
        X_train_threshold_ros = X_train.copy()
        y_train_threshold_ros = y_train.copy()
    else:
        X_train_threshold_ros, y_train_threshold_ros = oversample(
            X_train.copy(), y_train.copy(), weight)

    print('--------------------')
    print(
        f'GOSDT_depth: {gosdt_depth}, regularization: {regularization}, weight: {weight}')
    print("X:", X_train_threshold_ros.shape)
    print("y:", y_train_threshold_ros.shape)
    print('--------------------')

    config = {
        "regularization": regularization,
        "depth_budget": gosdt_depth,
        "balance": False,
        "time_limit": 400,
        "model": out + '.json',
    }

    model = GOSDT(config)
    try:
        model.fit(X_train_threshold_ros, y_train_threshold_ros)
    except:
        print('COULD NOT FINISH GOSDT PROCESS')
        return None
    if model.time == -1:
        print('EXCEEDED TIME LIMIT')
        return None

    # Process the model and emit outputs.
    def strip(feature):
        indices = [feature.find('<'), feature.find(
            '='), feature.find('>')]
        stop = min([x for x in indices if x > 0])
        return feature[:stop]

    model_features = '|'.join([strip(feature)
                              for feature in model.tree.features()])

    def metrics(y, y_hat):
        tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()
        precision = tp / (tp + fp + 0.000000001)
        recall = tp / (tp + fn + 0.000000001)
        fpr = fp / (fp + tn + 0.000000001)
        return precision, recall, fpr

    y_hat_train, y_hat_test = model.predict(
        X_train), model.predict(X_test)
    
    train_index_p1, test_index_p1, train_index_p2, test_index_p2 = fetch_person_index(
        y_train_index, y_test_index)

    # Overall Results
    train_precision_overall, train_recall_overall, train_fpr_overall = metrics(y_train, y_hat_train)
    test_precision_overall, test_recall_overall, test_fpr_overall = metrics(y_test, y_hat_test)

    # Results for person 1.
    train_precision_p1, train_recall_p1, train_fpr_p1 = metrics(
        y_train[train_index_p1], y_hat_train[train_index_p1])
    test_precision_p1, test_recall_p1, test_fpr_p1 = metrics(
        y_test[test_index_p1], y_hat_test[test_index_p1])

    # Results for person 2.
    train_precision_p2, train_recall_p2, train_fpr_p2 = metrics(
        y_train[train_index_p2], y_hat_train[train_index_p2])
    test_precision_p2, test_recall_p2, test_fpr_p2 = metrics(
        y_test[test_index_p2], y_hat_test[test_index_p2])

    first_row = """n_features,max_thresholds,n_est,max_depth,sample_weight,gosdt_depth,regularization,weight,
                train_precision,train_recall,train_fpr,test_precision,test_recall,test_fpr,
                train_precision_p1,train_recall_p1,train_fpr_p1,test_precision_p1,test_recall_p1,test_fpr_p1,
                train_precision_p2,train_recall_p2,train_fpr_p2,test_precision_p2,test_recall_p2,test_fpr_p2,
                model_features""" + '\n'

    data_row = [n_features, max_thresholds, n_est, max_depth, sample_weight,
                gosdt_depth, regularization, weight,
                train_precision_overall, train_recall_overall, train_fpr_overall, test_precision_overall, test_recall_overall, test_fpr_overall,
                train_precision_p1, train_recall_p1, train_fpr_p1, test_precision_p1, test_recall_p1, test_fpr_p1,
                train_precision_p2, train_recall_p2, train_fpr_p2, test_precision_p2, test_recall_p2, test_fpr_p2,
                model_features]

    with open(out + '.csv', 'w') as outfile:
        outfile.write(first_row)
        outfile.write(','.join([str(x) for x in data_row]) + '\n')


def threshold_guess_stage(threshold_guess_params, out_folder, workers):
    threshold_guess_param_combinations = list(product(threshold_guess_params['n_features'],
                                                      threshold_guess_params['max_thresholds'],
                                                      threshold_guess_params['n_est'],
                                                      threshold_guess_params['max_depth'],
                                                      threshold_guess_params['sample_weight'],
                                                      ))

    args = [[n_features, max_thresholds, n_est, max_depth, sample_weight,
             os.path.join(out_folder, f'THRESHOLD_GUESS_NF_{n_features}_MT_{max_thresholds}_NEST_{n_est}_MD_{max_depth}_SW_{sample_weight}.json')]
            for n_features, max_thresholds, n_est, max_depth, sample_weight in threshold_guess_param_combinations]

    with mp.Pool(workers) as pool:
        pool.starmap(threshold_guess_, args)

    print('--------------------')
    print(f'FINISHED THRESHOLD GUESS STAGE')
    print('--------------------')


def gosdt_stage(params, threshold_guess_folder, out, workers):
    threshold_guess_params = params['threshold_guess']
    threshold_guess_param_combinations = list(product(threshold_guess_params['n_features'],
                                                      threshold_guess_params['max_thresholds'],
                                                      threshold_guess_params['n_est'],
                                                      threshold_guess_params['max_depth'],
                                                      threshold_guess_params['sample_weight'],
                                                      ))
    gosdt_params = params['gosdt']
    gosdt_param_combinations = list(product(gosdt_params['gosdt_depth'],
                                            gosdt_params['regularization'],
                                            gosdt_params['weight'],
                                            ))

    for i, (n_features, max_thresholds, n_est, max_depth, sample_weight) in enumerate(threshold_guess_param_combinations):
        X_train, X_test, y_train, y_test = load_data(
            n=n_features, label_type='10')
        y_train_index, y_test_index = y_train.index, y_test.index
        y_train, y_test = y_train.values, y_test.values

        # Find the relevant thresholds and header and cut the data.
        threshold_path = f'THRESHOLD_GUESS_NF_{n_features}_MT_{max_thresholds}_NEST_{n_est}_MD_{max_depth}_SW_{sample_weight}.json'
        full_threshold_path = os.path.join(
            threshold_guess_folder, threshold_path)
        threshold_guess_data = json.load(open(full_threshold_path, "r"))
        thresholds, header = threshold_guess_data['thresholds'], threshold_guess_data['header']

        X_train_threshold = cut(X_train.copy(), thresholds)[header]
        X_test_threshold = cut(X_test.copy(), thresholds)[header]

        args = [[X_train_threshold, y_train, X_test_threshold,
                 y_test, y_train_index, y_test_index, n_features,
                 max_thresholds, n_est, max_depth, sample_weight,
                 gosdt_depth, regularization, weight,
                 config2name(out, n_features, max_thresholds, n_est, max_depth, sample_weight, gosdt_depth, regularization, weight)]
                for j, (gosdt_depth, regularization, weight) in enumerate(gosdt_param_combinations)]

        with mp.Pool(workers) as pool:
            pool.starmap(gosdt_, args)

    print('--------------------')
    print(f'FINISHED GOSDT STAGE')
    print('--------------------')


def main(params, out_folder, out_csv, workers=1):
    threshold_guess_params = params['threshold_guess']
    gosdt_params = params['gosdt']

    # Prep out paths
    kThresholdGuessOutFolder = os.path.join(kOutFolder, 'threshold_guess')
    os.makedirs(kThresholdGuessOutFolder, exist_ok=True)
    kGosdtOutFolder = os.path.join(kOutFolder, 'gosdt')
    os.makedirs(kGosdtOutFolder, exist_ok=True)

    # 1 - Threshold guess
    threshold_guess_stage(threshold_guess_params,
                          kThresholdGuessOutFolder, workers)

    # 2 - GOSDT
    gosdt_stage(params, kThresholdGuessOutFolder, kGosdtOutFolder, workers)

    # Clean up and stitch together the outputs.
    threshold_guess_param_combinations = list(product(threshold_guess_params['n_features'],
                                                      threshold_guess_params['max_thresholds'],
                                                      threshold_guess_params['n_est'],
                                                      threshold_guess_params['max_depth'],
                                                      threshold_guess_params['sample_weight'],
                                                      ))
    gosdt_param_combinations = list(product(gosdt_params['gosdt_depth'],
                                            gosdt_params['regularization'],
                                            gosdt_params['weight'],
                                            ))

    cumulative_results = []
    for i, (n_features, max_thresholds, n_est, max_depth, sample_weight) in enumerate(threshold_guess_param_combinations):
        for j, (gosdt_depth, regularization, weight) in enumerate(gosdt_param_combinations):
            try:
                gosdt_path = config2name(kGosdtOutFolder, n_features, max_thresholds, n_est,
                                         max_depth, sample_weight, gosdt_depth, regularization, weight) + '.csv'
                results = pd.read_csv(gosdt_path)
                cumulative_results.append(results.iloc[0].values)
            except:
                ("FAILED")

    final_results = pd.DataFrame(cumulative_results)
    final_results.columns = ['n_features', 'max_thresholds', 'n_est', 'max_depth',
                             'sample_weight', 'gosdt_depth', 'regularization', 'weight',
                             'train_precision', 'train_recall', 'test_precision', 'test_recall',
                             'train_precision_p1', 'train_recall_p1', 'test_precision_p1', 'test_recall_p1',
                             'train_precision_p2', 'train_recall_p2', 'test_precision_p2', 'test_recall_p2',
                             'model_features']

    csv_path = os.path.join(out_folder, out_csv)
    final_results.to_csv(csv_path, index=False)

    print('----------------------')
    print('FINISHED GOSDT SWEEP')
    print('----------------------')


if __name__ == "__main__":
    kParamPath = '/Users/caleb/Desktop/latency/gosdt_sweep_results/v4/gosdt_sweep_params.json'
    kOutFolder = '/Users/caleb/Desktop/latency/gosdt_sweep_results/v4/'
    kCsvName = 'sweep.csv'
    kNumWorkers = 2

    params = json.load(open(kParamPath, 'r'))
    main(params, kOutFolder, kCsvName, kNumWorkers)
