import json
import os
from gosdt.model.gosdt import GOSDT
import numpy as np
import pandas as pd
from gosdt_sweep import config2name
from threshold_guess import cut
from helpers import load_data


def trace(gosdt):
    # Iterates through the model JSON in pre-order traversal and scrapes the paths,
    # returning a list of tuples representing the boolean conditions necessary to reach
    # that leaf, and the last value is the prediction.
    def walk(node, current_path):
        print(current_path)

        if 'prediction' in node:
            current_path.append((node['prediction']))
            lp.append(current_path.copy())
            current_path.pop()
            return

        current_path.append((node['name'], 1))
        walk(node['true'], current_path)
        current_path.pop()

        current_path.append((node['name'], 0))
        walk(node['false'], current_path)
        current_path.pop()
        return

    dt = gosdt.tree.__repr__()
    lp = []
    walk(dt, [])

    return lp


def calc_leave_accuracies(leave_paths, X_test, y_test, sweep_row):

    def predicate(row, path):
        for feature, value in path:
            if row[feature] != value:
                return False
        return True

    total_correct = 0

    for path in leave_paths:
        prediction = path[-1]
        path = path[:-1]

        correct = 0
        n = 0

        for i, row in X_test.iterrows():
            if predicate(row, path):
                if int(y_test.iloc[i]) == prediction:
                    correct += 1
                    if prediction == 1:
                        total_correct += 1
                n += 1

        print(f"PREDICTED {prediction}", path,
              f"ACCURACY: {correct / n}, N = {n}")

    print(
        f"FOUND OVERALL RECALL: {total_correct / sum(y_test)} VS SAVED RECALL: {sweep_row['test_recall']}")


def search_for_good_gosdt_model(dir, verbose=False):
    sweep = pd.read_csv(os.path.join(dir, "sweep.csv"))

    idx = np.where((sweep['test_precision'] > 0.35)
                   & (sweep['test_recall'] > 0.1))

    if verbose:
        for i in idx[0]:
            row = sweep.iloc[i]
            print(f"---------- {i} ---------")
            print(f"test recall: {row.test_recall}")
            print(f"test precision: {row.test_precision}")
            print(row.model_features)

    # Good indices:
    # 1613 --> (P/R) was 0.356, 0.142
    # 1743 --> (P/R) was 0.270, 0.255


def fetch_gosdt_tree(sweep, i, dir, X_test, y_test):
    print(f'------- fetching {i} ----------')
    row = sweep.iloc[i]
    print(row)

    # Find the file paths for the relevant threshold guess and gosdt model
    tg_name = f"THRESHOLD_GUESS_NF_{int(row['n_features'])}_MT_{int(row['max_thresholds'])}_NEST_{int(row['n_est'])}_MD_{int(row['max_depth'])}_SW_{row['sample_weight']}.json"
    relevant_threshold_guess = os.path.join(dir,
                                            'threshold_guess',
                                            tg_name)
    gosdt_name = config2name(dir + '/gosdt', int(row['n_features']), int(row['max_thresholds']), int(row['n_est']), int(row['max_depth']),
                             row['sample_weight'], int(
                                 row['gosdt_depth']), row['regularization'],
                             row['weight'])
    relevant_gosdt = gosdt_name + ".json"

    print('-------')
    print(relevant_threshold_guess)
    print(relevant_gosdt)
    print('-------')

    # load the appropriate items
    threshold_guess = json.load(open(relevant_threshold_guess, "r"))

    X_test_tg = cut(X_test.copy(), threshold_guess['thresholds'])[
        threshold_guess['header']]
    X_test_tg = X_test_tg.reset_index(drop=True)

    gosdt = GOSDT()
    gosdt.load(relevant_gosdt)

    lp = trace(gosdt)
    calc_leave_accuracies(lp, X_test_tg, y_test, row)


def main():
    kSweepPath = 'gosdt_sweep_results/v3/sweep.csv'
    kDir = 'gosdt_sweep_results/v3'
    sweep = pd.read_csv(kSweepPath)

    X_train, X_test, y_train, y_test = load_data(n=1000, label_type='10')
    fetch_gosdt_tree(sweep, 1613, kDir, X_test, y_test)
    fetch_gosdt_tree(sweep, 1743, kDir, X_test, y_test)


if __name__ == "__main__":
    main()
