import os
import pandas as pd
import numpy as np
import json
import time
import pathlib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, precision_score, confusion_matrix
import gosdt.libgosdt as gosdt
from gosdt.model.gosdt import GOSDT
from helpers import load_data
from tqdm import tqdm

# fit the tree using gradient boosted classifier
def fit_boosted_tree(X, y, n_est=10, lr=0.1, d=1, weight=None):
    clf = GradientBoostingClassifier(loss='deviance', learning_rate=lr, n_estimators=n_est, max_depth=d,
                                    random_state=42)
    clf.fit(X, y, sample_weight=weight)
    out = clf.score(X, y)
    return clf, out

# perform cut on the dataset
def cut(X, ts):
    df = X.copy()
    colnames = X.columns
    for j in range(len(ts)):
        for s in range(len(ts[j])):
            X[colnames[j]+'<='+str(ts[j][s])] = 1
            k = df[colnames[j]] > ts[j][s]
            X.loc[k, colnames[j]+'<='+str(ts[j][s])] = 0
        X = X.drop(colnames[j], axis=1)
    return X


# compute the thresholds
def get_thresholds(X, y, n_est, lr, d, weight=None, backselect=True):
    # got a complaint here...
    y = np.ravel(y)
    # X is a dataframe
    clf, out = fit_boosted_tree(X, y, n_est, lr, d, weight)
    #print('acc:', out, 'acc cv:', score.mean())
    thresholds = []
    for j in range(X.shape[1]):
        tj = np.array([])
        for i in range(len(clf.estimators_)):
            f = clf.estimators_[i,0].tree_.feature
            t = clf.estimators_[i,0].tree_.threshold
            tj = np.append(tj, t[f==j])
        tj = np.unique(tj)
        thresholds.append(tj.tolist())

    X_new = cut(X, thresholds)
    clf1, out1 = fit_boosted_tree(X_new, y, n_est, lr, d, weight)
    #print('acc','1:', out1, 'acc1 cv:', scorep.mean())

    outp = 1
    Xp = X_new.copy()
    clfp = clf1
    itr=0
    if backselect:
        while outp >= out1 and itr < X_new.shape[1]-1:
            vi = clfp.feature_importances_
            if vi.size > 0:
                c = Xp.columns
                i = np.argmin(vi)
                Xp = Xp.drop(c[i], axis=1)
                clfp, outp = fit_boosted_tree(Xp, y, n_est, lr, d, weight)
                itr += 1
            else:
                break
        Xp[c[i]] = X_new[c[i]]
        #_, _ = fit_boosted_tree(Xp, y, n_est, lr, d)

    h = Xp.columns
    #print('features:', h)
    return Xp, thresholds, h

# compute the thresholds
def compute_thresholds(X, y, n_est, max_depth, weight=None) :
    # n_est, max_depth: GBDT parameters
    # set LR to 0.1
    lr = 0.1
    start = time.perf_counter()
    X, thresholds, header = get_thresholds(X, y, n_est, lr, max_depth, weight, backselect=True)
    guess_time = time.perf_counter()-start

    return X, thresholds, header, guess_time

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

    for n_est in tqdm(range(minimum_n_est,maximum_n_est + 1)):
        row = [str(n_est)]
        for max_depth in range(minimum_depth, maximum_depth + 1):
            for weight in ['weights', 'no_weights']:
                if weight == 'no_weights':
                    X_train_threshold, thresholds, header, threshold_guess_time = compute_thresholds(X_train.copy(), y_train.copy(), n_est, max_depth)
                else:
                    X_train_threshold, thresholds, header, threshold_guess_time = compute_thresholds(X_train.copy(), y_train.copy(), n_est, max_depth, weight=sample_weight)
                row.append(str(X_train_threshold.shape[1]))
        f.write(', '.join(row) + '\n')
    f.close()

def fetch_sample_weights(y):
    N_pos = y.sum()
    N_neg = y.shape[0] - N_pos
    # ratio to weight positive samples at given prevalance of negative samples
    ratio = N_neg / N_pos
    sample_weight = [ratio if label==1 else 1 for label in y]
    return sample_weight

if __name__ == "__main__":
    # Read the dataset
    X_train, X_test, y_train, y_test = load_data(n=30, label_type='10')
    y_train, y_test = y_train.values, y_test.values

    # GBDT parameters for threshold and lower bound guesses
    # Tune to get out ~10 - 30
    n_est = 19
    max_depth = 2
    sample_weight = fetch_sample_weights(y_train)

    # guess thresholds
    X_train_threshold, thresholds, header, threshold_guess_time = compute_thresholds(X_train.copy(), y_train.copy(), n_est, max_depth, weight=sample_weight)
    print("X:", X_train_threshold.shape)
    print("y:", y_train.shape)

    X_test_threshold = cut(X_test.copy(), thresholds)
    X_test_threshold = X_test_threshold[header]

    depth, regularization = 6, 0.0003
    config = {
                "regularization": regularization,
                "depth_budget": depth,
                "balance" : Fals,
                "time_limit" : 8000,
                "model": "/Users/caleb/Desktop/latency/model.json",
                "tree": "/Users/caleb/Desktop/latency/tree.json",
        }
    print(config)
    model = GOSDT(config)
    
    model.fit(X_train_threshold, pd.DataFrame(y_train))

    y_hat_train, y_hat_test= model.predict(X_train_threshold), model.predict(X_test_threshold)
    conf_train, conf_test = model.tree.confidence(X_train_threshold), model.tree.confidence(X_test_threshold)
    y_conf_train = [confidence if label==1 else 1 - confidence for label, confidence, in zip(y_hat_train, conf_train)]
    y_conf_test = [confidence if label==1 else 1 - confidence for label, confidence, in zip(y_hat_test, conf_test)]
    overlay_pr_gosdt(y_train, y_test, y_conf_train, y_conf_test)
    plt.show()
    # print("evaluate the model, extracting tree and scores", flush=True)

    # get the results
    n_leaves = model.leaves()
    n_nodes = model.nodes()
    time = model.utime
    
    print("train acc:{}, test acc:{}".format(model.score(X_train_threshold, y_train), model.score(X_test_threshold, y_test)))
    print("train bacc:{}, test bacc:{}".format(balanced_accuracy_score(y_train, y_hat_train), balanced_accuracy_score(y_test, y_hat_test)))
    print("train precision:{}, test precision:{}".format(precision_score(y_train, y_hat_train), precision_score(y_test, y_hat_test)))
    
    tn, fp, fn, tp = confusion_matrix(y_train, y_hat_train).ravel()
    print("train fp {}, fn {}".format(fp, fn))
    
    print("Model training time: {}".format(time))
    print("# of leaves: {}".format(n_leaves))
    print(model.tree)
    # print(model.tree.source)








# guess thresholds
print("X:", X_train.shape)
print("y:", y_train.shape)
X_train_threshold, thresholds, header, threshold_guess_time = compute_thresholds(X_train.copy(), y_train.copy(), n_est, max_depth)
print("X:", X_train_threshold.shape)
print("y:", y_train.shape)

X_test_threshold = cut(X_test.copy(), thresholds)
X_test_threshold = X_test_threshold[header]

path = 'gosdt_wb/thresholds.json'
def save_thresholds(thresholds, path):
    json_obj = json.dumps(thresholds)
    with open(path, 'w') as out:
        out.write(json_obj)
save_thresholds(thresholds, path)

def load_thresholds(path):
    return json.load(open(path, 'r'))['thresholds']

# guess lower bound
start_time = time.perf_counter()
clf = GradientBoostingClassifier(n_estimators=n_est, max_depth=max_depth, random_state=42) # can directly fit on X_train
clf.fit(X_train_threshold, y_train.values.flatten())
warm_labels = clf.predict(X_train_threshold)
elapsed_time = time.perf_counter() - start_time
lb_time = elapsed_time

# save the labels from lower bound guesses as a tmp file and return the path to it.
labelsdir = pathlib.Path('./data/warm_lb_labels')
labelsdir.mkdir(exist_ok=True, parents=True)
labelpath = labelsdir / 'warm_label.tmp'
labelpath = str(labelpath)
pd.DataFrame(warm_labels, columns=['10']).to_csv(labelpath, header='10',index=None)


# train GOSDT model
# depth budget and regularition.
# should return within seconds or minutes
# can tune regulation less to get more complicated
# depth budget more
config = {
            "regularization": 0.01,
            "depth_budget": 3,
            "reference_LB": True,
            "warm_LB" : True,
            "path_to_labels" : labelpath,
            "balance" : True,
            "time_limit" : 120,
            # "non_binary" : False,
            # "diagnostics" : True,
# changing objective 
            "objective" : "bacc",
            "verbose" : True,
            # "model" : "model.json",
            # "tree" : "tree.json"
}

model = GOSDT(config)

model.fit(X_train_threshold, y_train)

print("evaluate the model, extracting tree and scores", flush=True)

# get the results
train_acc = model.score(X_train_threshold, y_train)
n_leaves = model.leaves()
n_nodes = model.nodes()
time = model.utime

print("Model training time: {}".format(time))
print("Training accuracy: {}".format(train_acc))
print("# of leaves: {}".format(n_leaves))
print(model.tree)