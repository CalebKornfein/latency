import os
import pandas as pd
import numpy as np
import time
import pathlib
from sklearn.ensemble import GradientBoostingClassifier
from helpers import load_data

# shenanigans to successfully import gosdt
curr = os.getcwd()
os.chdir('/Users/caleb/')
import gosdt.libgosdt as gosdt
from gosdt.model.gosdt import GOSDT
from gosdt.model.threshold_guess import compute_thresholds
os.chdir(curr)

# Read the dataset
X_train, X_test, y_train, y_test, X_new_train, y_new_test = load_data(label_type='HIV_Top_10')
X_train, X_test = X_train.iloc[:,:30], X_test.iloc[:,:30]
X_features = X_train.columns

# GBDT parameters for threshold and lower bound guesses
n_est = 40
max_depth = 1

# guess thresholds
print("X:", X_train.shape)
print("y:", y_train.shape)
X_train, thresholds, header, threshold_guess_time = compute_thresholds(X_train, y_train, n_est, max_depth)
y_train = pd.DataFrame(y_train, columns=['HIV_Top_10'])
print("X:", X_train.shape)
print("y:", y_train.shape)

# guess lower bound
start_time = time.perf_counter()
clf = GradientBoostingClassifier(n_estimators=n_est, max_depth=max_depth, random_state=42)
clf.fit(X_train, y_train.values.flatten())
warm_labels = clf.predict(X_train)
elapsed_time = time.perf_counter() - start_time
lb_time = elapsed_time

# save the labels from lower bound guesses as a tmp file and return the path to it.
labelsdir = pathlib.Path('./data/warm_lb_labels')
labelsdir.mkdir(exist_ok=True, parents=True)
labelpath = labelsdir / 'warm_label.tmp'
labelpath = str(labelpath)
pd.DataFrame(warm_labels, columns=['HIV_Top_25']).to_csv(labelpath, header='HIV_Top_25',index=None)


# train GOSDT model
config = {
            "regularization": 0.01,
            "depth_budget": 5,
            "reference_LB": True,
            "warm_LB" : True,
            "path_to_labels" : labelpath,
            "balance" : True,
            "time_limit" : 600,
            "non_binary" : False,
            "diagnostics" : True,
            "objective" : "bacc",
        }

config = {
            "regularization": 0.001,
            "depth_budget": 3,
            "reference_LB": False,
            "warm_LB" : False,
            "balance" : True,
            "time_limit" : 10000,
            "non_binary" : False,
            "diagnostics" : True,
            "look_ahead": True,
            "objective" : "bacc",
        }

model = GOSDT(config)

model.fit(X_train, y_train)

print("evaluate the model, extracting tree and scores", flush=True)

# get the results
train_acc = model.score(X_train, y_train)
n_leaves = model.leaves()
n_nodes = model.nodes()
time = model.utime

print("Model training time: {}".format(time))
print("Training accuracy: {}".format(train_acc))
print("# of leaves: {}".format(n_leaves))
print(model.tree)