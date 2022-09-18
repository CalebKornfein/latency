import csv
import pandas as pd
import os
import json

from sklearn.model_selection import train_test_split

def generate_dataset_given_feature_names(feature_names, out):
    data = []
    columns = []

    for file in ['data/rna.mat/rna.mat.csv', 'data/motif.mat.csv']:
        with open(file, 'r') as f:    
            
            reader = csv.DictReader(f)
            for row in reader:

                if row[''] in feature_names:
                    data_row = [float(v) if not v == row[''] else v for k, v in row.items()]
                    feature = data_row[0]
                    vals = data_row[1:]
                    
                    columns.append(feature)
                    data.append(vals)

    
    df = pd.DataFrame(data).T
    df.columns = columns
    df.to_csv(out, index=False)

def convert_label_dict_values_to_list(labels):
    return list(labels.values())

def load_labels(type='HIV_Float', list_val = True):
    labels = json.load(open('hiv.json', 'r'))[type]
    if list_val:
        labels = convert_label_dict_values_to_list(labels)
    return labels

def load_data(**kwargs):
    X = pd.read_csv('multiomics.csv').to_numpy()
    y = load_labels(**kwargs)
    return X, y

def create_train_test_split(X, y, random_state = 0, prop_train=0.80):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = random_state, train_size = prop_train)
    return X_train, X_test, y_train, y_test

def top_n_X(X_train, X_test, n):
    new_X_train, new_X_test = X_train[:, :n], X_test[:, :n]
    return new_X_train, new_X_test
