from random import randrange
from scipy import stats
import pandas as pd
import csv
import numpy as np
from tqdm import tqdm
import json
import os
from sklearn.model_selection import train_test_split

def fetch_train_test_indices(n_samples, save=True):
    train_indices, test_indices = train_test_split(np.arange(n_samples), test_size=0.2, random_state=0)
    if save:
        with open('data/processed/train_test_indices.json', 'w') as f:
            indices = {'train' : train_indices.tolist(),
                        'test'  : test_indices.tolist()}
            f.write(json.dumps(indices))
    return train_indices, test_indices

def split_train_test(train_indices, test_indices, train_out, test_out, feature_names):
    feature_set = set(feature_names)
    train_data = []
    test_data = []
    columns = []
    rows = list(json.load(open('data/processed/hiv.json', 'r'))['HIV_Float'].keys())
    train_rows = [rows[x] for x in train_indices]
    test_rows = [rows[x] for x in test_indices]

    for file in [('data/base/rna.mat/rna.mat.csv'), ('data/base/motif.mat.csv'), ('data/base/atac.mat/atac.csv')]:
        with open(file, 'r') as f:
            reader = csv.reader(f)
            next(reader)

            for row in tqdm(reader):
                feature = row[0]
                if feature in feature_set:
                    columns.append(feature)
                    vals = [float(x) for x in row[1:]]

                    train_vals = [vals[x] for x in train_indices]
                    train_data.append(train_vals)

                    test_vals = [vals[x] for x in test_indices]
                    test_data.append(test_vals)
    
    train = pd.DataFrame(train_data).T
    train.columns = columns
    train.index = train_rows
    
    test = pd.DataFrame(test_data).T
    test.columns = columns
    test.index = test_rows

    # Reorder datasets such that they remain sorted by descending abs spearman coefficient
    train = train[feature_names]
    test = test[feature_names]

    train.to_csv(train_out)
    test.to_csv(test_out)

def split_label(train_indices, test_indices):
    hiv = json.load(open('data/processed/hiv.json', 'r'))
    for label_type in ['HIV_Float', 'HIV_Binary']:
        labels = list(hiv[label_type].values())
        train = pd.DataFrame([labels[x] for x in train_indices])
        test = pd.DataFrame([labels[x] for x in test_indices])
        train.columns = [label_type]
        test.columns = [label_type]
        train.index = train_indices
        test.index = test.indices
        train.to_csv(f"data/processed/{label_type}_train.csv")
        test.to_csv(f"data/processed/{label_type}_test.csv")

def calculate_spearman(train_indices):
    data = []
    columns = ['feature', 'dataset', 'spearman', 'abs_spearman', 'pvalue']
    hiv = list(json.load(open('data/processed/hiv.json', 'r'))["HIV_Float"].values())
    hiv_train = [hiv[x] for x in train_indices]

    for file, dataset in [('data/base/rna.mat/rna.mat.csv', 'rna'), ('data/base/motif.mat.csv', 'motif'), ('data/base/atac.mat/atac.csv', 'atac')]:
        with open(file, 'r') as f:
            reader = csv.reader(f)
            next(reader)

            for row in reader:
                feature = row[0]
                vals = [float(x) for x in row[1:]]
                train_vals = [vals[x] for x in train_indices]
                spearman = stats.spearmanr(hiv_train, train_vals)
                
                data_row = [feature, dataset, spearman.correlation, abs(spearman.correlation), spearman.pvalue]

                data.append(data_row)

    df = pd.DataFrame(data, columns=columns).sort_values(by='abs_spearman', ascending=False)
    df.to_csv('data/processed/spearman.csv', index=False)
    print("Wrote spearman.csv")

def select_top_features(n=250, type=None):
    df = pd.read_csv('data/processed/spearman.csv')
    if type == None:
        top_features = df.head(n)
    else:
        top_features = df[df['dataset'] == type].head(n)
    
    top_features = list(top_features.feature.values)
    return top_features

if __name__ == "__main__":
    # 1) Generate train and test indices
    n_samples = 61708
    fetch_train_test_indices(n_samples, save=True)

    # 2) Calculate spearman coeffecients for all features against HIV using the train and test indices
    indices = json.load(open('data/processed/train_test_indices.json', 'r'))
    train_indices, test_indices = indices['train'], indices['test']
    calculate_spearman(train_indices, test_indices)

    # 2) Select features (ranked by absolute spearman correlation) for:
    # -- Top 250 overall
    # -- Top 50 RNA
    # -- Top 50 Motif
    # -- Top 50 ATAC
    overall = select_top_features(250)
    rna = select_top_features(50, 'rna')
    motif = select_top_features(50, 'motif')
    atac = select_top_features(50, 'atac')

    # 3) Generate a train and test dataset for each
    datasets = [('overall', overall),
                ('rna', rna),
                ('motif', motif),
                ('atac', atac)]
    
    for data_type, feature_names in datasets:
        train_out = os.path.join('data', 'processed', data_type + "_train.csv")
        test_out = os.path.join('data', 'processed', data_type + "_test.csv")
        split_train_test(train_indices, test_indices, train_out, test_out, feature_names)
