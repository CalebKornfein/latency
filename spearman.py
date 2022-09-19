from scipy import stats
from helpers import load_labels, generate_dataset_given_feature_names
import pandas as pd
import csv
import scipy.sparse as sp
import numpy as np
from tqdm import tqdm

def calculate_spearman():
    data = []
    columns = ['feature', 'dataset', 'spearman', 'abs_spearman', 'pvalue']
    hiv = load_labels()

    for file, dataset in [('data/rna.mat/rna.mat.csv', 'rna'), ('data/motif.mat.csv', 'motif')]:
        with open(file, 'r') as f:
            reader = csv.reader(f)
            next(reader)

            for row in reader:
                feature = row[0]
                vals = [float(x) for x in row[1:]]

                spearman = stats.spearmanr(hiv, vals)
                
                data_row = [feature, dataset, spearman.correlation, abs(spearman.correlation), spearman.pvalue]

                data.append(data_row)
    
    # load atac data and feature names
    atac = sp.load_npz('data/atac.mat/atac.mat.npz')

    f = open('data/atac.mat/atac.mat.rownames.txt', 'r')
    atac_features = [line.strip() for line in f.readlines()]

    for row in tqdm(range(atac.shape[0])):
        vals = np.array(atac[row, ].todense()).flatten()
        feature = atac_features[row]

        spearman = stats.spearmanr(hiv, vals)

        data_row = [feature, 'atac', spearman.correlation, abs(spearman.correlation), spearman.pvalue]
        data.append(data_row)

    df = pd.DataFrame(data, columns=columns).sort_values(by='abs_spearman', ascending=False)
    df.to_csv('spearman.csv', index=False)
    print("Wrote spearman.csv")

def select_top_features(n=200, type=None):
    df = pd.read_csv('spearman.csv')
    if type == None:
        top_features = df.head(n)
    else:
        top_features = df[df['dataset'].isin(type)].head(n)
    
    top_features = set(top_features.feature.values)
    return top_features

if __name__ == "__main__":
    # 1) Calculate spearman coeffecients for all features against HIV
    calculate_spearman()

    # 2) Choose the top 200 features to model with the highest absolute spearman correlation to model with
    feature_names = select_top_features(1000)

    # 3) Generate a dataset of exclusively these top 200 features
    generate_dataset_given_feature_names(feature_names, 'multiomics.csv')
