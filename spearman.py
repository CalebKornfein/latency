from scipy import stats
from helpers import load_labels, generate_dataset_given_feature_names
import pandas as pd
import csv

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
    
    df = pd.DataFrame(data, columns=columns)
    df.to_csv('spearman.csv', index=False)

def select_top_features(n=200):
    df = pd.read_csv('spearman.csv').sort_values(by='abs_spearman', ascending=False)
    top_features = df.head(n)
    top_features = set(top_features.feature.values)
    return top_features

if __name__ == "__main__":
    # 1) Calculate spearman coeffecients for all features against HIV
    #calculate_spearman()

    # 2) Choose the top 200 features to model with the highest absolute spearman correlation to model with
    feature_names = select_top_features(200)

    # 3) Generate a dataset of exclusively these top 200 features
    generate_dataset_given_feature_names(feature_names, 'multiomics.csv')
