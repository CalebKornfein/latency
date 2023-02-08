from random import randrange
from scipy import stats
import pandas as pd
import csv
import numpy as np
from tqdm import tqdm
import json
import os
from sklearn.model_selection import train_test_split


def scrape_hiv():
    for v in ['v1', 'v2', 'combined']:
        file = f'data/base/{v}/rna.mat.csv'
        rows = from_txt(f'data/base/{v}/rna.mat.rownames.txt', numerical=False)
        with open(file, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)
            for row in rows:
                line = next(reader)
                if row == 'HIV':
                    with open(f'data/base/{v}/hiv.txt', 'w') as out:
                        out.write('\n'.join([v for v in line]))
                    break


def from_txt(f, numerical=True):
    file = open(f, 'r')
    lines = [line.strip() for line in file.readlines()]
    if not numerical:
        return lines
    return [float(line) for line in lines]


def fetch_train_test_indices(n_samples, save=True):
    train_indices, test_indices = train_test_split(
        np.arange(n_samples), test_size=0.2, random_state=0)
    if save:
        with open('data/processed/train_test_indices.json', 'w') as f:
            indices = {'train': train_indices.tolist(),
                       'test': test_indices.tolist()}
            f.write(json.dumps(indices))
    return train_indices, test_indices


def load(f):
    file = open(f, 'r')
    reader = csv.reader(file, delimiter=',')
    first = next(reader)
    second = next(reader)
    print(len(first))
    print(len(second))
    print("First", first[0], first[-1])
    print("Second", second[0], second[-1])


def create_merged_motif():
    # Observation names
    columns = from_txt('data/base/v1/rna.mat.colnames.txt', False) + \
        from_txt('data/base/v2/rna.mat.colnames.txt', False)

    v1 = from_txt('data/base/v1/motif.mat.rownames.txt', numerical=False)
    v2 = from_txt('data/base/v2/motif.mat.rownames.txt', numerical=False)

    with open('data/base/combined/motif.mat.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(columns)

        r1 = csv.reader(open('data/base/v1/motif.mat.csv'))
        r2 = csv.reader(open('data/base/v2/motif.mat.csv'))
        next(r1)
        next(r2)

        for f1, f2 in tqdm(zip(v1, v2)):
            val1 = next(r1)
            val2 = next(r2)

            if f1 != f2:
                print(f"BREAKING,{f1} != {f2}")
                return

            row = [float(x) for x in (val1 + val2)]
            writer.writerow(row)


def create_merged_rna():
    # merge v1 and v2 data on overlapping features (inner join)

    columns = from_txt('data/base/v1/rna.mat.colnames.txt', False) + \
        from_txt('data/base/v2/rna.mat.colnames.txt', False)

    r1 = pd.read_csv('data/base/v1/rna.mat.csv')
    v1 = from_txt('data/base/v1/rna.mat.rownames.txt', numerical=False)
    r1.index = v1
    r1 = r1.T

    r2 = pd.read_csv('data/base/v2/rna.mat.csv')
    v2 = from_txt('data/base/v2/rna.mat.rownames.txt', numerical=False)
    r2.index = v2
    r2 = r2.T

    combined = pd.concat([r1, r2], join='inner')

    # transpose back
    combined = combined.T

    with open('data/base/combined/rna.mat.rownames.txt', 'w') as f:
        for line in list(combined.index):
            f.write(f"{line}\n")

    with open('data/base/combined/rna.mat.colnames.txt', 'w') as f:
        for line in list(combined.columns):
            f.write(f"{line}\n")

    combined.to_csv('data/base/combined/rna.mat.csv', index=False)


def calculate_spearman(version, train_indices):
    data = []
    columns = ['feature', 'dataset', 'spearman', 'abs_spearman', 'pvalue']

    hiv = from_txt(f'data/base/{version}/hiv.txt', numerical=True)
    if version == 'v1':
        print("converting train indices to v1")
        train_indices = [index for index in train_indices if index < 61708]
    elif version == 'v2':
        print("converting train indices to v2")
        train_indices = [
            index - 61708 for index in train_indices if index - 61708 >= 0]
    else:
        print("unrecognized version")
        return

    hiv_train = [hiv[x] for x in train_indices]

    for dataset in ['atac', 'rna', 'motif']:
        file = f'data/base/{version}/{dataset}.mat.csv'
        row_file = f'data/base/{version}/{dataset}.mat.rownames.txt'
        with open(file, 'r') as f:
            rows = from_txt(row_file, numerical=False)
            row_index = 0

            reader = csv.reader(f)
            next(reader)

            for row in tqdm(reader):
                # ATAC in V1 is the one dataset that has rownames written
                if (version == 'v1') and (dataset == 'atac'):
                    row = row[1:]

                vals = [float(x) for x in row]
                train_vals = [vals[x] for x in train_indices]
                spearman = stats.spearmanr(hiv_train, train_vals)
                feature = rows[row_index]
                data_row = [feature, dataset, spearman.correlation,
                            abs(spearman.correlation), spearman.pvalue]

                row_index += 1
                data.append(data_row)

    df = pd.DataFrame(data, columns=columns).sort_values(
        by='abs_spearman', ascending=False)
    df.to_csv(f'data/processed/spearman_check_v1.csv', index=False)
    print(f"Wrote spearman {version}.csv")


def calculate_combined_spearman(train_indices):
    data = []
    columns = ['feature', 'dataset', 'spearman', 'abs_spearman', 'pvalue']

    hiv = from_txt('data/base/v1/hiv.txt', numerical=True) + \
        from_txt('data/base/v2/hiv.txt', numerical=True)
    hiv_train = [hiv[x] for x in train_indices]

    for dataset in ['rna', 'motif']:
        file = f'data/base/combined/{dataset}.mat.csv'
        row_file = f'data/base/combined/{dataset}.mat.rownames.txt'
        with open(file, 'r') as f:
            rows = from_txt(row_file, numerical=False)
            row_index = 0

            reader = csv.reader(f)
            next(reader)

            for row in tqdm(reader):
                vals = [float(x) for x in row]
                train_vals = [vals[x] for x in train_indices]
                spearman = stats.spearmanr(hiv_train, train_vals)
                feature = rows[row_index]
                data_row = [feature, dataset, spearman.correlation,
                            abs(spearman.correlation), spearman.pvalue]

                row_index += 1
                data.append(data_row)

    df = pd.DataFrame(data, columns=columns).sort_values(
        by='abs_spearman', ascending=False)
    df.to_csv(f'data/processed/spearman_combined.csv', index=False)
    print(f"Wrote spearman combined .csv")


def load_train_test_indices(version='combined'):
    indices = json.load(open('data/processed/train_test_indices.json', 'r'))
    if version == 'combined':
        train, test = indices["train"], indices["test"]
    elif version == 'v1':
        train = [x for x in indices["train"] if x < 61708]
        test = [x for x in indices["test"] if x < 61708]
    elif version == 'v2':
        train = [x - 61708 for x in indices["train"] if x - 61708 >= 0]
        test = [x - 61708 for x in indices["test"] if x - 61708 >= 0]
    else:
        print("TYPE NOT FOUND")
        return
    return train, test


def select_top_features(n=50, type=None, version='combined'):
    df = pd.read_csv(f'data/processed/{version}/spearman_{version}.csv')
    features = list(df.feature)
    datasets = list(df.dataset)

    def include(feature, dataset):
        if 'MT' in feature:
            return False
        if 'HIV' in feature:
            return False
        if type and dataset != type:
            return False
        return True

    return [feature for feature, dataset in zip(features, datasets) if include(feature, dataset)][:n]


def split_labels(train_indices, test_indices, version):
    labels = from_txt(f'data/base/{version}/hiv.txt', numerical=True)
    train = pd.DataFrame([labels[x] for x in train_indices])
    test = pd.DataFrame([labels[x] for x in test_indices])
    train.columns = ['float']
    test.columns = ['float']
    train.index = train_indices
    test.index = test_indices
    train.to_csv(f"data/processed/{version}/HIV_train.csv")
    test.to_csv(f"data/processed/{version}/HIV_test.csv")


def split_train_test(train_out, test_out, feature_names, version):
    feature_set = set(feature_names)
    train_data = []
    test_data = []
    columns = []
    rows = from_txt(
        f'data/base/{version}/rna.mat.colnames.txt', numerical=False)
    train_indices, test_indices = load_train_test_indices(version=version)
    train_rows = [rows[x] for x in train_indices]
    test_rows = [rows[x] for x in test_indices]

    for type in ['rna', 'motif']:
        file = f'data/base/{version}/{type}.mat.csv'
        with open(file, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            features = from_txt(
                f'data/base/{version}/{type}.mat.rownames.txt', numerical=False)
            index = 0

            for i in tqdm(range(len(features))):
                row = next(reader)
                feature = features[index]
                if feature in feature_set:
                    columns.append(feature)
                    vals = [float(x) for x in row]

                    train_vals = [vals[x] for x in train_indices]
                    train_data.append(train_vals)

                    test_vals = [vals[x] for x in test_indices]
                    test_data.append(test_vals)
                index += 1

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


if __name__ == "__main__":
    # 1) Generate train and test indices
    n_samples_v1 = 61708
    n_samples_v2 = 30831
    train_indices, test_indices = train_test_split(
        np.arange(n_samples_v1 + n_samples_v2), test_size=0.2, random_state=0)

    # 2) Generate merged dataset
    # note -- ATAC not merged as features not aligned
    create_merged_rna()
    create_merged_motif()

    # 3) Find HIV values to calculate spearman against
    scrape_hiv()

    # 4) Calculate spearman coefficients for all features against HIV using the train data
    calculate_spearman('v1', train_indices)
    calculate_spearman('v2', train_indices)
    calculate_combined_spearman(train_indices)

    # 5) Select features (ranked by absolute spearman correlation) for:
    # -- Top 250 overall
    # -- Top 50 RNA
    # -- Top 50 Motif
    # -- Top 50 ATAC
    #
    # Then, write the datasets to a csv

    for version in ['v1', 'v2']:
        overall = select_top_features(100, version=version)
        # rna = select_top_features(50, type='rna', version=version)
        # motif = select_top_features(50, type='motif', version=version)
        # atac = select_top_features(50, type='atac', version=version)
        train, test = load_train_test_indices(version=version)
        split_labels(train, test, version=version)

        datasets = [('overall', overall)]
        for data_type, feature_names in datasets:
            train_out = os.path.join(
                'data', 'processed', version, data_type + "_train.csv")
            test_out = os.path.join(
                'data', 'processed', version, data_type + "_test.csv")
            split_train_test(train_out, test_out, feature_names, version)


# ------------------------- Old Functions---------------------------------------------
def create_merged_rna_old():
    # Observation names
    columns = from_txt('data/base/v1/rna.mat.colnames.txt', False) + \
        from_txt('data/base/v2/rna.mat.colnames.txt', False)

    # Handle RNA
    v1 = from_txt('data/base/v1/rna.mat.rownames.txt', numerical=False)
    v2 = from_txt('data/base/v2/rna.mat.rownames.txt', numerical=False)
    common = set(v1).intersection(set(v2))

    def fetch_feature(version, version_features, feature):
        with open(f'data/base/{version}/rna.mat.csv') as d:
            reader = csv.reader(d)
            next(reader)
            feature_index = version_features.index(feature)

            for i in range(feature_index + 1):
                line = next(reader)
            return [float(x) for x in line]

    with open('data/processed/combined_rna.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(columns)

        first_rna = csv.reader(open('data/base/v1/rna.mat.csv', 'r'))
        next(first_rna)
        for feature in tqdm(v1):
            if not feature in common:
                continue
            line = next(first_rna)
            line = [float(x) for x in line]
            line = line + fetch_feature('v2', v2, feature)
            writer.writerow(line)


def combined(out, feature_names):
    feature_set = set(feature_names)
    data = []
    columns = []
    rows = list(json.load(open('data/processed/hiv.json', 'r'))
                ['HIV_Float'].keys())

    for file in [('data/base/v1/rna.mat/rna.mat.csv'), ('data/base/v1/motif.mat.csv'), ('data/base/v1/atac.mat/atac.csv')]:
        with open(file, 'r') as f:
            reader = csv.reader(f)
            next(reader)

            for row in tqdm(reader):
                feature = row[0]
                if feature in feature_set:
                    columns.append(feature)
                    data.append([float(x) for x in row[1:]])

    df = pd.DataFrame(data).T
    df.columns = columns

    # Reorder dataset such that it remains sorted by descending abs spearman coefficient
    df = df[feature_names]
    df.to_csv(out)
