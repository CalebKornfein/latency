from random import randrange
from scipy import stats
import pandas as pd
import csv
import numpy as np
from tqdm import tqdm
import json
import os
from sklearn.model_selection import train_test_split
from helpers import load_data, add_label, top_10_percent


def scrape_hiv(versions=['combined']):
    # HIV comes from the rna.mat.csv dataset. This helper finds the appropriate
    # column and writes it its own file.
    for v in versions:
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
                    print("Found and wrote HIV!")
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


def create_merged_motif():
    # Observation names, ex: "AAACAGCCAAACTAAG-2"
    columns = from_txt('data/base/v1/motif.mat.colnames.txt', numerical=False) + \
        from_txt('data/base/v2/motif.mat.colnames.txt', numerical=False)

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

    # Transpose back the DataFrame
    combined = combined.T

    with open('data/base/combined/rna.mat.rownames.txt', 'w') as f:
        for line in tqdm(list(combined.index)):
            f.write(f"{line}\n")

    with open('data/base/combined/rna.mat.colnames.txt', 'w') as f:
        for line in tqdm(list(combined.columns)):
            f.write(f"{line}\n")

    combined.to_csv('data/base/combined/rna.mat.csv', index=False)


def calculate_spearman(version, train_indices):
    # Used for either JUST the V1 data or the V2 data, but not the combined data.
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
    # Used for the combined data, not for V1 or V2 alone.
    # This is because the combined data does not include ATAC.
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
    df.to_csv(f'data/processed/combined/spearman_combined.csv', index=False)
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
    # What type of feature it is
    feature_types = list(df.dataset)

    def include(feature, feature_type=None):
        # Filter out mitochondrial features which start with MT-
        if feature.find('MT-') == 0:
            return False
        if 'HIV' in feature:
            return False
        # Filter to a specific type if type is specified
        if (type != None) and (feature_type != type):
            return False
        return True

    return [feature for feature, feature_type in zip(features, feature_types) if include(feature, feature_type)][:n]


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
    # # Make the out directory for processed metrics
    # os.makedirs('data/processed/combined', exist_ok=True)

    # # 1) Generate train and test indices
    # n_samples_v1 = 61708
    # n_samples_v2 = 30831
    # train_indices, test_indices = fetch_train_test_indices(
    #     n_samples_v1 + n_samples_v2, save=True)
    # print("Generated train and test indices")

    # # 2) Generate merged dataset
    # # note -- ATAC not merged
    # # note note -- this can take a while due to large I/O and memory need!
    # print("Beginning to merge Motif and RNA datasets (this can take a while)")
    # create_merged_rna()
    # create_merged_motif()
    # print("Merged the Motif and Rna datasets")

    # # 3) Find HIV values to calculate spearman against
    # scrape_hiv()
    # print("Scraped HIV labels")

    # # 4) Calculate spearman coefficients for all features against HIV using the train data
    # calculate_combined_spearman(train_indices)
    # print("Calculated Spearman")

    # 5) Select features (ranked by absolute spearman correlation) for the top 2000 overall
    #    and write these features to a csv.
    #
    #    Here, the Mitochondrial features starting with MT- are filtered out.

    # Then, write the datasets to a csv

    for version in ['combined']:
        overall = select_top_features(2000, version=version)
        train, test = load_train_test_indices(version=version)
        split_labels(train, test, version=version)

        datasets = [('overall', overall)]
        for data_type, feature_names in datasets:
            train_out = os.path.join(
                'data', 'processed', version, data_type + "_train.csv")
            test_out = os.path.join(
                'data', 'processed', version, data_type + "_test.csv")
            split_train_test(train_out, test_out, feature_names, version)
    print("Generated dataset of top features and split it into training and testing sets")
    # For HIV add a label corresponding to whether the observations are in the top 10% of HIV values.
    add_label('10', top_10_percent)
    print("Added a label corresponding to the top 10 percent of HIV values")
