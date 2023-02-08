import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
plt.style.use('matplotlib_sty.txt')


def aggregate(l):
    l = l.dropna()
    features = []
    for row in l:
        if row == ' ' or row == 'nan':
            continue
        print(row)
        row_features = [x.strip() for x in row.split('|')]
        features = features + row_features
    return Counter(features)


def top_n_features(l, n=5):
    d = aggregate(l)
    for key in d.keys():
        d[key] = d[key] / len(l)

    def colors_from_values(values, palette_name):
        palette = sns.color_palette(palette_name, len(values))
        return np.array(palette).take(range(0, n, -1), axis=0)

    x, y = [], []
    for k, v in d.most_common(n):
        x.append(k)
        y.append(v)

    sns.set(style="white", color_codes=True)
    sns.barplot(x=x, y=y, palette="autumn")
    plt.title(f'Most commonly occurring features')
    plt.ylabel('Proportion occurrence')


def pr_curve(train_precision, train_recall, test_precision, test_recall):
    kPropPositiveLabel = 0.09751564205362064

    sns.set(style="whitegrid", color_codes=True)
    plt.figure()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve of GOSDT models")

    plt.scatter(train_recall, train_precision, marker=".",
                size=4, label='Train Prediction', c="blue")
    plt.scatter(test_recall, test_precision, marker=".",
                size=4, label='Test Prediction', c="orange")

    plt.plot([0, 1], [0.09751564205362064, 0.09751564205362064],
             linestyle="-", c="k")
    plt.legend()


def style_pr_curve(train_precision, train_recall, test_precision, test_recall):
    kPropPositiveLabel = 0.09751564205362064
    fig, ax = plt.subplots()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    plt.scatter(train_recall, train_precision, marker=".",
                s=40, label='Train Prediction')
    plt.scatter(test_recall, test_precision, marker=".",
                s=40, label='Test Prediction')

    plt.plot([0, 1], [0.09751564205362064, 0.09751564205362064],
             linestyle="dotted", c="k")
    plt.legend()
