import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
import os
import json
from gosdt_sweep import name2config, config2name


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


def top_n_features(model_features, figsdir, n=7):
    # Some models are "trivial" and have no features,
    # only predicting one class the entire time. Drop these.
    model_features = model_features.dropna()
    d = aggregate(model_features)
    for key in d.keys():
        d[key] = d[key] / len(model_features)

    x, y = [], []
    for k, v in d.most_common(n):
        x.append(k)
        y.append(v)

    fig, ax = plt.subplots(figsize=(9.2, 5))
    sns.set(rc={'figure.figsize': (6, 5)})
    sns.set(style="white", color_codes=True)
    sns.barplot(ax=ax, x=x, y=y, palette="autumn")
    plt.title(f'Commonly occurring features', fontsize=20)
    plt.ylabel('Proportion occurrence', fontsize=20)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)

    fig.savefig(os.path.join(os.getcwd(), figsdir,
                "gosdt_top_features.pdf"), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(os.getcwd(), figsdir,
                "gosdt_top_features.png"), bbox_inches='tight', dpi=300)


def style_pr_curve(train_precision, train_recall, test_precision, test_recall, figsdir):

    fig, ax = plt.subplots(figsize=(6, 5))
    plt.style.use('classic')
    ax.grid(True, linestyle='dotted')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("Recall", fontsize=16)
    ax.set_ylabel("Precision", fontsize=16)
    ax.set_title("Precision-recall curve of GOSDT models", fontsize=16)
    ax.tick_params(right=False, top=False)

    plt.scatter(train_recall, train_precision, s=20,
                c="orange", label='Train Prediction')
    plt.scatter(test_recall, test_precision, s=20,
                c="blue", label='Test Prediction')

    plt.legend(fontsize=16)

    fig.savefig(os.path.join(os.getcwd(), figsdir, "gosdt_pr.pdf"),
                bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(os.getcwd(), figsdir, "gosdt_pr.png"),
                bbox_inches='tight', dpi=300)


def style_pr_curve_by_person(train_precision, train_recall, test_precision_p1, test_recall_p1, test_precision_p2, test_recall_p2, figsdir):

    fig, ax = plt.subplots(figsize=(6, 5))
    plt.style.use('classic')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    ax.grid(True, linestyle='dotted')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("Recall", fontsize=16)
    ax.set_ylabel("Precision", fontsize=16)
    ax.set_title(
        "Precision-recall curve of GOSDT models by subject", fontsize=16, pad=14)
    ax.tick_params(right=False, top=False)

    plt.scatter(train_recall, train_precision,
                alpha=0.7, s=20, c="blue", label='Train Prediction')
    plt.scatter(test_recall_p1, test_precision_p1,
                alpha=0.7, s=20, c="red", label='Test Prediction Subject 1')
    plt.scatter(test_recall_p2, test_precision_p2,
                alpha=0.7, s=20, c="gold", label='Test Prediction Subject 2')

    plt.legend(fontsize=16)

    fig.savefig(os.path.join(os.getcwd(), figsdir,
                "gosdt_pr_by_subject.pdf"), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(os.getcwd(), figsdir,
                "gosdt_pr_by_subject.png"), bbox_inches='tight', dpi=300)


def main():
    kFigsDir = 'graphics/v3/gosdt'
    kSweepPath = 'gosdt_sweep_results/v3/sweep.csv'
    kGosdtDir = 'gosdt_sweep_results/v3'
    os.makedirs(kFigsDir, exist_ok=True)

    df = pd.read_csv(kSweepPath)
    model_features = df['model_features']
    train_precision, train_recall, test_precision, test_recall = df['train_precision'].values, df[
        'train_recall'].values, df['test_precision'].values, df['test_recall'].values
    test_precision_p1, test_recall_p1 = df['test_precision_p1'].values, df['test_recall_p1'].values
    test_precision_p2, test_recall_p2 = df['test_precision_p2'].values, df['test_recall_p2'].values

    top_n_features(model_features, kFigsDir, n=7)
    style_pr_curve(train_precision, train_recall,
                   test_precision, test_recall, kFigsDir)
    style_pr_curve_by_person(train_precision, train_recall, test_precision_p1,
                             test_recall_p1, test_precision_p2, test_recall_p2, kFigsDir)


if __name__ == "__main__":
    main()
