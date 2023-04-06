import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
import os

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


def top_n_features(model_features, figsdir, n=7, results_path=None):
    # Some models are "trivial" and have no features,
    # only predicting one class the entire time. Drop these.
    model_features = model_features.dropna()
    d = aggregate(model_features)
    for key in d.keys():
        d[key] = d[key] / len(model_features)

    x, y = [], []
    for k, v in d.most_common(len(d)):
        x.append(k)
        y.append(v)

    if results_path:
        df = pd.DataFrame([(x, y) for (x, y) in zip(x, y)])
        df.columns = ['Feature', 'Proportion of models with feature']
        df.to_csv(os.path.join(results_path, 'feature_proportions.csv'))


    fig, ax = plt.subplots(figsize=(9.2, 5))
    sns.set(rc={'figure.figsize': (6, 5)})
    sns.set(style="white", color_codes=True)
    sns.barplot(ax=ax, x=x[:n], y=y[:n], palette="autumn")
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
        "Precision-recall curve of GOSDT models by donor", fontsize=16, pad=14)
    ax.tick_params(right=False, top=False)

    plt.scatter(train_recall, train_precision,
                alpha=0.7, s=20, c="blue", label='Train Prediction')
    plt.scatter(test_recall_p1, test_precision_p1,
                alpha=0.7, s=20, c="red", label='Test Prediction Donor 1')
    plt.scatter(test_recall_p2, test_precision_p2,
                alpha=0.7, s=20, c="gold", label='Test Prediction Donor 2')

    plt.legend(fontsize=16)

    fig.savefig(os.path.join(os.getcwd(), figsdir,
                "gosdt_pr_by_donor.pdf"), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(os.getcwd(), figsdir,
                "gosdt_pr_by_donor.png"), bbox_inches='tight', dpi=300)

def style_roc_by_person(train_fpr, train_tpr, test_fpr_p1, test_tpr_p1, test_fpr_p2, test_tpr_p2, type='combined', figpath=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    plt.style.use('classic')
    ax.grid(True, linestyle='dotted')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("False Positive Rate", fontsize=16)
    ax.set_ylabel("True Positive Rate", fontsize=16)

    plt.title(f"ROC curve of GOSDT models by donor", fontsize=16, pad=14)

    plt.scatter(train_fpr, train_tpr, alpha=0.7, s=20, c="blue", label=f'Train')
    plt.scatter(test_fpr_p1, test_tpr_p1, alpha=0.7, s=20, c="red", label=f'Donor 1 Test')
    plt.scatter(test_fpr_p2, test_tpr_p2, alpha=0.7, s=20, c="gold", label=f'Donor 2 Test')
    plt.plot([0, 1], [0, 1], linestyle="--", c="k")
    plt.legend(fontsize=16, loc='lower right')
    if figpath:
        name = f"gosdt_{type}_roc"
        fig.savefig(os.path.join(os.getcwd(), figpath, name +
                    ".png"), bbox_inches='tight', dpi=300)
        fig.savefig(os.path.join(os.getcwd(), figpath, name +
                    ".pdf"), bbox_inches='tight', dpi=300)


def main():
    kFigsDir = 'graphics/v5/gosdt'
    kSweepPath = 'gosdt_sweep_results/v5/sweep.csv'
    kGosdtDir = 'gosdt_sweep_results/v5'
    os.makedirs(kFigsDir, exist_ok=True)

    df = pd.read_csv(kSweepPath)
    model_features = df['model_features']
    train_precision, train_recall, test_precision, test_recall = df['train_precision'].values, df[
        'train_recall'].values, df['test_precision'].values, df['test_recall'].values
    test_precision_p1, test_recall_p1 = df['test_precision_p1'].values, df['test_recall_p1'].values
    test_precision_p2, test_recall_p2 = df['test_precision_p2'].values, df['test_recall_p2'].values
    train_fpr, test_fpr_p1, test_fpr_p2, = df['train_fpr'], df['test_fpr_p1'], df['test_fpr_p2']

    top_n_features(model_features, kFigsDir, n=7, results_path=kGosdtDir)
    style_pr_curve(train_precision, train_recall,
                   test_precision, test_recall, kFigsDir)
    style_pr_curve_by_person(train_precision, train_recall, test_precision_p1,
                             test_recall_p1, test_precision_p2, test_recall_p2, kFigsDir)
    
    style_roc_by_person(train_fpr, train_recall, test_fpr_p1, test_recall_p1, test_fpr_p2, test_recall_p2, type='combined', figpath=kFigsDir)

if __name__ == "__main__":
    main()
