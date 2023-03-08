import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import os
import xgboost as xgb
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.utils import shuffle
from helpers import fetch_person_index, load_data, split_data_by_person
import json
import pandas as pd


def overlay_roc(clf, X_train, X_test, y_train, y_test, type='combined', figpath=None, title=None):
    y_train_pred = clf.predict_proba(X_train)[:, 1]
    y_test_pred = clf.predict_proba(X_test)[:, 1]
    train_fpr, train_tpr, train_thresholds = roc_curve(y_train, y_train_pred)
    train_aucs = auc(train_fpr, train_tpr)

    test_fpr, test_tpr, test_thresholds = roc_curve(y_test, y_test_pred)
    test_aucs = auc(test_fpr, test_tpr)

    fig, ax = plt.subplots()
    plt.style.use('classic')
    ax.grid(True, linestyle='dotted')
    ax.set_xlabel("False Positive Rate", fontsize=16)
    ax.set_ylabel("True Positive Rate", fontsize=16)

    if title is None:
        plt.title(f"ROC Curve", fontsize=16)
    else:
        plt.title(title)
    plt.plot(train_fpr, train_tpr, linestyle="--", marker=".",
             markersize=10, c="orange", label=f'Train, AUC = {round(train_aucs, 3)}')
    plt.plot(test_fpr, test_tpr, linestyle="--", marker=".",
             markersize=10, c="blue", label=f'Test, AUC = {round(test_aucs, 3)}')
    plt.plot([0, 1], [0, 1], linestyle="--", c="k")
    plt.legend(fontsize=16, loc='lower right')
    if figpath:
        name = f"xgb_{type}_roc"
        fig.savefig(os.path.join(os.getcwd(), figpath, name +
                    ".png"), bbox_inches='tight', dpi=300)
        fig.savefig(os.path.join(os.getcwd(), figpath, name +
                    ".pdf"), bbox_inches='tight', dpi=300)


def overlay_pr(clf, X_train, X_test, y_train, y_test, type='combined', figpath=None, title=None):

    y_train_pred = clf.predict_proba(X_train)[:, 1]
    y_test_pred = clf.predict_proba(X_test)[:, 1]

    train_pr, train_re, train_thresholds = precision_recall_curve(
        y_train, y_train_pred)
    test_pr, test_re, test_thresholds = precision_recall_curve(
        y_test, y_test_pred)

    train_auc = auc(train_re, train_pr)
    test_auc = auc(test_re, test_pr)

    fig, ax = plt.subplots()
    plt.style.use('classic')
    ax.grid(True, linestyle='dotted')

    ax.set_xlabel("Recall", fontsize=16)
    ax.set_ylabel("Precision", fontsize=16)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if title is None:
        plt.title(f"Precision-Recall Curve", fontsize=16)
    else:
        plt.title(title)
    plt.plot(train_re, train_pr, linestyle="--", marker=".",
             markersize=10, c="orange", label=f'Train, AUPR = {round(train_auc, 3)}')
    plt.plot(test_re, test_pr, linestyle="--", marker=".",
             markersize=10, c="blue", label=f'Test, AUPR = {round(test_auc, 3)}')
    plt.legend(fontsize=16)

    if figpath:
        name = f"xgboost_{type}_pr"
        fig.savefig(os.path.join(os.getcwd(), figpath, name +
                    ".png"), bbox_inches='tight', dpi=300)
        fig.savefig(os.path.join(os.getcwd(), figpath, name +
                    ".pdf"), bbox_inches='tight', dpi=300)


def fetch_aupr(clf, X_train, X_test, y_train, y_test):
    y_hat_train = clf.predict_proba(X_train)[:, 1]
    y_hat_test = clf.predict_proba(X_test)[:, 1]
    train_pr, train_re, train_thresholds = precision_recall_curve(
        y_train, y_hat_train)
    test_pr, test_re, test_thresholds = precision_recall_curve(
        y_test, y_hat_test)
    train_aupr = auc(train_re, train_pr)
    test_aupr = auc(test_re, test_pr)
    return train_aupr, test_aupr


def permutation_test(clf, X_train, X_test, y_train, y_test, results_path, n_shuffles=5):
    # 1 - establish baseline
    train_aupr, test_aupr = fetch_aupr(clf, X_train, X_test, y_train, y_test)

    # 2 - iterate through the features and permute the features
    X_train_iter, X_test_iter = X_train.copy(), X_test.copy()
    train_feature_importances = []
    test_feature_importances = []

    for feature in tqdm(X_train.columns):
        # average the feature importance over n_shuffles
        train_feature_aupr = []
        test_feature_aupr = []

        # Hold a temporary copy of the original.
        temp_train, temp_test = X_train_iter[feature], X_test_iter[feature]

        for _ in range(n_shuffles):
            X_train_iter[feature] = shuffle(X_train_iter[feature].values, random_state=0)
            X_test_iter[feature] = shuffle(X_test_iter[feature].values, random_state=0)

            # Measure aupr
            train_aupr_iter, test_aupr_iter = fetch_aupr(clf, X_train_iter, X_test_iter, y_train, y_test)
            train_feature_aupr.append(train_aupr_iter)
            test_feature_aupr.append(test_aupr_iter)
        
        # Put temporary copy back
        X_train_iter[feature], X_test_iter[feature] = temp_train, temp_test
        
        # Calculate feature importance
        train_feature_importance =  train_aupr - sum(train_feature_aupr) / len(train_feature_aupr)
        test_feature_importance =  test_aupr - sum(test_feature_aupr) / len(test_feature_aupr)
        train_feature_importances.append((feature, train_feature_importance))
        test_feature_importances.append((feature, test_feature_importance))
    
    sorted_train = sorted(train_feature_importances, key=lambda x: x[1], reverse=True)
    sorted_test = sorted(test_feature_importances, key=lambda x: x[1], reverse=True)

    df_train  = pd.DataFrame(sorted_train)
    df_train.columns = ['Feature', 'Train Importance']
    df_train.to_csv(os.path.join(results_path, 'train_importances.csv'))
    df_test = pd.DataFrame(sorted_test)
    df_test.columns = ['Feature', 'Test Importance']
    df_test.to_csv(os.path.join(results_path, 'test_importances.csv'))


def stem_plot(clf, X_train, X_test, y_train, y_test, figpath=None):
    y_hat_train = clf.predict_proba(X_train)[:, 1]
    y_hat_test = clf.predict_proba(X_test)[:, 1]

    ranks = sorted(zip(y_hat_test, y_test), reverse=True)
    stem_locations = [i + 1 for i,
                      (hat, real) in enumerate(ranks) if real == 1]

    fig, ax = plt.subplots(figsize=(7, 1.75))
    plt.style.use('classic')
    ax.grid(False)
    for stem_location in stem_locations:
        plt.plot([stem_location, stem_location], [
                 0, 1], color='blue', linewidth=0.03)

    n_test = len(y_hat_test)
    ax.set_xticks([1, 0.25 * n_test, 0.5 * n_test, 0.75 * n_test, n_test])
    ax.set_xticklabels([1, '25%', '50%', '75%', n_test], fontsize=16)
    ax.set_xlim([1, len(ranks) + 1])
    ax.set_ylim([0, 1])
    ax.tick_params(left=False, labelleft=False, bottom=True, labelbottom=True,
                   right=False, labelright=False, top=False, labeltop=False)

    total = len(stem_locations)
    cdf_x = stem_locations
    cdf_y = [i / total for i in range(total)]

    if figpath:
        fig.savefig(os.path.join(figpath, "xgb_stem.png"),
                    bbox_inches='tight', dpi=300)
        fig.savefig(os.path.join(figpath, "xgb_stem.pdf"),
                    bbox_inches='tight', dpi=300)


def cdf_plot(clf, X_train, X_test, y_train, y_test, figpath=None):
    y_hat_train = clf.predict_proba(X_train)[:, 1]
    y_hat_test = clf.predict_proba(X_test)[:, 1]

    ranks = sorted(zip(y_hat_test, y_test), reverse=True)
    total_ranks = len(ranks)
    cdf_x = [i for i, (hat, real) in enumerate(ranks) if real == 1]
    total = len(cdf_x)
    cdf_y = [i / total for i in range(total)]

    fig, ax = plt.subplots()
    plt.style.use('classic')
    ax.grid(True)
    plt.plot(cdf_x, cdf_y, color='blue', linewidth=3)

    ax.set_xlim([0, len(ranks) - 1])
    ax.set_xticks([0.10 * total_ranks, 0.25 * total_ranks,
                  0.5 * total_ranks, 0.75 * total_ranks])
    ax.set_xticklabels(['10%', '25%', '50%', '75%'], fontsize=16)
    ax.set_xlabel('Predicted rank', fontsize=16)

    ax.set_ylim([0, 1])
    ax.set_yticks([0.10, 0.25, 0.5, 0.75])
    ax.set_yticklabels(['10%', '25%', '50%', '75%'], fontsize=16)
    ax.set_ylabel('Percentage recovered', fontsize=16)

    ax.tick_params(left=True, labelleft=True, bottom=True, labelbottom=True)

    if figpath:
        fig.savefig(os.path.join(os.getcwd(), figpath,
                    "xgb_cdf.png"), bbox_inches='tight', dpi=300)
        fig.savefig(os.path.join(os.getcwd(), figpath,
                    "xgb_cdf.pdf"), bbox_inches='tight', dpi=300)


def comparison_pr(clf_combined, clf_p1, clf_p2, X_train, X_test, y_train, y_test, figpath=None):

    p1, p2 = split_data_by_person(X_train, X_test, y_train, y_test)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 5))
    fig.tight_layout()
    plt.style.use('classic')
    plt.suptitle(f"Precision-Recall Curves of XGBoost models", fontsize=21)
    plt.subplots_adjust(top=0.85)

    # FIRST THE COMBINED PR CURVE
    overall_y_hat_test_p1 = clf_combined.predict_proba(p1.X_test)[:, 1]
    overall_y_hat_test_p2 = clf_combined.predict_proba(p2.X_test)[:, 1]

    overall_test_pr_p1, overall_test_re_p1, test_thresholds = precision_recall_curve(
        p1.y_test, overall_y_hat_test_p1)
    overall_test_pr_p2, overall_test_re_p2, test_thresholds = precision_recall_curve(
        p2.y_test, overall_y_hat_test_p2)

    overall_test_p1_aupr = auc(overall_test_re_p1, overall_test_pr_p1)
    overall_test_p2_aupr = auc(overall_test_re_p2, overall_test_pr_p2)

    ax1.grid(True)
    ax1.set_xlabel("Recall", fontsize=16)
    ax1.set_ylabel("Precision", fontsize=16)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.tick_params(right=False, top=False)
    ax1.set_title("Trained on both donors", fontsize=16)

    ax1.plot(overall_test_re_p1, overall_test_pr_p1, linestyle="--", marker=".",
             markersize=8, c='orange', label=f'Donor 1, Test AUPR = {round(overall_test_p1_aupr, 3)}')
    ax1.plot(overall_test_re_p2, overall_test_pr_p2, linestyle="--", marker=".",
             markersize=8, c='blue', label=f'Donor 2, Test AUPR = {round(overall_test_p2_aupr, 3)}')
    ax1.legend(fontsize=16, loc='upper center')

    # THEN TRAINING ON P1

    p1_y_hat_test_p1 = clf_p1.predict_proba(p1.X_test)[:, 1]
    p1_y_hat_test_p2 = clf_p1.predict_proba(p2.X_test)[:, 1]

    p1_test_pr_p1, p1_test_re_p1, test_thresholds = precision_recall_curve(
        p1.y_test, p1_y_hat_test_p1)
    p1_test_pr_p2, p1_test_re_p2, test_thresholds = precision_recall_curve(
        p2.y_test, p1_y_hat_test_p2)

    p1_test_p1_aupr = auc(p1_test_re_p1, p1_test_pr_p1)
    p1_test_p2_aupr = auc(p1_test_re_p2, p1_test_pr_p2)

    ax2.grid(True)
    ax2.set_xlabel("Recall", fontsize=16)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.tick_params(bottom=True, labelbottom=True,
                    labelleft=False, left=False, right=False, top=False)
    ax2.set_title("Trained only on donor 1", fontsize=16)

    ax2.plot(p1_test_re_p1, p1_test_pr_p1, linestyle="--", marker=".",
             markersize=8, c='orange', label=f'Donor 1, Test AUPR = {round(p1_test_p1_aupr, 3)}')
    ax2.plot(p1_test_re_p2, p1_test_pr_p2, linestyle="--", marker=".",
             markersize=8, c='blue', label=f'Donor 2, Test AUPR = {round(p1_test_p2_aupr, 3)}')
    ax2.legend(fontsize=16, loc='upper center')

    # THEN TRAINING ON P2

    p2_y_hat_test_p1 = clf_p2.predict_proba(p1.X_test)[:, 1]
    p2_y_hat_test_p2 = clf_p2.predict_proba(p2.X_test)[:, 1]

    p2_test_pr_p1, p2_test_re_p1, test_thresholds = precision_recall_curve(
        p1.y_test, p2_y_hat_test_p1)
    p2_test_pr_p2, p2_test_re_p2, test_thresholds = precision_recall_curve(
        p2.y_test, p2_y_hat_test_p2)

    p2_test_p1_aupr = auc(p2_test_re_p1, p2_test_pr_p1)
    p2_test_p2_aupr = auc(p2_test_re_p2, p2_test_pr_p2)

    ax3.grid(True)
    ax3.set_xlabel("Recall", fontsize=16)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.tick_params(bottom=True, labelbottom=True,
                    labelleft=False, left=False, right=False, top=False)
    ax3.set_title("Trained only on donor 2", fontsize=16)

    ax3.plot(p2_test_re_p1, p2_test_pr_p1, linestyle="--", marker=".",
             markersize=8, c='orange', label=f'Donor 1, Test AUPR = {round(p2_test_p1_aupr, 3)}')
    ax3.plot(p2_test_re_p2, p2_test_pr_p2, linestyle="--", marker=".",
             markersize=8, c='blue', label=f'Donor 2, Test AUPR = {round(p2_test_p2_aupr, 3)}')
    ax3.legend(fontsize=16, loc='upper center')

    # Save the figure
    if figpath:
        fig.savefig(os.path.join(os.getcwd(), figpath,
                    "comparison_xgb_pr.png"), bbox_inches='tight', dpi=300)
        fig.savefig(os.path.join(os.getcwd(), figpath,
                    "comparison_xgb_pr.pdf"), bbox_inches='tight', dpi=300)



def comparison_roc(clf_combined, clf_p1, clf_p2, X_train, X_test, y_train, y_test, train_curve=True, figpath=None):

    p1, p2 = split_data_by_person(X_train, X_test, y_train, y_test)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 5))
    fig.tight_layout()
    plt.style.use('classic')
    plt.suptitle(f"ROC Curves of XGBoost models", fontsize=21)
    plt.subplots_adjust(top=0.85)

    # FIRST THE COMBINED ROC CURVE
    overall_y_hat_test_p1 = clf_combined.predict_proba(p1.X_test)[:, 1]
    overall_y_hat_test_p2 = clf_combined.predict_proba(p2.X_test)[:, 1]

    overall_test_fpr_p1, overall_test_tpr_p1, test_thresholds = roc_curve(
        p1.y_test, overall_y_hat_test_p1)
    overall_test_fpr_p2, overall_test_tpr_p2, test_thresholds = roc_curve(
        p2.y_test, overall_y_hat_test_p2)

    overall_test_p1_auc = auc(overall_test_fpr_p1, overall_test_tpr_p1)
    overall_test_p2_auc = auc(overall_test_fpr_p2, overall_test_tpr_p2)

    ax1.grid(True)
    ax1.set_xlabel("False Positive Rate", fontsize=16)
    ax1.set_ylabel("True Positive Rate", fontsize=16)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.tick_params(right=False, top=False)
    ax1.set_title("Trained on both donors", fontsize=16)

    ax1.plot(overall_test_fpr_p1, overall_test_tpr_p1, linestyle="--", marker=".",
             markersize=8, c='orange', label=f'Donor 1, Test AUC = {round(overall_test_p1_auc, 3)}')
    ax1.plot(overall_test_fpr_p2, overall_test_tpr_p2, linestyle="--", marker=".",
             markersize=8, c='blue', label=f'Donor 2, Test AUC = {round(overall_test_p2_auc, 3)}')
    
    if train_curve:
        overall_y_hat_train = clf_combined.predict_proba(X_train)[:, 1]
        overall_train_fpr, overall_test_tpr, test_thresholds = roc_curve(
            y_train, overall_y_hat_train)
        overall_train_auc = auc(overall_train_fpr, overall_test_tpr)
        ax1.plot(overall_train_fpr, overall_test_tpr, linestyle="--", marker=".",
            markersize=8, c='green', label=f'Train AUC = {round(overall_train_auc, 3)}')

    ax1.legend(fontsize=16, loc='lower right')

    # THEN TRAINING ON P1

    p1_y_hat_test_p1 = clf_p1.predict_proba(p1.X_test)[:, 1]
    p1_y_hat_test_p2 = clf_p1.predict_proba(p2.X_test)[:, 1]

    p1_test_fpr_p1, p1_test_tpr_p1, test_thresholds = roc_curve(
        p1.y_test, p1_y_hat_test_p1)
    p1_test_fpr_p2, p1_test_tpr_p2, test_thresholds = roc_curve(
        p2.y_test, p1_y_hat_test_p2)

    p1_test_p1_auc = auc(p1_test_fpr_p1, p1_test_tpr_p1)
    p1_test_p2_auc = auc(p1_test_fpr_p2, p1_test_tpr_p2)

    ax2.grid(True)
    ax2.set_xlabel("False Positive Rate", fontsize=16)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.tick_params(bottom=True, labelbottom=True,
                    labelleft=False, left=False, right=False, top=False)
    ax2.set_title("Trained only on donor 1", fontsize=16)

    ax2.plot(p1_test_fpr_p1, p1_test_tpr_p1, linestyle="--", marker=".",
             markersize=8, c='orange', label=f'Donor 1, Test AUC = {round(p1_test_p1_auc, 3)}')
    ax2.plot(p1_test_fpr_p2, p1_test_tpr_p2, linestyle="--", marker=".",
             markersize=8, c='blue', label=f'Donor 2, Test AUC = {round(p1_test_p2_auc, 3)}')
    
    if train_curve:
        p1_y_hat_train_p1 = clf_combined.predict_proba(p1.X_train)[:, 1]
        p1_train_fpr_p1, p1_train_tpr_p1, test_thresholds = roc_curve(
            p1.y_train, p1_y_hat_train_p1)
        overall_train_auc_p1 = auc(p1_train_fpr_p1, p1_train_tpr_p1)
        ax2.plot(p1_train_fpr_p1, p1_train_tpr_p1, linestyle="--", marker=".",
            markersize=8, c='green', label=f'Train AUC = {round(overall_train_auc_p1, 3)}')

    ax2.legend(fontsize=16, loc='lower right')

    # THEN TRAINING ON P2

    p2_y_hat_test_p1 = clf_p2.predict_proba(p1.X_test)[:, 1]
    p2_y_hat_test_p2 = clf_p2.predict_proba(p2.X_test)[:, 1]

    p2_test_fpr_p1, p2_test_tpr_p1, test_thresholds = roc_curve(
        p1.y_test, p2_y_hat_test_p1)
    p2_test_fpr_p2, p2_test_tpr_p2, test_thresholds = roc_curve(
        p2.y_test, p2_y_hat_test_p2)

    p2_test_p1_auc = auc(p2_test_fpr_p1, p2_test_tpr_p1)
    p2_test_p2_auc = auc(p2_test_fpr_p2, p2_test_tpr_p2)

    ax3.grid(True)
    ax3.set_xlabel("False positive Rate", fontsize=16)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.tick_params(bottom=True, labelbottom=True,
                    labelleft=False, left=False, right=False, top=False)
    ax3.set_title("Trained only on donor 2", fontsize=16)

    ax3.plot(p2_test_fpr_p1, p2_test_tpr_p1, linestyle="--", marker=".",
             markersize=8, c='orange', label=f'Donor 1, Test AUC = {round(p2_test_p1_auc, 3)}')
    ax3.plot(p2_test_fpr_p2, p2_test_tpr_p2, linestyle="--", marker=".",
             markersize=8, c='blue', label=f'Donor 2, Test AUC = {round(p2_test_p2_auc, 3)}')
    
    if train_curve:
        p2_y_hat_train_p2 = clf_combined.predict_proba(p2.X_train)[:, 1]
        p2_train_fpr_p2, p2_train_tpr_p2, test_thresholds = roc_curve(
            p2.y_train, p2_y_hat_train_p2)
        overall_train_auc_p2 = auc(p2_train_fpr_p2, p2_train_tpr_p2)
        ax3.plot(p2_train_fpr_p2, p2_train_tpr_p2, linestyle="--", marker=".",
            markersize=8, c='green', label=f'Train AUC = {round(overall_train_auc_p2, 3)}')

    ax3.legend(fontsize=16, loc='lower right')

    # Save the figure
    if figpath:
        name = "comparison_xgb_roc"
        if train_curve:
            name += "_with_train_curve"

        fig.savefig(os.path.join(os.getcwd(), figpath,
                    name + ".png"), bbox_inches='tight', dpi=300)
        fig.savefig(os.path.join(os.getcwd(), figpath,
                    name + ".pdf"), bbox_inches='tight', dpi=300)


def pr(y, y_hat):
    tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()
    precision = tp / (tp + fp + 0.0000001)
    recall = tp / (tp + fn + 0.0000001)
    return precision, recall


def results_by_person(clf, X_train, X_test, y_train, y_test):
    y_hat_train = clf.predict(X_train)
    y_hat_test = clf.predict(X_test)

    # PR overall.
    train_precision_overall, train_recall_overall = pr(y_train, y_hat_train)
    test_precision_overall, test_recall_overall = pr(y_test, y_hat_test)

    train_index_p1, test_index_p1, train_index_p2, test_index_p2 = fetch_person_index(
        y_train.index, y_test.index)

    # PR results for person 1.
    train_precision_p1, train_recall_p1 = pr(
        y_train[train_index_p1], y_hat_train[train_index_p1])
    test_precision_p1, test_recall_p1 = pr(
        y_test[test_index_p1], y_hat_test[test_index_p1])

    # PR results for person 2.
    train_precision_p2, train_recall_p2 = pr(
        y_train[train_index_p2], y_hat_train[train_index_p2])
    test_precision_p2, test_recall_p2 = pr(
        y_test[test_index_p2], y_hat_test[test_index_p2])


def main():
    kFigsDir = f'graphics/v3/xgboost'
    out = f'xgb_results/v3'

    os.makedirs(kFigsDir, exist_ok=True)

    X_train, X_test, y_train, y_test = load_data(n=1000, label_type='10')

    # Load the various XGBoost models
    clf_combined = xgb.XGBClassifier()
    clf_combined.load_model(os.path.join(out, "xgboost_model_overall.json"))

    clf_p1 = xgb.XGBClassifier()
    clf_p1.load_model(os.path.join(out, "xgboost_model_p1.json"))

    clf_p2 = xgb.XGBClassifier()
    clf_p2.load_model(os.path.join(out, "xgboost_model_p2.json"))

    # Note - permutation test takes a long time to run -- uncomment to run
    #permutation_test(clf_combined, X_train, X_test, y_train, y_test, results_path=out, n_shuffles=5)

    # Create the graphs!
    overlay_pr(clf_combined, X_train, X_test, y_train,
               y_test, type='overall', figpath=kFigsDir)
    overlay_roc(clf_combined, X_train, X_test, y_train,
                y_test, type='overall', figpath=kFigsDir)
    cdf_plot(clf_combined, X_train, X_test, y_train, y_test, figpath=kFigsDir)
    stem_plot(clf_combined, X_train, X_test, y_train, y_test, figpath=kFigsDir)
    comparison_pr(clf_combined, clf_p1, clf_p2, X_train,
                  X_test, y_train, y_test, figpath=kFigsDir)
    comparison_roc(clf_combined, clf_p1, clf_p2, X_train,
                   X_test, y_train, y_test, train_curve=False, figpath=kFigsDir)
    comparison_roc(clf_combined, clf_p1, clf_p2, X_train,
                   X_test, y_train, y_test, train_curve=True, figpath=kFigsDir)


if __name__ == "__main__":
    main()
