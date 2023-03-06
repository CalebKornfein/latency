import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from sklearn.metrics import roc_curve, auc, f1_score, mean_squared_error, zero_one_loss, precision_recall_curve


def overlay_roc(clf, X_train, X_test, y_train, y_test, type='combined', figpath=None, title=None):
    y_train_pred = clf.predict_proba(X_train)[:, 1]
    y_test_pred = clf.predict_proba(X_test)[:, 1]
    train_fpr, train_tpr, train_thresholds = roc_curve(y_train, y_train_pred)
    train_aucs = auc(train_fpr, train_tpr)

    test_fpr, test_tpr, test_thresholds = roc_curve(y_test, y_test_pred)
    test_aucs = auc(test_fpr, test_tpr)

    fig, ax = plt.subplots()
    plt.style.use('classic')
    ax.grid(True)
    ax.set_xlabel("False Positive Rate", fontsize=16)
    ax.set_ylabel("True Positive Rate", fontsize=16)
    if title is None:
        plt.title(f"ROC Curve", fontsize=16)
    else:
        plt.title(title)
    plt.plot(train_fpr, train_tpr, linestyle="--", marker=".",
             markersize=16, label=f'Train, AUC = {round(train_aucs, 3)}')
    plt.plot(test_fpr, test_tpr, linestyle="--", marker=".",
             markersize=16, label=f'Test, AUC = {round(test_aucs, 3)}')
    plt.plot([0, 1], [0, 1], linestyle="--", c="k")
    plt.legend(fontsize=16)
    fig.savefig(os.path.join(os.getcwd(), figpath,
                f"xgb_{type}_roc.png"), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(os.getcwd(), figpath,
                f"xgb_{type}_roc.pdf"), bbox_inches='tight', dpi=300)


def overlay_pr(clf, X_train, X_test, y_train, y_test, type='combined', figpath=None, title=None):
    kPropPositiveLabel = 0.09751564205362064

    y_train_pred = clf.predict_proba(X_train)[:, 1]
    y_test_pred = clf.predict_proba(X_test)[:, 1]

    train_pr, train_re, train_thresholds = precision_recall_curve(
        y_train, y_train_pred)
    test_pr, test_re, test_thresholds = precision_recall_curve(
        y_test, y_test_pred)

    train_auc = auc(train_re, train_pr)
    test_auc = auc(test_re, test_pr)

    fig, ax = plt.subplots(figsize=(7, 6))
    plt.style.use('classic')
    ax.grid(True)

    ax.set_xlabel("Recall", fontsize=16)
    ax.set_ylabel("Precision", fontsize=16)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if title is None:
        plt.title(f"Precision-Recall Curve", fontsize=16)
    else:
        plt.title(title)
    plt.plot(train_re, train_pr, linestyle="--", marker=".",
             markersize=16, label=f'Train, AUPR = {round(train_auc, 3)}')
    plt.plot(test_re, test_pr, linestyle="--", marker=".",
             markersize=16, label=f'Test, AUPR = {round(test_auc, 3)}')
    plt.plot([0, 1], [kPropPositiveLabel, kPropPositiveLabel],
             linestyle="--", c="k")
    plt.legend(fontsize=16)
    if figpath:
        fig.savefig(os.path.join(os.getcwd(), figpath,
                    f"xgboost_{type}_pr.png"), bbox_inches='tight', dpi=300)
        fig.savefig(os.path.join(os.getcwd(), figpath,
                    f"xgboost_{type}_pr.png"), bbox_inches='tight', dpi=300)


def overlay_roc_temp(clf, X_train, X_test, y_train, y_test, X_new_train, y_new_test, title=None):
    y_train_pred = clf.predict_proba(X_train)[:, 1]
    y_test_pred = clf.predict_proba(X_test)[:, 1]
    y_new_test_pred = clf.predict_proba(X_new_train)[:, 1]

    train_fpr, train_tpr, train_thresholds = roc_curve(y_train, y_train_pred)
    train_aucs = auc(train_fpr, train_tpr)

    test_fpr, test_tpr, test_thresholds = roc_curve(y_test, y_test_pred)
    test_aucs = auc(test_fpr, test_tpr)

    new_test_fpr, new_test_tpr, new_test_thresholds = roc_curve(
        y_new_test, y_new_test_pred)
    new_test_aucs = auc(new_test_fpr, new_test_tpr)

    plt.figure()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    if title is None:
        plt.title(f"ROC Curve")
    else:
        plt.title(title)
    plt.plot(train_fpr, train_tpr, linestyle="--", marker=".",
             markersize=15, label=f'Train, AUC = {round(train_aucs, 2)}')
    plt.plot(test_fpr, test_tpr, linestyle="--", marker=".",
             markersize=15, label=f'Test, AUC = {round(test_aucs, 2)}')
    plt.plot(new_test_fpr, new_test_tpr, linestyle="--", marker=".",
             markersize=15, label=f'Test on New Data, AUC = {round(new_test_aucs, 3)}')
    plt.plot([0, 1], [0, 1], linestyle="--", c="k")
    plt.legend()
