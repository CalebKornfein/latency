import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc, f1_score, mean_squared_error, zero_one_loss, precision_recall_curve
sns.set_style("whitegrid")
sns.set_context("talk")

'''
API for plotting standardized metrics for reporting. Pretty simple stuff but just to keep us all
on the same page as well as to not waste time coding them up ourselves. 

NOTE: For all models, this takes in predicted scores, NOT just the classification. 
This means for sklearn models you need to call model.predict_probaba(X).
For pytorch models you need to pass in the full softmax output.
'''

def binary_metrics(y_true, y_pred, paramdict=None, title=None, label=None, output_text=False, filepath="/metric_outputs/", binary=False, plot=True):
    '''
    :param y_true: an (Nx1) vector of the true labels for each sample.
    :param y_pred: an (Nx1) vector of output probabilities for the POSITIVE class, which in our case
        is influential.
    :param output_text: a boolean if you want to output the f1, macrof1, and auc scores to a textfile
    :param filepath: the filepath to which the output text will go
    :return: nothing, just generates plots
    '''
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    if binary:
        metric = zero_one_loss(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

    else:
        metric = mean_squared_error(y_true, y_pred)
        f1 = f1_score(y_true, np.array([y_pred[j] >= 0.5 for j in range(len(y_pred))], dtype="int"))

    # print("Metric: (zero one or MSE): {}".format(metric))
    # f1 = f1_score(y_true, np.array([y_pred[j] >= 0.5 for j in range(len(y_pred))], dtype="int"))
    aucs = auc(fpr, tpr)
    #acc = sum((y_pred >= 0.5) == y_true) / len(y_true)

    if output_text:
        f = open(filepath+"binaryclass.txt", "a")
        print(f"------{title}------", file=f)
        if paramdict:
            print(paramdict, file=f)
        print(f"ROC-AUC: {aucs}", file=f)
        print(f"Zero-One or MSE: {metric}", file=f)
        print(f"Accuracy: {acc}", file=f)
        print("----------------", file=f)
        f.close()
    if plot:
        # plt.figure(figsize=(10, 6))
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        if title is None:
            plt.title(f"ROC for Influential vs Incidential")
        else:
            plt.title(f"ROC for {title} || Auc = {round(aucs, 3)}")
        plt.plot(fpr, tpr, linestyle="--", marker=".", markersize=15,label="{} AUC: {:0.4}, F1: {:0.4}".format(label, aucs, f1))
        plt.plot([0, 1], [0, 1], linestyle="--", c="k")

    return aucs, f1

def overlay_roc(clf, X_train, X_test, y_train, y_test, title=None):
    y_train_pred = clf.predict_proba(X_train)[:,1]
    y_test_pred = clf.predict_proba(X_test)[:,1]
    train_fpr, train_tpr, train_thresholds = roc_curve(y_train, y_train_pred)
    train_aucs = auc(train_fpr, train_tpr)

    test_fpr, test_tpr, test_thresholds = roc_curve(y_test, y_test_pred)
    test_aucs = auc(test_fpr, test_tpr)

    plt.figure()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    if title is None:
        plt.title(f"ROC Curve")
    else:
        plt.title(title)
    plt.plot(train_fpr, train_tpr,linestyle="--", marker=".", markersize=15, label = f'Train, AUC = {round(train_aucs, 3)}')
    plt.plot(test_fpr, test_tpr,linestyle="--", marker=".", markersize=15, label = f'Train, AUC = {round(test_aucs, 3)}')
    plt.plot([0, 1], [0, 1], linestyle="--", c="k")
    plt.legend()

def overlay_pr(clf, X_train, X_test, y_train, y_test, title=None):
    y_train_pred = clf.predict_proba(X_train)[:,1]
    y_test_pred = clf.predict_proba(X_test)[:,1]

    train_pr, train_re, train_thresholds = precision_recall_curve(y_train, y_train_pred)
    test_pr, test_re, test_thresholds = precision_recall_curve(y_test, y_test_pred)

    train_auc = auc(train_re, train_pr)
    test_auc = auc(test_re, test_pr)

    plt.figure()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    if title is None:
        plt.title(f"Precision-Recall Curve")
    else:
        plt.title(title)
    plt.plot(train_re, train_pr, linestyle="--", marker=".", markersize=15, label = f'Train, AUC = {round(train_auc, 3)}')
    plt.plot(test_re, test_pr, linestyle="--", marker=".", markersize=15, label = f'Test, AUC = {round(test_auc, 3)}')
    plt.plot([0, 1], [0, 1], linestyle="--", c="k")
    plt.legend()


def overlay_roc_temp(clf, X_train, X_test, y_train, y_test, X_new_train, y_new_test, title=None):
    y_train_pred = clf.predict_proba(X_train)[:,1]
    y_test_pred = clf.predict_proba(X_test)[:,1]
    y_new_test_pred = clf.predict_proba(X_new_train)[:,1]

    train_fpr, train_tpr, train_thresholds = roc_curve(y_train, y_train_pred)
    train_aucs = auc(train_fpr, train_tpr)

    test_fpr, test_tpr, test_thresholds = roc_curve(y_test, y_test_pred)
    test_aucs = auc(test_fpr, test_tpr)

    new_test_fpr, new_test_tpr, new_test_thresholds = roc_curve(y_new_test, y_new_test_pred)
    new_test_aucs = auc(new_test_fpr, new_test_tpr)

    plt.figure()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    if title is None:
        plt.title(f"ROC Curve")
    else:
        plt.title(title)
    plt.plot(train_fpr, train_tpr,linestyle="--", marker=".", markersize=15, label = f'Train, AUC = {round(train_aucs, 3)}')
    plt.plot(test_fpr, test_tpr,linestyle="--", marker=".", markersize=15, label = f'Test, AUC = {round(test_aucs, 3)}')
    plt.plot(new_test_fpr, new_test_tpr,linestyle="--", marker=".", markersize=15, label = f'Test on New Data, AUC = {round(new_test_aucs, 3)}')
    plt.plot([0, 1], [0, 1], linestyle="--", c="k")
    plt.legend()

def overlay_pr_temp(clf, X_train, X_test, y_train, y_test, X_new_train, y_new_test, title=None):
    y_train_pred = clf.predict_proba(X_train)[:,1]
    y_test_pred = clf.predict_proba(X_test)[:,1]
    y_new_test_pred = clf.predict_proba(X_new_train)[:,1]

    train_pr, train_re, train_thresholds = precision_recall_curve(y_train, y_train_pred)
    test_pr, test_re, test_thresholds = precision_recall_curve(y_test, y_test_pred)
    new_test_pr, new_test_re, new_test_thresholds = precision_recall_curve(y_new_test, y_new_test_pred)

    train_auc = auc(train_re, train_pr)
    test_auc = auc(test_re, test_pr)
    new_test_auc = auc(new_test_re, new_test_pr)

    plt.figure()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    if title is None:
        plt.title(f"Precision-Recall Curve")
    else:
        plt.title(title)
    plt.plot(train_re, train_pr, linestyle="--", marker=".", markersize=15, label = f'Train, AUC = {round(train_auc, 3)}')
    plt.plot(test_re, test_pr, linestyle="--", marker=".", markersize=15, label = f'Test, AUC = {round(test_auc, 3)}')
    plt.plot(new_test_re, new_test_pr, linestyle="--", marker=".", markersize=15, label = f'Test on New Data, AUC = {round(new_test_auc, 3)}')
    plt.plot([0, 1], [0, 1], linestyle="--", c="k")
    plt.legend()

def multiclass_plot(clf, X,  y):
    # Code from: https://towardsdatascience.com/multiclass-classification-evaluation-with-roc-curves-and-roc-auc-294fd4617e3a
    plt.figure(figsize = (12, 8))
    bins = 20
    classes = clf.classes_
    y_proba = clf.predict_proba(X)
    y_pred = clf.predict(X)

    roc_auc_ovr = {}
    for i in range(len(classes)):
        # Gets the class
        c = classes[i]
        
        # Prepares an auxiliar dataframe to help with the plots
        df_aux = X.copy()
        df_aux['class'] = [1 if y == c else 0 for y in y]
        df_aux['prob'] = y_proba[:,i]
        df_aux['pred'] = [1 if y == c else 0 for y in y_pred]
        df_aux = df_aux.reset_index(drop = True)
        
        # Plots the probability distribution for the class and the rest
        ax = plt.subplot(2, 4, i+1)
        sns.histplot(x = "prob", data = df_aux, hue = 'class', color = 'b', ax = ax, bins = bins)
        ax.set_title(c)
        ax.legend([f"Class: {c}", "Rest"])
        ax.set_xlabel(f"P(x = {c})")
        
        # Calculates the ROC Coordinates and plots the ROC Curves
        ax_bottom = plt.subplot(2, 4, i+5)
        fpr, tpr, thresholds = roc_curve(df_aux['class'], df_aux['prob'])
        aucs = auc(fpr, tpr)
        plt.plot(fpr, tpr, linestyle="--", marker=".", markersize=15,label=f"{i+1} AUC: {round(aucs, 2)}")
        ax_bottom.set_title(f"ROC for {i+1}, AUC: {round(aucs, 2)}")

    plt.tight_layout()