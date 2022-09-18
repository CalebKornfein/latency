import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc, RocCurveDisplay, f1_score, log_loss, mean_squared_error, zero_one_loss
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