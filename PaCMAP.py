from helpers import load_data
import pandas as pd
import numpy as np
import pacmap
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def PaCMAP(level='Observation', label_type='HIV_Binary', data = 'Balanced'):
    if data == 'Balanced':
        X_train, X_test, y_train, y_test = load_data(top_n = 10, balanced=True, label_type=label_type)
        data_label = "balanced data"
    else:
        top_n = 50
        X_train, X_test, y_train, y_test = load_data(top_n = top_n, label_type=label_type)
        data_label = f"top {top_n} features"

    if level == 'Feature':
        annotations = list(X_train.columns)
        X_train = X_train.T

    transformer = MinMaxScaler()
    X_train_minmax = transformer.fit_transform(X_train.to_numpy())

    embedding = pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0)
    X_train_PaCMAP = embedding.fit_transform(X_train_minmax)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_title(f"{level} level PaCMAP using {data_label}")
    if level == 'Observation':
        ax.scatter(X_train_PaCMAP[:, 0], X_train_PaCMAP[:, 1], cmap="Spectral", c=y_train.to_numpy(), s=0.6)
    else:
        if data == 'Balanced':
            ax.scatter(X_train_PaCMAP[:, 0], X_train_PaCMAP[:, 1], s=0.6)
            for i, label in enumerate(annotations):
                ax.annotate(label, (X_train_PaCMAP[i, 0], X_train_PaCMAP[i,1]))
        else:
            ax.scatter(X_train_PaCMAP[:, 0], X_train_PaCMAP[:, 1], s=0.6)
            for i, label in enumerate(annotations):
                ax.annotate(f"{label} ({i})", (X_train_PaCMAP[i, 0], X_train_PaCMAP[i,1]))

if __name__ == "__main__":
    # Try running the various combinations of PaCMAP
    for level in ['Observation', 'Feature']:
        for data in ['Balanced', 'Overall']:
            PaCMAP(level=level, data=data)
    plt.show()
    