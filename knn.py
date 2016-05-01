# coding=utf-8
import itertools

import math
import numpy as np
import pandas as pd


def dist(f1, f2):
    f1, f2 = float(f1), float(f2)
    return np.sqrt((f1 + f2) / (f1 * f2))


def map_by_index(df, fun):
    keys = df.keys()
    for col in keys:
        for row in keys:
            if col != row:
                df.set_value(col, row, fun(col, row, df.get_value(col, row)))


def filter_anomalies(data, filter_field, k, delta):
    freqs = data[filter_field].value_counts()
    print pd.DataFrame(freqs).to_html(classes='table')

    matrix = pd.DataFrame(data=None, index=freqs.keys(), columns=freqs.keys())

    # calc distances
    map_by_index(matrix, lambda col, row, x: dist(freqs[col], freqs[row]))
    print matrix.to_html(classes='table')

    # matrix os sorted by distance
    # so k nearest objects is [0:k] for each row
    # ..but we need to skip diagonal elements
    k_dists = {}
    for i, key in enumerate(freqs.keys()):
        if i < k:
            id = k
        else:
            id = k - 1

        k_dists[key] = matrix.iloc[i, id]

    # update distances
    map_by_index(matrix, lambda col, row, x: np.max([x, k_dists[col]]))
    print matrix.to_html(classes='table')

    # calc local density
    lrd = pd.Series(index=freqs.keys())
    for i, key in enumerate(freqs.keys()):
        if i < k:
            id = k + 1
        else:
            id = k

        s = float(matrix.iloc[0:id, i].sum())
        lrd[key] = 1. / (s / k)

    print pd.DataFrame(lrd).to_html(classes='table')

    # calc lof
    lof = pd.Series(index=freqs.keys())
    for i, key in enumerate(freqs.keys()):
        row = matrix[key]
        row = row.sort_values(ascending=True)[:k]
        names = row.keys()

        s = 0.0
        for name in names:
            s += lrd[name] / lrd[key]

        lof[key] = s / k

    print pd.DataFrame(lof).to_html(classes='table')

    # find anomaly values
    outliers = lof[(lof > 1 + delta) | (lof < 1 - delta)]
    print(pd.DataFrame(outliers).to_html(classes='table'))

    mask = data[filter_field].isin(outliers.keys())

    print data[mask].to_html(classes='table')

    return data[~mask]


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    train_set = pd.read_csv('data/close1.csv', sep=';')
    filtered_ts = filter_anomalies(train_set, 'Тип объекта', 3, 0.15)

