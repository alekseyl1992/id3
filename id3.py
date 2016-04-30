# coding=utf-8

import numpy as np


def split_set(data_set):
    return data_set.iloc[:, :-1], data_set.iloc[:, -1:]


def entropy(data, field):
    classes = data[field].unique()

    e = 0.0
    for cls in classes:
        subset = data.loc[data[field] == cls]
        p = float(len(subset)) / len(data)
        e += p * np.log2(p)

    return -e


def gain(data, field):
    classes = data[field].unique()

    entropy_sum = 0.0
    for cls in classes:
        subset = data.loc[data[field] == cls]
        p = float(len(subset)) / len(data)
        entropy_sum += p * entropy(subset, field)

    return entropy(data, field) - entropy_sum


def split_by_field(data, field):
    classes = data[field].unique()

    subsets = []

    for cls in classes:
        subset = data.loc[data[field] == cls]
        subsets.append(subset)

    return subsets, classes


def most_common_value(data, target_field):
    subsets, values = split_by_field(data, target_field)
    lens = [len(subset) for subset in subsets]
    max_len_id = np.argmax(lens)

    return values[max_len_id]


def fit(data, tree, target_field, fields):
    fields = fields[:]  # clone

    if len(data) == 0:
        return

    # if all the same
    val = data[target_field].iloc[0]
    if (data[target_field] == val).all(0):
        tree['value'] = val
        return

    max_gain = 0.0
    max_gain_field = None

    for field in fields:
        g = gain(data, field)
        if g >= max_gain:
            max_gain = g
            max_gain_field = field

    subsets, values = split_by_field(data, max_gain_field)

    print('=============')
    print('Split by {} (gain: {})'.format(max_gain_field, max_gain))
    for subset in subsets:
        print('--------------')
        print(subset)
        print('--------------')
    print('=============')

    tree['field'] = max_gain_field
    children = tree['children'] = {}

    fields.remove(max_gain_field)

    # if len(fields) == 0:
    #     tree['value'] = most_common_value(data, target_field)
    #     return

    if len(subsets) > 1:
        for subset, value in zip(subsets, values):
            child = children[value] = {}
            fit(subset, child, target_field, fields)
    else:
        tree['value'] = most_common_value(data, target_field)
        return


def predict(tree, x):
    value = tree.get('value', None)
    if value:
        return value
    else:
        field = tree['field']
        for value, node in tree['children'].items():
            if x[field] == value:
                return predict(node, x)


def accuracy(tree, data, target_field):
    ys = data[target_field]

    ps = []
    for d in data.iterrows():
        p = predict(tree, d[1])
        ps.append(p)

    right_guesses = (ps == ys).sum()

    return float(right_guesses) / len(data), right_guesses, len(data)



