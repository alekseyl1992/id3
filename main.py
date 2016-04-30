# coding=utf-8
import json

import pandas as pd

import id3
from plot_tree import plot_tree


def main():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    train_set = pd.read_csv('data/close1.csv', sep=';')
    test_set = pd.read_csv('data/close1_test2.csv', sep=';')

    tree = {}
    target_field = 'Доступность'
    fields = list(train_set.columns.values)
    fields.remove(target_field)

    id3.fit(train_set, tree, target_field, fields)

    json_str = json.dumps(tree)
    print(json_str)

    print('Train Set Accuracy: {} ({}/{})'.format(*id3.accuracy(tree, train_set, target_field)))
    print('Test Set Accuracy: {} ({}/{})'.format(*id3.accuracy(tree, test_set, target_field)))

    plot_tree(tree, u'Decision Tree')

main()