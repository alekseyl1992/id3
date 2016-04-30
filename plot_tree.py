# coding=utf-8
import uuid

from graphviz import Digraph


def plot_tree_recursive(dot, tree, edge_name=None, parent_node=None):
    if tree.has_key('value'):
        # leaf
        label = tree.get('value', '').decode('utf8')
    else:
        label = tree.get('field', '').decode('utf8')

    name = str(uuid.uuid4()).decode('utf8')
    dot.node(name, label)

    if parent_node:
        edge_name = edge_name.decode('utf8')
        dot.edge(parent_node, name, label=edge_name)

    if tree.has_key('children'):
        for edge_name, child in tree['children'].items():
            plot_tree_recursive(dot, child, edge_name, name)


def plot_tree(tree, name):
    dot = Digraph(comment=name, encoding='utf8', format='svg')
    plot_tree_recursive(dot, tree)

    dot.render('plot.gv', view=True)
