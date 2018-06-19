"""Sample script for decision tress"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from sklearn.model_selection import train_test_split
from sklearn import tree
import pydotplus
import pandas as pd
import matplotlib.pyplot as plt

from sample_script.tree import decision_tree
from SHAP.tree import TreeExplainer


def try_decision_tree():
    """
    The sample script of visualizing decision tree (scikit-learn)
    :return:
    """
    # Get data
    path = os.path.abspath(".")

    # data = pd.read_csv(path+'/sample_script/tree/simple_train.csv')

    # Simple test data, to be deleted
    data = pd.read_csv(path+'/sample_script/tree/simple_train copy2.csv')
    # delete end.

    data, predictors = preprocess_data(data)

    # Get model
    model = decision_tree.get_decision_tree(data)

    # Output model graph
    graph = pydotplus.graph_from_dot_data(tree.export_graphviz(model, max_depth=5,
                                                               out_file=None, filled=True,
                                                               feature_names=predictors))
    graph.write_png("./sample_script/tree/decision_tree.png")

    explainer = TreeExplainer(model).shap_values(x=data[0])
    print(explainer[0, :])

    plt.bar(range(len(predictors)), explainer[0, :-1], tick_label=predictors)
    plt.savefig("shap.png")


def preprocess_data(data):
    """
    Preprocess the input data to train a model
    :param data: pandas DataFrame
    :return: [x_train, x_test, y_train, y_test]
    """

    # Process data
    y_data = data.SalePrice
    # predictors = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath',
    #               'BedroomAbvGr', 'TotRmsAbvGrd']

    # Simple test data,
    predictors = ['LotArea', 'YearBuilt', '1stFlrSF']
    # delete end.

    x_data = data[predictors]

    print(x_data)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                        train_size=0.7, test_size=0.3,
                                                        random_state=0)

    return [x_train, x_test, y_train, y_test], predictors


if __name__ == "__main__":
    try_decision_tree()
