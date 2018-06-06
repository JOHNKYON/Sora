"""Sample script for decision tress"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn import tree
import pydotplus
import pandas as pd

from sample_script.tree import decision_tree


def try_decision_tree():
    """
    The sample script of visualizing decision tree (scikit-learn)
    :return:
    """
    # Get data
    path = os.path.abspath(".")

    data = pd.read_csv(path+'/sample_script/tree/train.csv')

    data = preprocess_data(data)

    # Get model
    model = decision_tree.get_decision_tree(data)

    # Output model graph
    graph = pydotplus.graph_from_dot_data(tree.export_graphviz(model, max_depth=5, out_file=None))
    graph.write_png("./sample_script/tree/decision_tree.png")


def preprocess_data(data):
    """
    Preprocess the input data to train a model
    :param data: pandas DataFrame
    :return: [x_train, x_test, y_train, y_test]
    """
    col_range = list(range(5))

    # Process data
    y_data = data.SalePrice
    x_data = data.drop(['SalePrice'], axis=1).iloc[:, col_range]

    x_data = x_data.select_dtypes(exclude=['object'])
    print(x_data)

    # Impute data
    my_imputer = Imputer()

    col_with_missing = (col for col in x_data.columns
                        if x_data[col].isnull().any())

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                        train_size=0.7, test_size=0.3,
                                                        random_state=0)

    for col in col_with_missing:
        # x_train[col + '_was_missing'] = x_train[col].isnull()
        # x_test[col + '_was_missing'] = x_test[col].isnull()
        x_train.loc[:, (col + '_was_missing')] = x_train.loc[:, col].isnull()
        x_test.loc[:, (col + '_was_missing')] = x_test.loc[:, col].isnull()

    x_train = my_imputer.fit_transform(x_train)
    x_test = my_imputer.fit_transform(x_test)

    return [x_train, x_test, y_train, y_test]


if __name__ == "__main__":
    try_decision_tree()
