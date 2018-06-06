"""Decision tree demo"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn import tree
from sklearn.metrics import mean_absolute_error


def get_decision_tree(data):
    """
    Get a simple demo model of decision tree
    :return: sklearn.tree.decisionTreeRegressor model
    """
    x_train, x_test, y_train, y_test = data

    model = tree.DecisionTreeRegressor()
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    print("Mean absolute value is " + str(mean_absolute_error(y_test, preds)))
    return model
