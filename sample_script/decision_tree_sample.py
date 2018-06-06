"""This is a sample of how the toolkit works on decision tree"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models_demo.tree import decision_tree


def decision_tree_sample():
    """
    This is a example of decision tree visualization
    :return:
    """
    print("\033[0;33mGet a trained decision tree model\033[0m")
    model = decision_tree.get_tree_regressor()

    # TODO: Model visualization

    # TODO: Explanation of the model(SHAP)
