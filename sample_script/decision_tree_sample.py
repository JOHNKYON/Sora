"""This is a sample of how the toolkit works on decision tree"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models_demo.tree import decision_tree

print("\033[0;33mGet a trained decision tree model\033[0m")

decision_tree.get_tree_regressor()
