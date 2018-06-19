"""
This module is an implement of SHAP (SHapley Additive exPlanation) values on
tree ensembles. See "https://arxiv.org/abs/1802.03888" for more details.

Citation: Lundberg S M, Erion G G, Lee S I. Consistent Individualized Feature Attribution
for Tree Ensembles[J]. arXiv preprint arXiv:1802.03888, 2018.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class TreeExplainer:
    """
    Class of SHAP explainer for tree ensembles.
    Support sklearn.decisionTreeRegressor
    """

    def __init__(self, model, **kwargs):
        self.tree = Tree(model.tree_)

        # Preallocate space for the unique path data
        depth = self.tree.max_depth + 2
        s = (depth * (depth + 1)) // 2
        self.feature_indexes = np.zeros(s, dtype=np.int32)
        self.zero_fractions = np.zeros(s, dtype=np.float64)
        self.one_fractions = np.zeros(s, dtype=np.float64)
        self.pweights = np.zeros(s, dtype=np.float64)

    def shap_values(self, x):
        """
        Initiation for calculate the SHAP values for each features.
        :param x:
        :return:
        """
        if str(type(x)).endswith("DataFrame'>") or str(type(x)).endswith("Series'>"):
            x = x.values

        self.n_outputs = self.tree.values.shape[1]

        # Only one example
        if len(x.shape) == 1:
            values = np.zeros((x.shape[0] + 1, self.n_outputs))
            x_missing = np.zeros(x.shape[0], dtype=np.bool)

            self.tree_shap(self.tree, x, x_missing, values)

            if self.n_outputs == 1:
                return values[:, 0]
            else:
                return [values[:, i] for i in range(self.n_outputs)]

        # Other cases
        else:
            values = np.zeros((x.shape[0], x.shape[1] + 1, self.n_outputs))
            x_missing = np.zeros(x.shape[1], dtype=np.bool)

            for i in range(x.shape[0]):
                self.tree_shap(self.tree, x[i, :], x_missing, values[i, :, :])

            if self.n_outputs == 1:
                return values[:, :, 0]
            else:
                return [values[:, :, i] for i in range(self.n_outputs)]

    def tree_shap(self, tree, x, x_missing, values, condition=0, condition_feature=0):
        """
        The algorithm to calculate the SHAP values
        :param tree:
        :param x:
        :param x_missing:
        :param values:
        :return:
        """

        if condition == 0:
            values[-1, :] += tree.values[0, :]

        # Start the recursive algorithm
        tree_shap_recursive(tree.children_left, tree.children_right,
                            tree.children_default, tree.feature,
                            tree.threshold, tree.values, tree.node_sample_weight,
                            x, x_missing, values, 0, 0, self.feature_indexes,
                            self.zero_fractions, self.one_fractions, self.pweights,
                            1, 1, -1, condition, condition_feature, 1)


def tree_shap_recursive(children_left, children_right, children_default,
                        features, thresholds, tree_values,
                        node_sample_weight, x, x_missing, values, node_index, unique_depth,
                        parent_feature_indexes, parent_zero_fractions, parent_one_fractions,
                        parent_pweights, parent_zero_fraction, parent_one_fraction,
                        parent_feature_index, condition, condition_feature, condition_fraction):
    """
    Recursive algorithm to calculate tree shap
    :param children_leftm:
    :param children_right:
    :param features:
    :param threshold:
    :param tree_values:
    :param node_sample_weight:
    :param x:
    :param x_missing:
    :param values:
    :param node_index:
    :param unique_depth:
    :return:
    """
    # Stop when there's no weight coming
    if condition_fraction == 0:
        return

    # extend the unique path
    feature_indexes = parent_feature_indexes[unique_depth + 1:]
    feature_indexes[:unique_depth + 1] = parent_feature_indexes[:unique_depth + 1]
    zero_fractions = parent_zero_fractions[unique_depth + 1:]
    zero_fractions[:unique_depth + 1] = parent_zero_fractions[:unique_depth + 1]
    one_fractions = parent_one_fractions[unique_depth + 1:]
    one_fractions[:unique_depth + 1] = parent_one_fractions[:unique_depth + 1]
    pweights = parent_pweights[unique_depth + 1:]
    pweights[:unique_depth + 1] = parent_pweights[:unique_depth + 1]

    if condition == 0 or condition_feature != parent_feature_index:
        extend(
            feature_indexes, zero_fractions, one_fractions, pweights,
            unique_depth, parent_zero_fraction, parent_one_fraction, parent_feature_index
        )

    split_index = features[node_index]

    # leaf node
    if children_right[node_index] == -1:
        for i in range(1, unique_depth + 1):
            w = unwound_path_sum(feature_indexes, zero_fractions,
                                 one_fractions, pweights, unique_depth, i)

            values[feature_indexes[i], :] += w * (one_fractions[i] - zero_fractions[i]) \
                                             * tree_values[node_index, :] * condition_fraction

    # internal node
    else:
        # find which branch is "hot" (meaning x would follow it)
        hot_index = 0
        cleft = children_left[node_index]
        cright = children_right[node_index]
        if x_missing[split_index] == 1:
            hot_index = children_default[node_index]
        elif x[split_index] < thresholds[node_index]:
            hot_index = cleft
        else:
            hot_index = cright
        cold_index = (cright if hot_index == cleft else cleft)
        w = node_sample_weight[node_index]
        hot_zero_fraction = node_sample_weight[hot_index] / w
        cold_zero_fraction = node_sample_weight[cold_index] / w
        incoming_zero_fraction = 1.
        incoming_one_fraction = 1.

        # see if we have already split on this feature,
        # if so we undo that split so we can redo it for this node
        path_index = 0
        while path_index <= unique_depth:
            if feature_indexes[path_index] == split_index:
                break
            path_index += 1

        if path_index != unique_depth + 1:
            incoming_zero_fraction = zero_fractions[path_index]
            incoming_one_fraction = one_fractions[path_index]
            unwind(feature_indexes, zero_fractions, one_fractions,
                   pweights, unique_depth, path_index)
            unique_depth -= 1

        # divide up the condition_fraction among the recursive calls
        hot_condition_fraction = condition_fraction
        cold_condition_fraction = condition_fraction
        if condition > 0 and split_index == condition_feature:
            cold_condition_fraction = 0.
            unique_depth -= 1
        elif condition < 0 and split_index == condition_feature:
            hot_condition_fraction *= hot_zero_fraction
            cold_condition_fraction *= cold_zero_fraction
            unique_depth -= 1

        tree_shap_recursive(
            children_left, children_right, children_default, features,
            thresholds, tree_values, node_sample_weight,
            x, x_missing, values, hot_index, unique_depth + 1,
            feature_indexes, zero_fractions, one_fractions, pweights,
            hot_zero_fraction * incoming_zero_fraction, incoming_one_fraction,
            split_index, condition, condition_feature, hot_condition_fraction
        )

        tree_shap_recursive(
            children_left, children_right, children_default, features,
            thresholds, tree_values, node_sample_weight,
            x, x_missing, values, cold_index, unique_depth + 1,
            feature_indexes, zero_fractions, one_fractions, pweights,
            cold_zero_fraction * incoming_zero_fraction, 0,
            split_index, condition, condition_feature, cold_condition_fraction
        )


def extend(feature_indexes, zero_fractions, one_fractions, pweights,
           unique_depth, zero_fraction, one_fraction, feature_index):
    feature_indexes[unique_depth] = feature_index
    zero_fractions[unique_depth] = zero_fraction
    one_fractions[unique_depth] = one_fraction
    if unique_depth == 0:
        pweights[unique_depth] = 1.
    else:
        pweights[unique_depth] = 0.

    for i in range(unique_depth - 1, -1, -1):
        pweights[i + 1] += one_fraction * pweights[i] * (i + 1.) / (unique_depth + 1.)
        pweights[i] = zero_fraction * pweights[i] * (unique_depth - i) / (unique_depth + 1.)


def unwind(feature_indexes, zero_fractions, one_fractions, pweights,
           unique_depth, path_index):
    one_fraction = one_fractions[path_index]
    zero_fraction = zero_fractions[path_index]
    next_one_portion = pweights[unique_depth]

    for i in range(unique_depth - 1, -1, -1):
        if one_fraction != 0.:
            tmp = pweights[i]
            pweights[i] = next_one_portion * (unique_depth + 1.) / ((i + 1.) * one_fraction)
            next_one_portion = tmp - pweights[i] * zero_fraction * (unique_depth - i) / (unique_depth + 1.)
        else:
            pweights[i] = (pweights[i] * (unique_depth + 1)) / (zero_fraction * (unique_depth - i))

    for i in range(path_index, unique_depth):
        feature_indexes[i] = feature_indexes[i + 1]
        zero_fractions[i] = zero_fractions[i + 1]
        one_fractions[i] = one_fractions[i + 1]


def unwound_path_sum(feature_indexes, zero_fractions, one_fractions, pweights, unique_depth, path_index):
    one_fraction = one_fractions[path_index]
    zero_fraction = zero_fractions[path_index]
    next_one_portion = pweights[unique_depth]
    total = 0

    for i in range(unique_depth - 1, -1, -1):
        if one_fraction != 0.:
            tmp = next_one_portion * (unique_depth + 1.) / ((i + 1.) * one_fraction)
            total += tmp
            next_one_portion = pweights[i] - tmp * zero_fraction * ((unique_depth - i) / (unique_depth + 1.))
        else:
            total += (pweights[i] / zero_fraction) / ((unique_depth - i) / (unique_depth + 1.))

    return total


class Tree:
    """
    Class of Tree for SHAP explainer
    Support sklearn.decisionTreeRegressor
    """

    def __init__(self, model):
        self.children_left = model.children_left
        self.children_right = model.children_right
        self.children_default = self.children_left
        self.feature = model.feature
        self.threshold = model.threshold.astype(np.float64)
        self.values = model.value[:, 0, :]

        self.node_sample_weight = model.weighted_n_node_samples
        self.max_depth = model.max_depth
