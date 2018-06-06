"""Model demos of decision tree"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from sklearn import tree
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
import pandas as pd


def get_tree_regressor():
    """
    Build the decision tree model trained on provided dataset in "models_demo/data
    :return: model (sklearn.tree.DecisionTreeRegressor)
    """""

    path = os.path.abspath('..')

    melb_data = pd.read_csv(path+'/models_demo/data/melb_data.csv')

    melb_target = melb_data.Price
    melb_predictors = melb_data.drop(['Price'], axis=1)

    # Remove numeric predictors
    melb_numeric_predictors = melb_predictors.select_dtypes(exclude=['object'])

    # Split data into training and validation data for both predictos and target
    train_x, test_x, train_y, test_y = train_test_split(melb_numeric_predictors, melb_target,
                                                        random_state=0)

    # Imputating data
    my_imputer = Imputer()
    imputed_x_train = my_imputer.fit_transform(train_x)
    imputed_x_test = my_imputer.transform(test_x)

    # Get model
    model = tree.DecisionTreeRegressor()

    # Train model
    model.fit(imputed_x_train, train_y)

    # Test model
    pred_y = model.predict(imputed_x_test)

    print("The mean absolute error of this model on test data-set is")
    print(mean_absolute_error(test_y, pred_y))

    return model
