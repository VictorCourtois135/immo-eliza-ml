from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error,mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
from .model_template import ModelTemplate

class MyDecisionTree(ModelTemplate):
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_test_real = np.expm1(y_test)

        self.model = DecisionTreeRegressor(
            max_depth=9,
            min_samples_leaf=10,
            min_samples_split=10,
            random_state=42
        )
        self.model.fit(self.X_train, self.y_train)
