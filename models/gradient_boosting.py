from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error,mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
from .model_template import ModelTemplate


class MyGradientBoost(ModelTemplate):
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_test_real = np.expm1(y_test)

        self.model = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.7,
            min_samples_leaf=20,
            random_state=42
        )
        self.model.fit(self.X_train, self.y_train)
