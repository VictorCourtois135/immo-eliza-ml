from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error,mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
from .model_template import ModelTemplate


class MyRidge(ModelTemplate):
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_test_real = np.expm1(y_test)

        steps_ridge = [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=100))
        ]
        self.model = Pipeline(steps_ridge)
        self.model.fit(self.X_train, self.y_train)
