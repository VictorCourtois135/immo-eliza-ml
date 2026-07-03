from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
from .model_template import ModelTemplate

class MyLinearRegression(ModelTemplate):
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_test_real = np.expm1(y_test)

        steps = [
            ("scaler", StandardScaler()),
            ("linreg", LinearRegression())
        ]
        self.model = Pipeline(steps)
        self.model.fit(self.X_train, self.y_train)