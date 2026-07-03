from sklearn.ensemble import  RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
from .model_template import ModelTemplate

class MyRandomForest(ModelTemplate):
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_test_real = np.expm1(y_test)

        self.model = RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            max_features=0.5,
            max_samples=0.9,
            oob_score=True,
            random_state=42
        )
        self.model.fit(self.X_train, self.y_train)

    def oob_score(self):
        """Métrique spécifique au Random Forest (bagging)."""
        return self.model.oob_score_

